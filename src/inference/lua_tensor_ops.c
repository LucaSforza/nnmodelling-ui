#define _POSIX_C_SOURCE 200809L
#include "inference_internal.h"
#include "utils/utils.h"

#include <lua.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void nn_inference_raw_getfield(lua_State *state, int index, const char *key)
{
    index = lua_absindex(state, index);
    lua_pushstring(state, key);
    lua_rawget(state, index);
}

int nn_inference_tensor_read(lua_State *state, int index)
{
    if (!lua_istable(state, index)) return 0;
    nn_inference_raw_getfield(state, index, "shape");
    bool valid = lua_istable(state, -1);
    lua_pop(state, 1);
    nn_inference_raw_getfield(state, index, "dtype");
    valid = valid && lua_type(state, -1) == LUA_TSTRING;
    lua_pop(state, 1);
    return valid;
}

void nn_inference_tensor_push(lua_State *state, const char *dtype,
                        const char *const *dimensions, size_t count)
{
    lua_createtable(state, 0, 2);
    lua_pushstring(state, dtype);
    lua_setfield(state, -2, "dtype");
    lua_createtable(state, (int)count, 0);
    for (size_t i = 0; i < count; ++i) {
        char *end = NULL;
        long value = strtol(dimensions[i], &end, 10);
        if (end && *end == '\0') lua_pushinteger(state, value);
        else lua_pushstring(state, dimensions[i]);
        lua_seti(state, -2, (lua_Integer)i + 1);
    }
    lua_setfield(state, -2, "shape");
}

int nn_inference_push_error(lua_State *state, const char *message)
{
    lua_pushnil(state);
    lua_pushstring(state, message);
    return 2;
}

int nn_inference_tensor_rank(lua_State *state)
{
    if (!nn_inference_tensor_read(state, 1)) return nn_inference_push_error(state, "expected a tensor");
    nn_inference_raw_getfield(state, 1, "shape");
    lua_pushinteger(state, (lua_Integer)lua_rawlen(state, -1));
    return 1;
}

int nn_inference_tensor_dtype(lua_State *state)
{
    if (!nn_inference_tensor_read(state, 1)) return nn_inference_push_error(state, "expected a tensor");
    nn_inference_raw_getfield(state, 1, "dtype");
    return 1;
}

int nn_inference_dimension_index(lua_State *state, int tensor, lua_Integer dim, size_t *count)
{
    nn_inference_raw_getfield(state, tensor, "shape");
    *count = lua_rawlen(state, -1);
    lua_pop(state, 1);
    lua_Integer resolved = dim < 0 ? (lua_Integer)*count + dim + 1 : dim;
    if (resolved < 1 || (size_t)resolved > *count) return 0;
    return (int)resolved;
}

int nn_inference_tensor_dimension(lua_State *state)
{
    if (!nn_inference_tensor_read(state, 1) || !lua_isinteger(state, 2))
        return nn_inference_push_error(state, "expected a tensor and integer dimension");
    size_t count;
    int index = nn_inference_dimension_index(state, 1, lua_tointeger(state, 2), &count);
    if (!index) return nn_inference_push_error(state, "dimension is out of range");
    nn_inference_raw_getfield(state, 1, "shape");
    lua_rawgeti(state, -1, index);
    return 1;
}

bool nn_inference_read_shape(lua_State *state, int tensor, char ***dims, size_t *count,
                       bool temporary)
{
    nn_inference_raw_getfield(state, tensor, "shape");
    *count = lua_rawlen(state, -1);
    if (*count > 64) { lua_pop(state, 1); return false; }
    *dims = temporary ? nn_inference_lua_temp_calloc(state, *count ? *count : 1, sizeof(**dims))
                      : calloc(*count ? *count : 1, sizeof(**dims));
    if (!*dims) { nn_inference_note_host_allocation_failure(state); lua_pop(state, 1); return false; }
    for (size_t i = 0; i < *count; ++i) {
        lua_rawgeti(state, -1, (lua_Integer)i + 1);
        bool supported_type = lua_isinteger(state, -1) || lua_type(state, -1) == LUA_TSTRING;
        if (lua_isinteger(state, -1)) {
            char text[32]; snprintf(text, sizeof(text), "%lld", (long long)lua_tointeger(state, -1));
            (*dims)[i] = temporary ? nn_inference_lua_temp_copy(state, text) : nn_text_copy(text);
        } else if (lua_type(state, -1) == LUA_TSTRING)
            (*dims)[i] = temporary ? nn_inference_lua_temp_copy(state, lua_tostring(state, -1))
                                   : nn_text_copy(lua_tostring(state, -1));
        lua_pop(state, 1);
        if (!(*dims)[i]) {
            if (supported_type) nn_inference_note_host_allocation_failure(state);
            if (temporary) nn_inference_lua_temp_dimensions_free(state, *dims, i);
            else { for (size_t j = 0; j < i; ++j) free((*dims)[j]); free(*dims); }
            *dims = NULL; *count = 0;
            lua_pop(state, 1); return false;
        }
    }
    lua_pop(state, 1);
    return true;
}
int nn_inference_tensor_with_dimension(lua_State *state)
{
    if (!nn_inference_tensor_read(state, 1) || !lua_isinteger(state, 2) || !lua_isinteger(state, 3))
        return nn_inference_push_error(state, "expected tensor, integer dimension and integer size");
    char **dims = NULL; size_t count = 0;
    if (!nn_inference_read_shape(state, 1, &dims, &count, true)) return nn_inference_push_error(state, "invalid tensor shape");
    size_t unused;
    int index = nn_inference_dimension_index(state, 1, lua_tointeger(state, 2), &unused);
    if (!index || lua_tointeger(state, 3) < 1) {
        nn_inference_lua_temp_dimensions_free(state, dims, count);
        return nn_inference_push_error(state, "dimension or size is invalid");
    }
    char text[32]; snprintf(text, sizeof(text), "%lld", (long long)lua_tointeger(state, 3));
    nn_inference_lua_temp_free(state, dims[index - 1]); dims[index - 1] = nn_inference_lua_temp_copy(state, text);
    if (!dims[index - 1]) {
        nn_inference_note_host_allocation_failure(state);
        nn_inference_lua_temp_dimensions_free(state, dims, count);
        return nn_inference_push_error(state, "out of memory");
    }
    nn_inference_raw_getfield(state, 1, "dtype"); const char *dtype = lua_tostring(state, -1);
    nn_inference_tensor_push(state, dtype, (const char *const *)dims, count);
    nn_inference_lua_temp_dimensions_free(state, dims, count);
    return 1;
}

int nn_inference_tensor_create(lua_State *state)
{
    if (!lua_istable(state, 1) || lua_type(state, 2) != LUA_TSTRING)
        return nn_inference_push_error(state, "expected shape and dtype");
    size_t count = lua_rawlen(state, 1);
    if (count > 64) return nn_inference_push_error(state, "invalid tensor rank");
    char **dims = nn_inference_lua_temp_calloc(state, count, sizeof(*dims));
    if (!dims) return nn_inference_push_error(state, "out of memory");
    for (size_t i = 0; i < count; ++i) {
        lua_rawgeti(state, 1, (lua_Integer)i + 1);
        bool supported_type = lua_isinteger(state, -1) || lua_type(state, -1) == LUA_TSTRING;
        if (lua_isinteger(state, -1)) {
            char text[32]; snprintf(text, sizeof(text), "%lld", (long long)lua_tointeger(state, -1));
            dims[i] = nn_inference_lua_temp_copy(state, text);
        } else if (lua_type(state, -1) == LUA_TSTRING) dims[i] = nn_inference_lua_temp_copy(state, lua_tostring(state, -1));
        lua_pop(state, 1);
        if (!dims[i]) {
            if (supported_type) nn_inference_note_host_allocation_failure(state);
            nn_inference_lua_temp_dimensions_free(state, dims, count);
            return nn_inference_push_error(state, "out of memory");
        }
    }
    nn_inference_tensor_push(state, lua_tostring(state, 2), (const char *const *)dims, count);
    nn_inference_lua_temp_dimensions_free(state, dims, count);
    return 1;
}

int nn_inference_tensor_flatten(lua_State *state)
{
    if (!nn_inference_tensor_read(state, 1) || !lua_isinteger(state, 2) || !lua_isinteger(state, 3))
        return nn_inference_push_error(state, "expected tensor and integer dimension range");
    char **dims = NULL; size_t count = 0;
    if (!nn_inference_read_shape(state, 1, &dims, &count, true)) return nn_inference_push_error(state, "invalid tensor shape");
    lua_Integer start = lua_tointeger(state, 2), end = lua_tointeger(state, 3);
    /* Package dimensions use PyTorch axes: positive axes are zero-based. */
    if (start < 0) start += (lua_Integer)count + 1;
    else ++start;
    if (end < 0) end += (lua_Integer)count + 1;
    else ++end;
    if (start < 1 || end < start || (size_t)end > count) {
        nn_inference_lua_temp_dimensions_free(state, dims, count); return nn_inference_push_error(state, "flatten dimension range is invalid");
    }
    long long product = 1;
    for (lua_Integer i = start; i <= end; ++i) {
        char *stop = NULL; long value = strtol(dims[i - 1], &stop, 10);
        if (!stop || *stop || value < 1 || product > 2147483647LL / value) {
            nn_inference_lua_temp_dimensions_free(state, dims, count); return nn_inference_push_error(state, "flatten dimensions must have a bounded numeric product");
        }
        product *= value;
    }
    char **out = nn_inference_lua_temp_calloc(state, count - (size_t)(end - start), sizeof(*out));
    if (!out) { nn_inference_lua_temp_dimensions_free(state, dims, count); return nn_inference_push_error(state, "out of memory"); }
    size_t n = 0;
    bool copied = true;
    for (size_t i = 0; i < (size_t)start - 1; ++i) {
        out[n] = nn_inference_lua_temp_copy(state, dims[i]);
        if (!out[n++]) copied = false;
    }
    char product_text[32]; snprintf(product_text, sizeof(product_text), "%lld", product);
    out[n] = nn_inference_lua_temp_copy(state, product_text); if (!out[n++]) copied = false;
    for (size_t i = (size_t)end; i < count; ++i) {
        out[n] = nn_inference_lua_temp_copy(state, dims[i]);
        if (!out[n++]) copied = false;
    }
    if (!copied) {
        nn_inference_note_host_allocation_failure(state);
        nn_inference_lua_temp_dimensions_free(state, out, n); nn_inference_lua_temp_dimensions_free(state, dims, count);
        return nn_inference_push_error(state, "out of memory");
    }
    nn_inference_raw_getfield(state, 1, "dtype"); const char *dtype = lua_tostring(state, -1);
    nn_inference_tensor_push(state, dtype, (const char *const *)out, n);
    nn_inference_lua_temp_dimensions_free(state, out, n); nn_inference_lua_temp_dimensions_free(state, dims, count);
    return 1;
}

void nn_inference_set_tensor_functions(lua_State *state)
{
    lua_newtable(state);
    lua_pushcfunction(state, nn_inference_tensor_rank); lua_setfield(state, -2, "rank");
    lua_pushcfunction(state, nn_inference_tensor_dtype); lua_setfield(state, -2, "dtype");
    lua_pushcfunction(state, nn_inference_tensor_dimension); lua_setfield(state, -2, "dimension");
    lua_pushcfunction(state, nn_inference_tensor_with_dimension); lua_setfield(state, -2, "with_dimension");
    lua_pushcfunction(state, nn_inference_tensor_flatten); lua_setfield(state, -2, "flatten");
    lua_pushcfunction(state, nn_inference_tensor_create); lua_setfield(state, -2, "create");
    lua_setglobal(state, "tensor");
}