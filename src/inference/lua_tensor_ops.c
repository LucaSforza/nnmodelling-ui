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

static bool nn_inference_tensor_shape_valid(lua_State *state, int tensor, size_t *count)
{
    if (!nn_inference_tensor_read(state, tensor)) return false;
    nn_inference_raw_getfield(state, tensor, "shape");
    *count = lua_rawlen(state, -1);
    bool valid = *count <= 64;
    for (size_t i = 0; valid && i < *count; ++i) {
        lua_rawgeti(state, -1, (lua_Integer)i + 1);
        valid = lua_isinteger(state, -1) || lua_type(state, -1) == LUA_TSTRING;
        lua_pop(state, 1);
    }
    lua_pop(state, 1);
    return valid;
}

static void nn_inference_tensor_clone(lua_State *state, int tensor, int dtype,
                                      size_t count, bool append, lua_Integer size)
{
    tensor = lua_absindex(state, tensor);
    dtype = lua_absindex(state, dtype);
    nn_inference_raw_getfield(state, tensor, "shape");
    int input_shape = lua_absindex(state, -1);
    lua_createtable(state, (int)(count + (append ? 1 : 0)), 0);
    int output = lua_absindex(state, -1);
    lua_pushvalue(state, dtype);
    lua_setfield(state, output, "dtype");
    lua_createtable(state, (int)(count + (append ? 1 : 0)), 0);
    int shape = lua_absindex(state, -1);
    for (size_t i = 1; i <= count; ++i) {
        lua_rawgeti(state, input_shape, (lua_Integer)i);
        lua_rawseti(state, shape, (lua_Integer)i);
    }
    if (append) {
        lua_pushinteger(state, size);
        lua_rawseti(state, shape, (lua_Integer)count + 1);
    }
    lua_setfield(state, output, "shape");
}

static int nn_inference_tensor_append_dimension(lua_State *state)
{
    size_t count = 0;
    if (!nn_inference_tensor_shape_valid(state, 1, &count) || !lua_isinteger(state, 2) ||
        lua_tointeger(state, 2) < 1)
        return nn_inference_push_error(state, "expected a tensor and positive integer size");
    if (count == 64) return nn_inference_push_error(state, "tensor rank exceeds limit");
    nn_inference_raw_getfield(state, 1, "dtype");
    int dtype = lua_absindex(state, -1);
    nn_inference_tensor_clone(state, 1, dtype, count, true, lua_tointeger(state, 2));
    return 1;
}

static int nn_inference_tensor_with_dtype(lua_State *state)
{
    size_t count = 0;
    if (!nn_inference_tensor_shape_valid(state, 1, &count) || lua_type(state, 2) != LUA_TSTRING ||
        lua_rawlen(state, 2) == 0)
        return nn_inference_push_error(state, "expected a tensor and nonempty dtype");
    nn_inference_tensor_clone(state, 1, 2, count, false, 0);
    return 1;
}

static bool nn_inference_dimensions_equal(lua_State *state, int first, int second)
{
    int first_type = lua_type(state, first), second_type = lua_type(state, second);
    if (first_type == LUA_TNUMBER && second_type == LUA_TNUMBER)
        return lua_tointeger(state, first) == lua_tointeger(state, second);
    if (first_type != LUA_TSTRING || second_type != LUA_TSTRING) return false;
    size_t first_length = 0, second_length = 0;
    const char *first_text = lua_tolstring(state, first, &first_length);
    const char *second_text = lua_tolstring(state, second, &second_length);
    return first_length == second_length && !memcmp(first_text, second_text, first_length);
}

static int nn_inference_tensor_equal(lua_State *state)
{
    size_t first_count = 0, second_count = 0;
    if (!nn_inference_tensor_shape_valid(state, 1, &first_count) ||
        !nn_inference_tensor_shape_valid(state, 2, &second_count))
        return nn_inference_push_error(state, "expected two valid tensors");

    nn_inference_raw_getfield(state, 1, "dtype");
    size_t first_dtype_length = 0;
    const char *first_dtype = lua_tolstring(state, -1, &first_dtype_length);
    nn_inference_raw_getfield(state, 2, "dtype");
    size_t second_dtype_length = 0;
    const char *second_dtype = lua_tolstring(state, -1, &second_dtype_length);
    bool equal = first_dtype_length == second_dtype_length &&
        !memcmp(first_dtype, second_dtype, first_dtype_length) && first_count == second_count;
    lua_pop(state, 2);
    if (equal) {
        nn_inference_raw_getfield(state, 1, "shape");
        int first_shape = lua_absindex(state, -1);
        nn_inference_raw_getfield(state, 2, "shape");
        int second_shape = lua_absindex(state, -1);
        for (size_t i = 1; equal && i <= first_count; ++i) {
            lua_rawgeti(state, first_shape, (lua_Integer)i);
            lua_rawgeti(state, second_shape, (lua_Integer)i);
            equal = nn_inference_dimensions_equal(state, -2, -1);
            lua_pop(state, 2);
        }
        lua_pop(state, 2);
    }
    lua_pushboolean(state, equal);
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
    lua_pushcfunction(state, nn_inference_tensor_append_dimension); lua_setfield(state, -2, "append_dimension");
    lua_pushcfunction(state, nn_inference_tensor_with_dtype); lua_setfield(state, -2, "with_dtype");
    lua_pushcfunction(state, nn_inference_tensor_equal); lua_setfield(state, -2, "equal");
    lua_pushcfunction(state, nn_inference_tensor_flatten); lua_setfield(state, -2, "flatten");
    lua_pushcfunction(state, nn_inference_tensor_create); lua_setfield(state, -2, "create");
    lua_setglobal(state, "tensor");
}
