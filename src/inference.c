#define _POSIX_C_SOURCE 200809L
#include "inference.h"

#include "catalog.h"
#include "model.h"
#include "project.h"
#include "utils.h"

#include <lauxlib.h>
#include <lua.h>
#include <lualib.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#define LUA_MEMORY_LIMIT (16u * 1024u * 1024u)
#define LUA_INSTRUCTION_LIMIT 200000u
#define RULE_FILE_LIMIT (1024u * 1024u)

#define LUA_TEMP_ALLOCATION_LIMIT 512
typedef struct {
    size_t used;
    unsigned instructions;
    bool host_allocation_failed;
    size_t temporary_count;
    void *temporary[LUA_TEMP_ALLOCATION_LIMIT];
} LuaBudget;
typedef struct { char *dtype; char **dimensions; size_t count; } Tensor;
typedef struct { const NNEdge *edge; size_t order; } Incoming;
typedef struct {
    NNInferenceResult view;
    char *id, *message, *dtype, *cause_node_id, *source_file;
    char **dimensions;
} Result;

struct NNInferenceReport { Result *items; size_t count; bool failed; };

typedef struct Evaluation Evaluation;
typedef struct {
    Evaluation *evaluation;
    const NNNode *node;
    size_t depth;
    const Tensor *inherited;
    char *cause_node_id;
    Tensor inherited_scratch;
    Tensor output_scratch;
    char *message_scratch;
} LuaContext;
typedef struct {
    LuaContext *context;
    Tensor *inputs;
    size_t input_count;
} RuleCall;
struct Evaluation {
    const NNModel *model;
    const NNCatalog *catalog;
    NNInferenceReport *report;
    const NNDataset *dataset;
    size_t invocations;
    bool allocation_failed;
};

static NNInferenceStatus evaluate_scope(Evaluation *evaluation, const char *scope,
                                        const Tensor *inherited, size_t depth,
                                        Tensor *output, char **message,
                                        char **cause_node_id);
static int inference_subflow(lua_State *state);
static const char *package_kind(Evaluation *evaluation, const NNNode *node);
static bool set_parameters(lua_State *state, const NNNode *node);
static int invoke_rule(lua_State *state);
static char invoke_rule_registry_key;
static int extract_rule_result(lua_State *state);
static char extract_result_registry_key;

typedef struct {
    Tensor *output;
    char **message;
    NNInferenceStatus status;
} ExtractResult;

static void *limited_alloc(void *data, void *ptr, size_t old_size, size_t new_size)
{
    LuaBudget *budget = data;
    if (!new_size) {
        if (ptr) budget->used -= old_size;
        free(ptr);
        return NULL;
    }
    size_t prior = ptr ? old_size : 0;
    if (new_size > LUA_MEMORY_LIMIT || budget->used - prior > LUA_MEMORY_LIMIT - new_size)
        return NULL;
    void *grown = realloc(ptr, new_size);
    if (grown) budget->used = budget->used - prior + new_size;
    return grown;
}

static void instruction_hook(lua_State *state, lua_Debug *debug)
{
    (void)debug;
    LuaBudget *budget = *(LuaBudget **)lua_getextraspace(state);
    budget->instructions += 1000;
    if (budget->instructions > LUA_INSTRUCTION_LIMIT) {
        lua_pushliteral(state, "inference instruction limit exceeded");
        lua_error(state);
    }
}

static void note_host_allocation_failure(lua_State *state)
{
    LuaBudget *budget = *(LuaBudget **)lua_getextraspace(state);
    if (budget) budget->host_allocation_failed = true;
}

static LuaBudget *lua_budget(lua_State *state)
{
    return *(LuaBudget **)lua_getextraspace(state);
}

static void *lua_temp_calloc(lua_State *state, size_t count, size_t size)
{
    if (size && count > SIZE_MAX / size) {
        note_host_allocation_failure(state);
        return NULL;
    }
    LuaBudget *budget = lua_budget(state);
    if (!budget || budget->temporary_count == LUA_TEMP_ALLOCATION_LIMIT) {
        note_host_allocation_failure(state);
        return NULL;
    }
    void *memory = calloc(count ? count : 1, size ? size : 1);
    if (!memory) {
        note_host_allocation_failure(state);
        return NULL;
    }
    budget->temporary[budget->temporary_count++] = memory;
    return memory;
}

static void *lua_temp_copy(lua_State *state, const char *text)
{
    if (!text) return NULL;
    void *copy = nn_text_copy(text);
    if (!copy) {
        note_host_allocation_failure(state);
        return NULL;
    }
    LuaBudget *budget = lua_budget(state);
    if (!budget || budget->temporary_count == LUA_TEMP_ALLOCATION_LIMIT) {
        free(copy);
        note_host_allocation_failure(state);
        return NULL;
    }
    budget->temporary[budget->temporary_count++] = copy;
    return copy;
}

static void lua_temp_free(lua_State *state, void *memory)
{
    if (!memory) return;
    LuaBudget *budget = lua_budget(state);
    if (budget) {
        for (size_t i = 0; i < budget->temporary_count; ++i) {
            if (budget->temporary[i] != memory) continue;
            budget->temporary[i] = budget->temporary[--budget->temporary_count];
            break;
        }
    }
    free(memory);
}

static void lua_temp_detach(lua_State *state, void *memory)
{
    LuaBudget *budget = lua_budget(state);
    if (!memory || !budget) return;
    for (size_t i = 0; i < budget->temporary_count; ++i) {
        if (budget->temporary[i] != memory) continue;
        budget->temporary[i] = budget->temporary[--budget->temporary_count];
        return;
    }
}

static void lua_temp_dimensions_free(lua_State *state, char **dimensions, size_t count)
{
    if (!dimensions) return;
    for (size_t i = 0; i < count; ++i) lua_temp_free(state, dimensions[i]);
    lua_temp_free(state, dimensions);
}

static void lua_temp_cleanup(LuaBudget *budget)
{
    while (budget->temporary_count)
        free(budget->temporary[--budget->temporary_count]);
}

static void raw_getfield(lua_State *state, int index, const char *key)
{
    index = lua_absindex(state, index);
    lua_pushstring(state, key);
    lua_rawget(state, index);
}

static int tensor_read(lua_State *state, int index)
{
    if (!lua_istable(state, index)) return 0;
    raw_getfield(state, index, "shape");
    bool valid = lua_istable(state, -1);
    lua_pop(state, 1);
    raw_getfield(state, index, "dtype");
    valid = valid && lua_type(state, -1) == LUA_TSTRING;
    lua_pop(state, 1);
    return valid;
}

static void tensor_push(lua_State *state, const char *dtype,
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

static int push_error(lua_State *state, const char *message)
{
    lua_pushnil(state);
    lua_pushstring(state, message);
    return 2;
}

static int tensor_rank(lua_State *state)
{
    if (!tensor_read(state, 1)) return push_error(state, "expected a tensor");
    raw_getfield(state, 1, "shape");
    lua_pushinteger(state, (lua_Integer)lua_rawlen(state, -1));
    return 1;
}

static int tensor_dtype(lua_State *state)
{
    if (!tensor_read(state, 1)) return push_error(state, "expected a tensor");
    raw_getfield(state, 1, "dtype");
    return 1;
}

static int dimension_index(lua_State *state, int tensor, lua_Integer dim, size_t *count)
{
    raw_getfield(state, tensor, "shape");
    *count = lua_rawlen(state, -1);
    lua_pop(state, 1);
    lua_Integer resolved = dim < 0 ? (lua_Integer)*count + dim + 1 : dim;
    if (resolved < 1 || (size_t)resolved > *count) return 0;
    return (int)resolved;
}

static int tensor_dimension(lua_State *state)
{
    if (!tensor_read(state, 1) || !lua_isinteger(state, 2))
        return push_error(state, "expected a tensor and integer dimension");
    size_t count;
    int index = dimension_index(state, 1, lua_tointeger(state, 2), &count);
    if (!index) return push_error(state, "dimension is out of range");
    raw_getfield(state, 1, "shape");
    lua_rawgeti(state, -1, index);
    return 1;
}

static bool read_shape(lua_State *state, int tensor, char ***dims, size_t *count,
                       bool temporary)
{
    raw_getfield(state, tensor, "shape");
    *count = lua_rawlen(state, -1);
    if (*count > 64) { lua_pop(state, 1); return false; }
    *dims = temporary ? lua_temp_calloc(state, *count ? *count : 1, sizeof(**dims))
                      : calloc(*count ? *count : 1, sizeof(**dims));
    if (!*dims) { note_host_allocation_failure(state); lua_pop(state, 1); return false; }
    for (size_t i = 0; i < *count; ++i) {
        lua_rawgeti(state, -1, (lua_Integer)i + 1);
        bool supported_type = lua_isinteger(state, -1) || lua_type(state, -1) == LUA_TSTRING;
        if (lua_isinteger(state, -1)) {
            char text[32]; snprintf(text, sizeof(text), "%lld", (long long)lua_tointeger(state, -1));
            (*dims)[i] = temporary ? lua_temp_copy(state, text) : nn_text_copy(text);
        } else if (lua_type(state, -1) == LUA_TSTRING)
            (*dims)[i] = temporary ? lua_temp_copy(state, lua_tostring(state, -1))
                                   : nn_text_copy(lua_tostring(state, -1));
        lua_pop(state, 1);
        if (!(*dims)[i]) {
            if (supported_type) note_host_allocation_failure(state);
            if (temporary) lua_temp_dimensions_free(state, *dims, i);
            else { for (size_t j = 0; j < i; ++j) free((*dims)[j]); free(*dims); }
            *dims = NULL; *count = 0;
            lua_pop(state, 1); return false;
        }
    }
    lua_pop(state, 1);
    return true;
}

static void free_dimensions(char **dimensions, size_t count)
{
    if (!dimensions) return;
    for (size_t i = 0; i < count; ++i) free(dimensions[i]);
    free(dimensions);
}

static int tensor_with_dimension(lua_State *state)
{
    if (!tensor_read(state, 1) || !lua_isinteger(state, 2) || !lua_isinteger(state, 3))
        return push_error(state, "expected tensor, integer dimension and integer size");
    char **dims = NULL; size_t count = 0;
    if (!read_shape(state, 1, &dims, &count, true)) return push_error(state, "invalid tensor shape");
    size_t unused;
    int index = dimension_index(state, 1, lua_tointeger(state, 2), &unused);
    if (!index || lua_tointeger(state, 3) < 1) {
        lua_temp_dimensions_free(state, dims, count);
        return push_error(state, "dimension or size is invalid");
    }
    char text[32]; snprintf(text, sizeof(text), "%lld", (long long)lua_tointeger(state, 3));
    lua_temp_free(state, dims[index - 1]); dims[index - 1] = lua_temp_copy(state, text);
    if (!dims[index - 1]) {
        note_host_allocation_failure(state);
        lua_temp_dimensions_free(state, dims, count);
        return push_error(state, "out of memory");
    }
    raw_getfield(state, 1, "dtype"); const char *dtype = lua_tostring(state, -1);
    tensor_push(state, dtype, (const char *const *)dims, count);
    lua_temp_dimensions_free(state, dims, count);
    return 1;
}

static int tensor_create(lua_State *state)
{
    if (!lua_istable(state, 1) || lua_type(state, 2) != LUA_TSTRING)
        return push_error(state, "expected shape and dtype");
    size_t count = lua_rawlen(state, 1);
    if (!count || count > 64) return push_error(state, "invalid tensor rank");
    char **dims = lua_temp_calloc(state, count, sizeof(*dims));
    if (!dims) return push_error(state, "out of memory");
    for (size_t i = 0; i < count; ++i) {
        lua_rawgeti(state, 1, (lua_Integer)i + 1);
        bool supported_type = lua_isinteger(state, -1) || lua_type(state, -1) == LUA_TSTRING;
        if (lua_isinteger(state, -1)) {
            char text[32]; snprintf(text, sizeof(text), "%lld", (long long)lua_tointeger(state, -1));
            dims[i] = lua_temp_copy(state, text);
        } else if (lua_type(state, -1) == LUA_TSTRING) dims[i] = lua_temp_copy(state, lua_tostring(state, -1));
        lua_pop(state, 1);
        if (!dims[i]) {
            if (supported_type) note_host_allocation_failure(state);
            lua_temp_dimensions_free(state, dims, count);
            return push_error(state, "out of memory");
        }
    }
    tensor_push(state, lua_tostring(state, 2), (const char *const *)dims, count);
    lua_temp_dimensions_free(state, dims, count);
    return 1;
}

static int tensor_flatten(lua_State *state)
{
    if (!tensor_read(state, 1) || !lua_isinteger(state, 2) || !lua_isinteger(state, 3))
        return push_error(state, "expected tensor and integer dimension range");
    char **dims = NULL; size_t count = 0;
    if (!read_shape(state, 1, &dims, &count, true)) return push_error(state, "invalid tensor shape");
    lua_Integer start = lua_tointeger(state, 2), end = lua_tointeger(state, 3);
    /* Package dimensions use PyTorch axes: positive axes are zero-based. */
    if (start < 0) start += (lua_Integer)count + 1;
    else ++start;
    if (end < 0) end += (lua_Integer)count + 1;
    else ++end;
    if (start < 1 || end < start || (size_t)end > count) {
        lua_temp_dimensions_free(state, dims, count); return push_error(state, "flatten dimension range is invalid");
    }
    long long product = 1;
    for (lua_Integer i = start; i <= end; ++i) {
        char *stop = NULL; long value = strtol(dims[i - 1], &stop, 10);
        if (!stop || *stop || value < 1 || product > 2147483647LL / value) {
            lua_temp_dimensions_free(state, dims, count); return push_error(state, "flatten dimensions must have a bounded numeric product");
        }
        product *= value;
    }
    char **out = lua_temp_calloc(state, count - (size_t)(end - start), sizeof(*out));
    if (!out) { lua_temp_dimensions_free(state, dims, count); return push_error(state, "out of memory"); }
    size_t n = 0;
    bool copied = true;
    for (size_t i = 0; i < (size_t)start - 1; ++i) {
        out[n] = lua_temp_copy(state, dims[i]);
        if (!out[n++]) copied = false;
    }
    char product_text[32]; snprintf(product_text, sizeof(product_text), "%lld", product);
    out[n] = lua_temp_copy(state, product_text); if (!out[n++]) copied = false;
    for (size_t i = (size_t)end; i < count; ++i) {
        out[n] = lua_temp_copy(state, dims[i]);
        if (!out[n++]) copied = false;
    }
    if (!copied) {
        note_host_allocation_failure(state);
        lua_temp_dimensions_free(state, out, n); lua_temp_dimensions_free(state, dims, count);
        return push_error(state, "out of memory");
    }
    raw_getfield(state, 1, "dtype"); const char *dtype = lua_tostring(state, -1);
    tensor_push(state, dtype, (const char *const *)out, n);
    lua_temp_dimensions_free(state, out, n); lua_temp_dimensions_free(state, dims, count);
    return 1;
}

static void set_tensor_functions(lua_State *state)
{
    lua_newtable(state);
    lua_pushcfunction(state, tensor_rank); lua_setfield(state, -2, "rank");
    lua_pushcfunction(state, tensor_dtype); lua_setfield(state, -2, "dtype");
    lua_pushcfunction(state, tensor_dimension); lua_setfield(state, -2, "dimension");
    lua_pushcfunction(state, tensor_with_dimension); lua_setfield(state, -2, "with_dimension");
    lua_pushcfunction(state, tensor_flatten); lua_setfield(state, -2, "flatten");
    lua_pushcfunction(state, tensor_create); lua_setfield(state, -2, "create");
    lua_setglobal(state, "tensor");
}

static int resolve_input(lua_State *state)
{
    LuaContext *context = lua_touserdata(state, lua_upvalueindex(1));
    const char *binding = lua_type(state, 1) == LUA_TSTRING ? lua_tostring(state, 1) : NULL;
    if (!context) {
        lua_newtable(state); lua_pushliteral(state, "unresolved"); lua_setfield(state, -2, "status");
        lua_pushliteral(state, "input binding is unavailable"); lua_setfield(state, -2, "message");
        return 1;
    }
    if (context->inherited) {
        lua_newtable(state); lua_pushliteral(state, "success"); lua_setfield(state, -2, "status");
        tensor_push(state, context->inherited->dtype,
                    (const char *const *)context->inherited->dimensions,
                    context->inherited->count);
        lua_setfield(state, -2, "output");
        return 1;
    }
    if (!binding) {
        lua_newtable(state); lua_pushliteral(state, "unresolved"); lua_setfield(state, -2, "status");
        lua_pushliteral(state, "input binding is unavailable"); lua_setfield(state, -2, "message");
        return 1;
    }
    const NNDataset *dataset = context->evaluation->dataset;
    if (!dataset) {
        lua_newtable(state); lua_pushliteral(state, "unresolved"); lua_setfield(state, -2, "status");
        lua_pushliteral(state, "input binding has no selected dataset"); lua_setfield(state, -2, "message");
        return 1;
    }
    for (size_t i = 0; i < dataset->input_count; ++i) {
        const NNTensorSlot *slot = &dataset->inputs[i];
        if (strcmp(slot->name, binding)) continue;
        if (slot->shape.type != NN_VALUE_ARRAY) break;
        size_t count = slot->shape.as.array.count;
        char **dims = lua_temp_calloc(state, count ? count : 1, sizeof(*dims));
        if (!dims) break;
        bool valid = true;
        for (size_t d = 0; d < count; ++d) {
            const NNValue *value = &slot->shape.as.array.items[d];
            char text[32];
            if (value->type == NN_VALUE_INT) snprintf(text, sizeof(text), "%lld", value->as.integer);
            else if (value->type == NN_VALUE_STRING) snprintf(text, sizeof(text), "%s", value->as.string);
            else { valid = false; break; }
            dims[d] = lua_temp_copy(state, text);
            if (!dims[d]) { note_host_allocation_failure(state); valid = false; }
        }
        if (!valid) { lua_temp_dimensions_free(state, dims, count); break; }
        lua_newtable(state); lua_pushliteral(state, "success"); lua_setfield(state, -2, "status");
        tensor_push(state, slot->dtype, (const char *const *)dims, count);
        lua_setfield(state, -2, "output"); lua_temp_dimensions_free(state, dims, count);
        return 1;
    }
    lua_newtable(state); lua_pushliteral(state, "unresolved"); lua_setfield(state, -2, "status");
    lua_pushfstring(state, "dataset input binding '%s' was not found", binding);
    lua_setfield(state, -2, "message");
    return 1;
}

static void set_services(lua_State *state, LuaContext *context)
{
    lua_newtable(state);
    lua_pushlightuserdata(state, context);
    lua_pushcclosure(state, resolve_input, 1);
    lua_setfield(state, -2, "resolve_input");
    const char *kind = package_kind(context->evaluation, context->node);
    if (kind && !strcmp(kind, "subflow")) {
        lua_pushlightuserdata(state, context);
        lua_pushcclosure(state, inference_subflow, 1);
        lua_setfield(state, -2, "infer_subflow");
    }
    lua_setglobal(state, "services");
}

static int invoke_rule(lua_State *state)
{
    RuleCall *call = lua_touserdata(state, 2);
    if (!call || !call->context) return luaL_error(state, "invalid inference call context");
    lua_pushvalue(state, 1);
    lua_newtable(state);
    lua_createtable(state, (int)call->input_count, 0);
    for (size_t i = 0; i < call->input_count; ++i) {
        tensor_push(state, call->inputs[i].dtype,
                    (const char *const *)call->inputs[i].dimensions,
                    call->inputs[i].count);
        lua_seti(state, -2, (lua_Integer)i + 1);
    }
    lua_setfield(state, -2, "inputs");
    if (!set_parameters(state, call->context->node))
        return luaL_error(state, "unable to build inference parameters");
    set_services(state, call->context);
    lua_getglobal(state, "services");
    lua_call(state, 3, 1);
    return 1;
}

static bool push_value(lua_State *state, const NNValue *value)
{
    switch (value->type) {
    case NN_VALUE_BOOL: lua_pushboolean(state, value->as.boolean); return true;
    case NN_VALUE_INT: lua_pushinteger(state, value->as.integer); return true;
    case NN_VALUE_REAL: lua_pushnumber(state, value->as.real); return true;
    case NN_VALUE_STRING: lua_pushstring(state, value->as.string); return true;
    case NN_VALUE_ARRAY:
        if (value->as.array.count > 1024) return false;
        lua_createtable(state, (int)value->as.array.count, 0);
        for (size_t i = 0; i < value->as.array.count; ++i) {
            if (!push_value(state, &value->as.array.items[i])) return false;
            lua_seti(state, -2, (lua_Integer)i + 1);
        }
        return true;
    default: return false;
    }
}

static bool set_parameters(lua_State *state, const NNNode *node)
{
    lua_createtable(state, 0, (int)node->parameter_count);
    for (size_t i = 0; i < node->parameter_count; ++i) {
        if (push_value(state, &node->parameters[i].value))
            lua_setfield(state, -2, node->parameters[i].key);
        else return false;
    }
    return true;
}

static char *read_rule(const NNPackage *package, size_t *length, bool *allocation_failed)
{
    if (!package || !package->directory || !package->inference_file ||
        package->inference_file[0] == '/' || strstr(package->inference_file, "..")) return NULL;
    size_t a = strlen(package->directory), b = strlen(package->inference_file);
    if (b > SIZE_MAX - 2 || a > SIZE_MAX - b - 2) return NULL;
    char *path = malloc(a + b + 2);
    if (!path) { *allocation_failed = true; return NULL; }
    snprintf(path, a + b + 2, "%s/%s", package->directory, package->inference_file);
    struct stat info;
    if (lstat(path, &info) || !S_ISREG(info.st_mode) || info.st_size < 0 ||
        (size_t)info.st_size > RULE_FILE_LIMIT) { free(path); return NULL; }
    FILE *file = fopen(path, "rb"); free(path);
    if (!file) return NULL;
    *length = (size_t)info.st_size;
    char *source = malloc(*length + 1);
    if (!source) { *allocation_failed = true; fclose(file); return NULL; }
    bool okay = fread(source, 1, *length, file) == *length && !ferror(file);
    fclose(file);
    if (!okay) { free(source); return NULL; }
    source[*length] = '\0';
    return source;
}

static void open_safe_libraries(lua_State *state)
{
    luaL_requiref(state, LUA_GNAME, luaopen_base, 1); lua_pop(state, 1);
    lua_pushnil(state); lua_setglobal(state, "dofile");
    lua_pushnil(state); lua_setglobal(state, "loadfile");
    lua_pushnil(state); lua_setglobal(state, "load");
    lua_pushnil(state); lua_setglobal(state, "collectgarbage");
    luaL_requiref(state, LUA_MATHLIBNAME, luaopen_math, 1); lua_pop(state, 1);
    set_tensor_functions(state);
}

static int initialize_lua(lua_State *state)
{
    open_safe_libraries(state);
    lua_pushlightuserdata(state, &invoke_rule_registry_key);
    lua_pushcfunction(state, invoke_rule);
    lua_rawset(state, LUA_REGISTRYINDEX);
    lua_pushlightuserdata(state, &extract_result_registry_key);
    lua_pushcfunction(state, extract_rule_result);
    lua_rawset(state, LUA_REGISTRYINDEX);
    return 0;
}

static bool protected_lua_initialize(lua_State *state)
{
    lua_pushcfunction(state, initialize_lua);
    return lua_pcall(state, 0, 0, 0) == LUA_OK;
}

bool nn_inference_validate_source(const char *source, char *error, size_t capacity)
{
    nn_error_set(error, capacity, "");
    if (!source) {
        nn_error_set(error, capacity, "inference source is missing");
        return false;
    }
    size_t length = strlen(source);
    if (length > RULE_FILE_LIMIT) {
        nn_error_set(error, capacity, "inference source exceeds 1 MiB");
        return false;
    }
    LuaBudget budget = {0};
    lua_State *state = lua_newstate(limited_alloc, &budget, 0);
    if (!state) {
        nn_error_set(error, capacity, "unable to create bounded Lua state");
        return false;
    }
    *(LuaBudget **)lua_getextraspace(state) = &budget;
    if (!protected_lua_initialize(state)) {
        nn_error_set(error, capacity, "unable to initialize bounded Lua state");
        lua_close(state);
        return false;
    }
    lua_sethook(state, instruction_hook, LUA_MASKCOUNT, 1000);
    int status = luaL_loadbufferx(state, source, length, "inference", "t");
    if (status == LUA_OK) status = lua_pcall(state, 0, 1, 0);
    const char *validation_message = NULL;
    if (status == LUA_OK && !lua_isfunction(state, -1)) {
        status = LUA_ERRRUN;
        validation_message = "inference source must return a function";
    }
    bool valid = status == LUA_OK;
    if (!valid && error && capacity) {
        const char *message = validation_message;
        if (!message && lua_type(state, -1) == LUA_TSTRING) message = lua_tostring(state, -1);
        nn_error_set(error, capacity, message ? message : "invalid inference source");
    }
    lua_close(state);
    return valid;
}

static bool tensor_from_lua(lua_State *state, int index, Tensor *tensor, bool tracked)
{
    if (!tensor_read(state, index)) return false;
    raw_getfield(state, index, "dtype");
    tensor->dtype = tracked ? lua_temp_copy(state, lua_tostring(state, -1))
                            : nn_text_copy(lua_tostring(state, -1));
    lua_pop(state, 1);
    if (!tensor->dtype) {
        if (!tracked) note_host_allocation_failure(state);
        return false;
    }
    char **dims = NULL;
    size_t count = 0;
    if (!read_shape(state, index, &dims, &count, tracked)) {
        if (tracked) lua_temp_free(state, tensor->dtype);
        else free(tensor->dtype);
        memset(tensor, 0, sizeof(*tensor));
        return false;
    }
    tensor->dimensions = dims;
    tensor->count = count;
    if (tracked) {
        lua_temp_detach(state, tensor->dtype);
        lua_temp_detach(state, dims);
        for (size_t i = 0; i < count; ++i) lua_temp_detach(state, dims[i]);
    }
    return true;
}

static void tensor_detach_lua_temporaries(lua_State *state, Tensor *tensor)
{
    lua_temp_detach(state, tensor->dtype);
    lua_temp_detach(state, tensor->dimensions);
    for (size_t i = 0; i < tensor->count; ++i)
        lua_temp_detach(state, tensor->dimensions[i]);
}

static void extraction_message(lua_State *state, ExtractResult *result, const char *message)
{
    *result->message = nn_text_copy(message);
    if (!*result->message) note_host_allocation_failure(state);
}

static int extract_rule_result(lua_State *state)
{
    ExtractResult *result = lua_touserdata(state, 2);
    if (!result || !result->output || !result->message)
        return luaL_error(state, "invalid inference result extraction context");
    raw_getfield(state, 1, "status");
    const char *kind = lua_type(state, -1) == LUA_TSTRING ? lua_tostring(state, -1) : NULL;
    lua_pop(state, 1);
    result->status = NN_INFERENCE_RUNTIME_FAULT;
    if (kind && !strcmp(kind, "success")) {
        raw_getfield(state, 1, "output");
        if (lua_isnil(state, -1)) {
            result->status = NN_INFERENCE_UNRESOLVED;
            extraction_message(state, result,
                "rule output is missing because its upstream tensor is unresolved");
        } else if (tensor_from_lua(state, -1, result->output, true)) {
            tensor_detach_lua_temporaries(state, result->output);
            result->status = NN_INFERENCE_SUCCESS;
        } else {
            extraction_message(state, result, "rule returned an invalid tensor");
        }
        lua_pop(state, 1);
    } else if (kind && (!strcmp(kind, "error") || !strcmp(kind, "unresolved"))) {
        raw_getfield(state, 1, "message");
        if (lua_type(state, -1) == LUA_TSTRING) {
            *result->message = nn_text_copy(lua_tostring(state, -1));
            if (!*result->message) note_host_allocation_failure(state);
        }
        lua_pop(state, 1);
        if (!*result->message)
            extraction_message(state, result, "rule returned a diagnostic without a message");
        result->status = !strcmp(kind, "error") ? NN_INFERENCE_SEMANTIC_ERROR
                                                  : NN_INFERENCE_UNRESOLVED;
    } else {
        extraction_message(state, result, "rule returned an invalid result");
    }
    lua_pushinteger(state, (lua_Integer)result->status);
    return 1;
}

static void tensor_dispose(Tensor *tensor)
{
    free(tensor->dtype); free_dimensions(tensor->dimensions, tensor->count);
    memset(tensor, 0, sizeof(*tensor));
}

static bool tensor_copy(Tensor *destination, const Tensor *source)
{
    destination->dtype = nn_text_copy(source->dtype);
    destination->dimensions = calloc(source->count ? source->count : 1,
                                     sizeof(*destination->dimensions));
    if (!destination->dtype || !destination->dimensions) { tensor_dispose(destination); return false; }
    destination->count = source->count;
    for (size_t i = 0; i < source->count; ++i) {
        destination->dimensions[i] = nn_text_copy(source->dimensions[i]);
        if (!destination->dimensions[i]) { tensor_dispose(destination); return false; }
    }
    return true;
}

static NNInferenceStatus execute_rule(Evaluation *evaluation, const NNNode *node,
                                      const NNPackage *package, size_t depth,
                                      const Tensor *inherited,
                                      Tensor *inputs, size_t input_count,
                                       Tensor *output, char **message,
                                       char **source_file, size_t *source_line,
                                       char **cause_node_id)
{
    size_t source_length = 0;
    bool read_allocation_failed = false;
    char *source = read_rule(package, &source_length, &read_allocation_failed);
    if (read_allocation_failed) evaluation->allocation_failed = true;
    if (!source) { *message = nn_text_copy("inference rule unavailable or invalid"); return NN_INFERENCE_RUNTIME_FAULT; }
    LuaBudget budget = {0};
    lua_State *state = lua_newstate(limited_alloc, &budget, 0);
    if (!state) { free(source); *message = nn_text_copy("unable to create bounded Lua state"); return NN_INFERENCE_RUNTIME_FAULT; }
    *(LuaBudget **)lua_getextraspace(state) = &budget;
    if (!protected_lua_initialize(state)) {
        lua_close(state);
        free(source);
        *message = nn_text_copy("unable to initialize bounded Lua state");
        return NN_INFERENCE_RUNTIME_FAULT;
    }
    lua_sethook(state, instruction_hook, LUA_MASKCOUNT, 1000);
    int status = luaL_loadbufferx(state, source, source_length, package->id, "t");
    free(source);
    bool compilation_error = status == LUA_ERRSYNTAX;
    if (compilation_error) {
        const char *compile_message = lua_tostring(state, -1);
        *source_file = nn_text_copy(package->inference_file);
        *source_line = nn_inference_error_line(compile_message);
        *message = nn_text_copy(compile_message ? compile_message : "Lua compilation failed");
    }
    if (status == LUA_OK) status = lua_pcall(state, 0, 1, 0);
    if (status == LUA_OK && !lua_isfunction(state, -1)) status = LUA_ERRRUN;
    LuaContext context = { .evaluation = evaluation, .node = node,
                           .depth = depth, .inherited = inherited };
    RuleCall call = { .context = &context, .inputs = inputs,
                      .input_count = input_count };
    if (status == LUA_OK) {
        lua_pushlightuserdata(state, &invoke_rule_registry_key);
        lua_rawget(state, LUA_REGISTRYINDEX);
        lua_pushvalue(state, 1);
        lua_pushlightuserdata(state, &call);
        status = lua_pcall(state, 2, 1, 0);
        lua_remove(state, 1);
    }
    if (context.cause_node_id) {
        *cause_node_id = nn_text_copy(context.cause_node_id);
        if (!*cause_node_id) evaluation->allocation_failed = true;
    }
    free(context.cause_node_id);
    lua_temp_detach(state, context.inherited_scratch.dtype);
    lua_temp_detach(state, context.inherited_scratch.dimensions);
    for (size_t i = 0; i < context.inherited_scratch.count; ++i)
        lua_temp_detach(state, context.inherited_scratch.dimensions[i]);
    tensor_dispose(&context.inherited_scratch);
    tensor_dispose(&context.output_scratch);
    free(context.message_scratch);
    NNInferenceStatus result = compilation_error ? NN_INFERENCE_COMPILATION_ERROR
                                                  : NN_INFERENCE_RUNTIME_FAULT;
    if (status == LUA_OK && lua_istable(state, -1)) {
        ExtractResult extracted = { .output = output, .message = message,
                                    .status = NN_INFERENCE_RUNTIME_FAULT };
        lua_pushlightuserdata(state, &extract_result_registry_key);
        lua_rawget(state, LUA_REGISTRYINDEX);
        lua_pushvalue(state, 1);
        lua_pushlightuserdata(state, &extracted);
        int extraction_status = lua_pcall(state, 2, 1, 0);
        lua_remove(state, 1);
        if (extraction_status == LUA_OK && lua_isinteger(state, -1)) {
            result = (NNInferenceStatus)lua_tointeger(state, -1);
            lua_pop(state, 1);
        } else {
            if (lua_gettop(state)) lua_pop(state, 1);
            result = NN_INFERENCE_RUNTIME_FAULT;
            free(*message);
            *message = nn_text_copy("Lua allocation failure while extracting inference result");
            if (!*message) evaluation->allocation_failed = true;
            tensor_detach_lua_temporaries(state, output);
            tensor_dispose(output);
        }
        if (result != NN_INFERENCE_SUCCESS && output->dtype) {
            tensor_detach_lua_temporaries(state, output);
            tensor_dispose(output);
        }
    } else if (status != LUA_OK && !compilation_error) {
        const char *error = lua_type(state, -1) == LUA_TSTRING
            ? lua_tostring(state, -1) : NULL;
        *message = nn_text_copy(error ? error : "Lua inference failed");
    }
    if (budget.host_allocation_failed) evaluation->allocation_failed = true;
    lua_close(state);
    lua_temp_cleanup(&budget);
    return result;
}

static size_t find_node_index(const NNModel *model, const char *id)
{
    for (size_t i = 0; i < nn_model_node_count(model); ++i)
        if (!strcmp(nn_model_node_at(model, i)->id, id)) return i;
    return (size_t)-1;
}

static bool result_set(Result *result, const NNNode *node, NNInferenceStatus status,
                       char *message, Tensor *tensor, const char *cause,
                       const char *source_file, size_t source_line,
                       const char *code_override)
{
    free(result->id); free(result->message); free(result->dtype);
    free(result->cause_node_id); free(result->source_file);
    free_dimensions(result->dimensions, result->view.dimension_count);
    memset(result, 0, sizeof(*result));
    result->id = nn_text_copy(node->id); result->message = message;
    result->view.node_id = result->id; result->view.status = status;
    result->view.message = result->message;
    result->cause_node_id = nn_text_copy(cause);
    result->source_file = nn_text_copy(source_file);
    result->view.cause_node_id = result->cause_node_id;
    result->view.source_file = result->source_file;
    result->view.source_line = source_line;
    if (code_override) result->view.code = code_override;
    else if (status == NN_INFERENCE_COMPILATION_ERROR) result->view.code = "lua.compile";
    else if (status == NN_INFERENCE_SEMANTIC_ERROR) result->view.code = "model.semantic";
    else if (status == NN_INFERENCE_UNRESOLVED) result->view.code = "model.incomplete";
    else if (status == NN_INFERENCE_RUNTIME_FAULT) result->view.code = "analysis.internal";
    else result->view.code = NULL;
    if (status == NN_INFERENCE_SUCCESS && tensor) {
        result->dtype = nn_text_copy(tensor->dtype);
        result->dimensions = calloc(tensor->count ? tensor->count : 1, sizeof(*result->dimensions));
        for (size_t i = 0; i < tensor->count && result->dimensions; ++i)
            result->dimensions[i] = nn_text_copy(tensor->dimensions[i]);
        result->view.dtype = result->dtype;
        result->view.dimensions = (const char *const *)result->dimensions;
        result->view.dimension_count = tensor->count;
    }
    if (!result->id || (status != NN_INFERENCE_SUCCESS && (!message || !result->view.code)) ||
        (cause && !result->cause_node_id) || (source_file && !result->source_file) ||
        (status == NN_INFERENCE_COMPILATION_ERROR && !result->source_file)) return false;
    if (status == NN_INFERENCE_SUCCESS && tensor) {
        if (!result->dtype || !result->dimensions) return false;
        for (size_t i = 0; i < tensor->count; ++i)
            if (!result->dimensions[i]) return false;
    }
    return true;
}

static Result *report_result(Evaluation *evaluation, const NNNode *node)
{
    size_t index = find_node_index(evaluation->model, node->id);
    if (index == (size_t)-1) return NULL;
    return &evaluation->report->items[index];
}

static const char *package_kind(Evaluation *evaluation, const NNNode *node)
{
    const NNPackage *package = nn_catalog_find(evaluation->catalog,
                                               node->package_id, node->package_version);
    return package ? package->kind : NULL;
}

static void scope_set_status(Evaluation *evaluation, const char *scope,
                             NNInferenceStatus status, const char *message)
{
    for (size_t i = 0; i < nn_model_node_count(evaluation->model); ++i) {
        const NNNode *node = nn_model_node_at(evaluation->model, i);
        if (strcmp(node->scope_id ? node->scope_id : "", scope ? scope : "")) continue;
        Result *result = report_result(evaluation, node);
        if (result && !result_set(result, node, status, nn_text_copy(message), NULL,
                                  NULL, NULL, 0, NULL)) evaluation->report->failed = true;
    }
}

static int incoming_compare(const void *left, const void *right)
{
    const Incoming *a = left, *b = right;
    return a->order < b->order ? -1 : a->order > b->order ? 1 : 0;
}

static int inference_subflow(lua_State *state)
{
    LuaContext *context = lua_touserdata(state, lua_upvalueindex(1));
    const char *kind = context ? package_kind(context->evaluation, context->node) : NULL;
    if (!context || !kind || strcmp(kind, "subflow"))
        return luaL_error(state, "infer_subflow is only available to subflow packages");
    if (++context->evaluation->invocations > 256 || context->depth >= 32)
        return luaL_error(state, "subflow inference limit exceeded");
    if (lua_isnil(state, 1)) {
        lua_newtable(state);
        lua_pushliteral(state, "unresolved"); lua_setfield(state, -2, "status");
        lua_pushliteral(state, "subflow input tensor is unresolved");
        lua_setfield(state, -2, "message");
        return 1;
    }
    tensor_dispose(&context->inherited_scratch);
    tensor_dispose(&context->output_scratch);
    free(context->message_scratch);
    context->message_scratch = NULL;
    free(context->cause_node_id);
    context->cause_node_id = NULL;
    if (!tensor_from_lua(state, 1, &context->inherited_scratch, true)) {
        return luaL_error(state, "infer_subflow expects a tensor");
    }
    NNInferenceStatus status = evaluate_scope(context->evaluation, context->node->id,
                                               &context->inherited_scratch, context->depth + 1,
                                               &context->output_scratch,
                                               &context->message_scratch,
                                               &context->cause_node_id);
    tensor_dispose(&context->inherited_scratch);
    if (status == NN_INFERENCE_RUNTIME_FAULT) {
        lua_pushstring(state, context->message_scratch
                           ? context->message_scratch : "nested inference runtime fault");
        return lua_error(state);
    }
    lua_newtable(state);
    if (status == NN_INFERENCE_SUCCESS) {
        lua_pushliteral(state, "success"); lua_setfield(state, -2, "status");
        tensor_push(state, context->output_scratch.dtype,
                    (const char *const *)context->output_scratch.dimensions,
                    context->output_scratch.count);
        lua_setfield(state, -2, "output");
    } else {
        lua_pushstring(state, status == NN_INFERENCE_SEMANTIC_ERROR ? "error" : "unresolved");
        lua_setfield(state, -2, "status");
        lua_pushstring(state, context->message_scratch
                           ? context->message_scratch : "nested scope is unresolved");
        lua_setfield(state, -2, "message");
    }
    free(context->message_scratch);
    context->message_scratch = NULL;
    tensor_dispose(&context->output_scratch);
    return 1;
}

static NNInferenceStatus evaluate_scope(Evaluation *evaluation, const char *scope,
                                         const Tensor *inherited, size_t depth,
                                         Tensor *scope_output, char **scope_message,
                                         char **scope_cause)
{
    const NNModel *model = evaluation->model;
    size_t count = nn_model_node_count(model);
    Tensor *outputs = calloc(count ? count : 1, sizeof(*outputs));
    size_t *indegree = calloc(count ? count : 1, sizeof(*indegree));
    bool *done = calloc(count ? count : 1, sizeof(*done));
    if (!outputs || !indegree || !done) {
        evaluation->allocation_failed = true;
        free(outputs); free(indegree); free(done);
        *scope_message = nn_text_copy("unable to allocate scope inference state");
        return NN_INFERENCE_RUNTIME_FAULT;
    }
    size_t scope_nodes = 0, inputs_n = 0, outputs_n = 0, input_index = count, output_index = count;
    for (size_t i = 0; i < count; ++i) {
        const NNNode *node = nn_model_node_at(model, i);
        if (strcmp(node->scope_id ? node->scope_id : "", scope ? scope : "")) continue;
        ++scope_nodes;
        const char *kind = package_kind(evaluation, node);
        if (kind && !strcmp(kind, "input")) { ++inputs_n; input_index = i; }
        if (kind && !strcmp(kind, "output")) { ++outputs_n; output_index = i; }
    }
    bool nested_scope = scope && *scope;
        if (!scope_nodes || (nested_scope && (!inputs_n || !outputs_n))) {
        *scope_message = nn_text_copy("subflow scope is empty or missing an Input/Output boundary");
        scope_set_status(evaluation, scope, NN_INFERENCE_UNRESOLVED, *scope_message);
        free(outputs); free(indegree); free(done); return NN_INFERENCE_UNRESOLVED;
    }
    if (nested_scope && (inputs_n != 1 || outputs_n != 1)) {
        *scope_message = nn_text_copy("subflow scope must contain exactly one immediate Input and Output");
        scope_set_status(evaluation, scope, NN_INFERENCE_SEMANTIC_ERROR, *scope_message);
        free(outputs); free(indegree); free(done); return NN_INFERENCE_SEMANTIC_ERROR;
    }
    for (size_t e = 0; e < nn_model_edge_count(model); ++e) {
        const NNEdge *edge = nn_model_edge_at(model, e);
        const NNNode *source = nn_model_find_node(model, edge->source_id);
        const NNNode *target = nn_model_find_node(model, edge->target_id);
        bool edge_claims_scope = !strcmp(edge->scope_id ? edge->scope_id : "", scope ? scope : "");
        bool touches_scope = (source && !strcmp(source->scope_id ? source->scope_id : "", scope ? scope : "")) ||
                             (target && !strcmp(target->scope_id ? target->scope_id : "", scope ? scope : ""));
        bool endpoint_mismatch = edge_claims_scope &&
            (!source || !target || strcmp(source->scope_id ? source->scope_id : "", scope ? scope : "") ||
             strcmp(target->scope_id ? target->scope_id : "", scope ? scope : ""));
        if (endpoint_mismatch || (touches_scope && !edge_claims_scope)) {
            *scope_message = nn_text_copy("scope contains a cross-scope or malformed edge");
            scope_set_status(evaluation, scope, NN_INFERENCE_SEMANTIC_ERROR, *scope_message);
            free(outputs); free(indegree); free(done); return NN_INFERENCE_SEMANTIC_ERROR;
        }
    }
    for (size_t e = 0; e < nn_model_edge_count(model); ++e) {
        const NNEdge *edge = nn_model_edge_at(model, e);
        if (strcmp(edge->scope_id ? edge->scope_id : "", scope ? scope : "")) continue;
        size_t target = find_node_index(model, edge->target_id);
        const NNNode *target_node = target != (size_t)-1 ? nn_model_node_at(model, target) : NULL;
        if (target_node && !strcmp(target_node->scope_id ? target_node->scope_id : "", scope ? scope : "")) ++indegree[target];
    }
    NNInferenceStatus output_status = NN_INFERENCE_UNRESOLVED;
    NNInferenceStatus scope_failure = NN_INFERENCE_UNRESOLVED;
    char *failure_message = NULL;
    for (size_t step = 0; step < scope_nodes; ++step) {
        size_t index = count;
        for (size_t i = 0; i < count; ++i) {
            const NNNode *candidate = nn_model_node_at(model, i);
            if (!done[i] && !strcmp(candidate->scope_id ? candidate->scope_id : "", scope ? scope : "") && !indegree[i]) { index = i; break; }
        }
        if (index == count) break;
        done[index] = true;
        const NNNode *node = nn_model_node_at(model, index);
        const NNPackage *package = nn_catalog_find(evaluation->catalog, node->package_id, node->package_version);
        size_t incoming_count = 0;
        for (size_t e = 0; e < nn_model_edge_count(model); ++e) {
            const NNEdge *edge = nn_model_edge_at(model, e);
            if (!strcmp(edge->target_id, node->id) && !strcmp(edge->scope_id ? edge->scope_id : "", scope ? scope : "")) ++incoming_count;
        }
        Tensor *inputs = calloc(incoming_count ? incoming_count : 1, sizeof(*inputs));
        Incoming *incoming = calloc(incoming_count ? incoming_count : 1, sizeof(*incoming));
        size_t used = 0; bool missing = !inputs || !incoming, malformed = false;
        if (!inputs || !incoming) evaluation->allocation_failed = true;
        if (inputs && incoming) for (size_t e = 0; e < nn_model_edge_count(model); ++e) {
            const NNEdge *edge = nn_model_edge_at(model, e);
            if (strcmp(edge->target_id, node->id) || strcmp(edge->scope_id ? edge->scope_id : "", scope ? scope : "")) continue;
            size_t order = e;
            if (package && !strcmp(package->kind, "join") && !nn_join_handle_order(edge->target_handle_id, &order)) { malformed = true; break; }
            incoming[used++] = (Incoming){ .edge = edge, .order = order };
        }
        if (package && !strcmp(package->kind, "join") && !malformed) {
            qsort(incoming, used, sizeof(*incoming), incoming_compare);
            for (size_t i = 1; i < used; ++i) if (incoming[i-1].order == incoming[i].order) malformed = true;
        }
        size_t edge_count = used; used = 0;
        if (inputs && incoming && !malformed) for (size_t i = 0; i < edge_count; ++i) {
            size_t source = find_node_index(model, incoming[i].edge->source_id);
            if (source == (size_t)-1 || !outputs[source].dtype) { missing = true; break; }
            inputs[used++] = outputs[source];
        }
        char *message = NULL, *source_file = NULL, *cause_node_id = NULL;
        size_t source_line = 0;
        Tensor output = {0}; NNInferenceStatus status;
        if (malformed) { status = NN_INFERENCE_SEMANTIC_ERROR; message = nn_text_copy("join target handle must be in-<positive integer>"); }
        else if (missing) { status = NN_INFERENCE_UNRESOLVED; message = nn_text_copy("an upstream tensor is unresolved"); }
        else if (!package) { status = NN_INFERENCE_RUNTIME_FAULT; message = nn_text_copy("package is absent from active catalog"); }
        else if (incoming_count == 0 && (!package->kind || strcmp(package->kind, "input"))) {
            status = NN_INFERENCE_UNRESOLVED;
            message = nn_text_copy("node has no upstream tensor");
        }
        else status = execute_rule(evaluation, node, package, depth,
                                   node == nn_model_node_at(model, input_index) ? inherited : NULL,
                                   inputs, used, &output, &message, &source_file,
                                   &source_line, &cause_node_id);
        if (status == NN_INFERENCE_UNRESOLVED && !cause_node_id && missing) {
            for (size_t e = 0; incoming && e < edge_count && !cause_node_id; ++e) {
                const NNEdge *edge = incoming[e].edge;
                const NNNode *source = nn_model_find_node(model, edge->source_id);
                Result *upstream = source ? report_result(evaluation, source) : NULL;
                if (!upstream) continue;
                if (upstream->cause_node_id) {
                    cause_node_id = nn_text_copy(upstream->cause_node_id);
                    if (!cause_node_id) evaluation->allocation_failed = true;
                }
                else if (upstream->view.status == NN_INFERENCE_SEMANTIC_ERROR ||
                         upstream->view.status == NN_INFERENCE_COMPILATION_ERROR ||
                         upstream->view.status == NN_INFERENCE_RUNTIME_FAULT ||
                         upstream->view.status == NN_INFERENCE_UNRESOLVED)
                    { cause_node_id = nn_text_copy(source->id);
                      if (!cause_node_id) evaluation->allocation_failed = true; }
            }
        }
        Result *result = report_result(evaluation, node);
        if (result && !result_set(result, node, status, message,
                                  status == NN_INFERENCE_SUCCESS ? &output : NULL,
                                  cause_node_id,
                                  source_file, source_line,
                                  cause_node_id ? "model.blocked" : NULL))
            evaluation->report->failed = true;
        if (index == output_index && scope_cause && cause_node_id) {
            *scope_cause = nn_text_copy(cause_node_id);
            if (!*scope_cause) evaluation->allocation_failed = true;
        }
        free(source_file);
        free(cause_node_id);
        if (status == NN_INFERENCE_RUNTIME_FAULT) {
            scope_failure = status;
            free(failure_message);
            failure_message = nn_text_copy(message ? message : "nested inference runtime fault");
        } else if (status == NN_INFERENCE_SEMANTIC_ERROR && scope_failure == NN_INFERENCE_UNRESOLVED) {
            scope_failure = status;
            failure_message = nn_text_copy(message ? message : "nested semantic error");
        }
        if (status == NN_INFERENCE_SUCCESS) outputs[index] = output;
        else tensor_dispose(&output);
        if (index == output_index) {
            output_status = status;
            if (status == NN_INFERENCE_SUCCESS && !tensor_copy(scope_output, &outputs[index])) {
                evaluation->allocation_failed = true;
                output_status = NN_INFERENCE_RUNTIME_FAULT;
                *scope_message = nn_text_copy("unable to copy subflow output tensor");
            }
            else *scope_message = nn_text_copy(message ? message : "subflow Output is unresolved");
        }
        free(inputs); free(incoming);
        for (size_t e = 0; e < nn_model_edge_count(model); ++e) {
            const NNEdge *edge = nn_model_edge_at(model, e);
            if (strcmp(edge->source_id, node->id) || strcmp(edge->scope_id ? edge->scope_id : "", scope ? scope : "")) continue;
            size_t target = find_node_index(model, edge->target_id);
            if (target != (size_t)-1 && indegree[target]) --indegree[target];
        }
    }
    if (output_index < count && !done[output_index]) {
        output_status = NN_INFERENCE_UNRESOLVED;
        *scope_message = nn_text_copy("subflow Output is disconnected or cyclic");
    }
    if (output_status != NN_INFERENCE_SUCCESS && scope_failure != NN_INFERENCE_UNRESOLVED) {
        output_status = scope_failure;
        free(*scope_message);
        *scope_message = failure_message;
        failure_message = NULL;
    }
    if (scope_cause && !*scope_cause && output_index < count &&
        output_status != NN_INFERENCE_SUCCESS) {
        const NNNode *output_node = nn_model_node_at(model, output_index);
        Result *output_result = report_result(evaluation, output_node);
        const char *root = output_result && output_result->cause_node_id
            ? output_result->cause_node_id : output_node->id;
        *scope_cause = nn_text_copy(root);
        if (!*scope_cause) evaluation->allocation_failed = true;
    }
    free(failure_message);
    for (size_t i = 0; i < count; ++i) tensor_dispose(&outputs[i]);
    free(outputs); free(indegree); free(done);
    return output_status;
}

NNInferenceReport *nn_infer_project(const NNProject *project)
{
    if (!project) return NULL;
    const NNModel *model = nn_project_model((NNProject *)project);
    size_t count = nn_model_node_count(model);
    NNInferenceReport *report = calloc(1, sizeof(*report));
    if (!report) return NULL;
    report->items = calloc(count ? count : 1, sizeof(*report->items));
    if (!report->items) { free(report); return NULL; }
    report->count = count;
    Evaluation evaluation = { .model = model,
        .catalog = nn_project_catalog(project), .report = report,
        .dataset = nn_project_active_dataset(project) };
    Tensor ignored = {0}; char *message = NULL;
    (void)evaluate_scope(&evaluation, "", NULL, 0, &ignored, &message, NULL);
    free(message); tensor_dispose(&ignored);
    for (size_t i = 0; i < count; ++i) {
        const NNNode *node = nn_model_node_at(model, i);
        if (report->items[i].id) continue;
        const char *scope = node->scope_id ? node->scope_id : "";
        Result *result = report_result(&evaluation, node);
        if (!*scope) {
            if (result && !result_set(result, node, NN_INFERENCE_UNRESOLVED,
                                    nn_text_copy("node could not be evaluated"), NULL,
                                    NULL, NULL, 0, NULL)) report->failed = true;
        } else {
            const NNNode *owner = nn_model_find_node(model, scope);
            const char *owner_kind = owner ? package_kind(&evaluation, owner) : NULL;
            if (owner_kind && !strcmp(owner_kind, "subflow")) {
                /* Valid children skipped because their owner never delegated. */
                for (size_t child = 0; child < count; ++child) {
                    const NNNode *nested = nn_model_node_at(model, child);
                    if (strcmp(nested->scope_id ? nested->scope_id : "", scope)) continue;
                    Result *unseen = report_result(&evaluation, nested);
                    if (unseen && !unseen->id &&
                        !result_set(unseen, nested, NN_INFERENCE_UNRESOLVED,
                                    nn_text_copy("subflow was not invoked by its owner"), NULL,
                                    NULL, NULL, 0, NULL)) report->failed = true;
                }
            } else if (result && !result_set(result, node, NN_INFERENCE_UNRESOLVED,
                                            nn_text_copy("orphan scope has no subflow owner"), NULL,
                                            NULL, NULL, 0, NULL)) report->failed = true;
        }
    }
    /* Normalize skipped nested scopes only after all owner outcomes exist. */
    for (size_t i = 0; i < count; ++i) {
        Result *child_result = &report->items[i];
        if (!child_result->message ||
            strcmp(child_result->message, "subflow was not invoked by its owner")) continue;
        const NNNode *child = nn_model_node_at(model, i);
        const char *owner_id = child->scope_id ? child->scope_id : "";
        const NNNode *owner = *owner_id ? nn_model_find_node(model, owner_id) : NULL;
        const char *cause = NULL;
        for (size_t depth = 0; owner && depth < 32; ++depth) {
            Result *owner_result = report_result(&evaluation, owner);
            if (!owner_result || owner_result->view.status == NN_INFERENCE_SUCCESS) break;
            if (owner_result->cause_node_id) {
                cause = owner_result->cause_node_id;
                break;
            }
            if (owner_result->message &&
                !strcmp(owner_result->message, "subflow was not invoked by its owner")) {
                const char *parent_id = owner->scope_id ? owner->scope_id : "";
                owner = *parent_id ? nn_model_find_node(model, parent_id) : NULL;
                continue;
            }
            cause = owner->id;
            break;
        }
        if (cause && !result_set(child_result, child, NN_INFERENCE_UNRESOLVED,
                                 nn_text_copy("subflow was blocked by its owner"), NULL,
                                 cause, NULL, 0, "model.blocked"))
            report->failed = true;
    }
    if (report->failed || evaluation.allocation_failed) {
        nn_inference_free(report);
        return NULL;
    }
    return report;
}

void nn_inference_free(NNInferenceReport *report)
{
    if (!report) return;
    for (size_t i = 0; i < report->count; ++i) {
        Result *item = &report->items[i];
        free(item->id); free(item->message); free(item->dtype);
        free(item->cause_node_id); free(item->source_file);
        free_dimensions(item->dimensions, item->view.dimension_count);
    }
    free(report->items); free(report);
}

size_t nn_inference_count(const NNInferenceReport *report) { return report ? report->count : 0; }
const NNInferenceResult *nn_inference_at(const NNInferenceReport *report, size_t index)
{ return report && index < report->count ? &report->items[index].view : NULL; }

const char *nn_inference_category(NNInferenceStatus status)
{
    switch (status) {
    case NN_INFERENCE_SUCCESS: return "success";
    case NN_INFERENCE_COMPILATION_ERROR: return "lua-compilation";
    case NN_INFERENCE_SEMANTIC_ERROR: return "model";
    case NN_INFERENCE_UNRESOLVED: return "incomplete";
    case NN_INFERENCE_RUNTIME_FAULT: return "internal";
    }
    return "internal";
}

const char *nn_inference_severity(NNInferenceStatus status)
{
    switch (status) {
    case NN_INFERENCE_SUCCESS: return "info";
    case NN_INFERENCE_COMPILATION_ERROR:
    case NN_INFERENCE_SEMANTIC_ERROR: return "error";
    case NN_INFERENCE_UNRESOLVED: return "warning";
    case NN_INFERENCE_RUNTIME_FAULT: return "internal";
    }
    return "internal";
}

size_t nn_inference_error_line(const char *message)
{
    if (!message) return 0;
    for (const char *p = message; *p; ++p) {
        if (*p != ':') continue;
        const char *digits = p + 1;
        if (*digits < '0' || *digits > '9') continue;
        size_t line = 0;
        do {
            unsigned digit = (unsigned)(*digits - '0');
            if (line > (SIZE_MAX - digit) / 10) return 0;
            line = line * 10 + digit;
            ++digits;
        } while (*digits >= '0' && *digits <= '9');
        if (*digits == ':' && line) return line;
    }
    return 0;
}
