#define _POSIX_C_SOURCE 200809L
#include "inference.h"

#include "catalog.h"
#include "model.h"
#include "project.h"

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

typedef struct { size_t used; unsigned instructions; } LuaBudget;
typedef struct { char *dtype; char **dimensions; size_t count; } Tensor;
typedef struct { const NNEdge *edge; size_t order; } Incoming;
typedef struct {
    const NNDataset *dataset;
    const NNNode *node;
} LuaContext;
typedef struct { NNInferenceResult view; char *id, *message, *dtype; char **dimensions; } Result;

struct NNInferenceReport { Result *items; size_t count; };

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

static char *copy_string(const char *text)
{
    if (!text) return NULL;
    size_t length = strlen(text) + 1;
    char *copy = malloc(length);
    if (copy) memcpy(copy, text, length);
    return copy;
}

static int tensor_read(lua_State *state, int index)
{
    if (!lua_istable(state, index)) return 0;
    lua_getfield(state, index, "shape");
    bool valid = lua_istable(state, -1);
    lua_pop(state, 1);
    lua_getfield(state, index, "dtype");
    valid = valid && lua_isstring(state, -1);
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
    lua_getfield(state, 1, "shape");
    lua_pushinteger(state, (lua_Integer)lua_rawlen(state, -1));
    return 1;
}

static int tensor_dtype(lua_State *state)
{
    if (!tensor_read(state, 1)) return push_error(state, "expected a tensor");
    lua_getfield(state, 1, "dtype");
    return 1;
}

static int dimension_index(lua_State *state, int tensor, lua_Integer dim, size_t *count)
{
    lua_getfield(state, tensor, "shape");
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
    lua_getfield(state, 1, "shape");
    lua_geti(state, -1, index);
    return 1;
}

static bool read_shape(lua_State *state, int tensor, char ***dims, size_t *count)
{
    lua_getfield(state, tensor, "shape");
    *count = lua_rawlen(state, -1);
    if (*count > 64) { lua_pop(state, 1); return false; }
    *dims = calloc(*count ? *count : 1, sizeof(**dims));
    if (!*dims) { lua_pop(state, 1); return false; }
    for (size_t i = 0; i < *count; ++i) {
        lua_geti(state, -1, (lua_Integer)i + 1);
        if (lua_isinteger(state, -1)) {
            char text[32]; snprintf(text, sizeof(text), "%lld", (long long)lua_tointeger(state, -1));
            (*dims)[i] = copy_string(text);
        } else if (lua_isstring(state, -1)) (*dims)[i] = copy_string(lua_tostring(state, -1));
        lua_pop(state, 1);
        if (!(*dims)[i]) { lua_pop(state, 1); return false; }
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
    if (!read_shape(state, 1, &dims, &count)) return push_error(state, "invalid tensor shape");
    size_t unused;
    int index = dimension_index(state, 1, lua_tointeger(state, 2), &unused);
    if (!index || lua_tointeger(state, 3) < 1) {
        free_dimensions(dims, count);
        return push_error(state, "dimension or size is invalid");
    }
    char text[32]; snprintf(text, sizeof(text), "%lld", (long long)lua_tointeger(state, 3));
    free(dims[index - 1]); dims[index - 1] = copy_string(text);
    lua_getfield(state, 1, "dtype"); const char *dtype = lua_tostring(state, -1);
    tensor_push(state, dtype, (const char *const *)dims, count);
    free_dimensions(dims, count);
    return 1;
}

static int tensor_create(lua_State *state)
{
    if (!lua_istable(state, 1) || !lua_isstring(state, 2))
        return push_error(state, "expected shape and dtype");
    size_t count = lua_rawlen(state, 1);
    if (!count || count > 64) return push_error(state, "invalid tensor rank");
    char **dims = calloc(count, sizeof(*dims));
    if (!dims) return push_error(state, "out of memory");
    for (size_t i = 0; i < count; ++i) {
        lua_geti(state, 1, (lua_Integer)i + 1);
        if (lua_isinteger(state, -1)) {
            char text[32]; snprintf(text, sizeof(text), "%lld", (long long)lua_tointeger(state, -1));
            dims[i] = copy_string(text);
        } else if (lua_isstring(state, -1)) dims[i] = copy_string(lua_tostring(state, -1));
        lua_pop(state, 1);
        if (!dims[i]) { free_dimensions(dims, count); return push_error(state, "invalid tensor dimension"); }
    }
    tensor_push(state, lua_tostring(state, 2), (const char *const *)dims, count);
    free_dimensions(dims, count);
    return 1;
}

static int tensor_flatten(lua_State *state)
{
    if (!tensor_read(state, 1) || !lua_isinteger(state, 2) || !lua_isinteger(state, 3))
        return push_error(state, "expected tensor and integer dimension range");
    char **dims = NULL; size_t count = 0;
    if (!read_shape(state, 1, &dims, &count)) return push_error(state, "invalid tensor shape");
    lua_Integer start = lua_tointeger(state, 2), end = lua_tointeger(state, 3);
    /* Package dimensions use PyTorch axes: positive axes are zero-based. */
    if (start < 0) start += (lua_Integer)count + 1;
    else ++start;
    if (end < 0) end += (lua_Integer)count + 1;
    else ++end;
    if (start < 1 || end < start || (size_t)end > count) {
        free_dimensions(dims, count); return push_error(state, "flatten dimension range is invalid");
    }
    long long product = 1;
    for (lua_Integer i = start; i <= end; ++i) {
        char *stop = NULL; long value = strtol(dims[i - 1], &stop, 10);
        if (!stop || *stop || value < 1 || product > 2147483647LL / value) {
            free_dimensions(dims, count); return push_error(state, "flatten dimensions must have a bounded numeric product");
        }
        product *= value;
    }
    char **out = calloc(count - (size_t)(end - start), sizeof(*out));
    if (!out) { free_dimensions(dims, count); return push_error(state, "out of memory"); }
    size_t n = 0;
    for (size_t i = 0; i < (size_t)start - 1; ++i) out[n++] = copy_string(dims[i]);
    char product_text[32]; snprintf(product_text, sizeof(product_text), "%lld", product);
    out[n++] = copy_string(product_text);
    for (size_t i = (size_t)end; i < count; ++i) out[n++] = copy_string(dims[i]);
    lua_getfield(state, 1, "dtype"); const char *dtype = lua_tostring(state, -1);
    tensor_push(state, dtype, (const char *const *)out, n);
    free_dimensions(out, n); free_dimensions(dims, count);
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
    const char *binding = lua_tostring(state, 1);
    if (!context || !binding || !context->dataset) {
        lua_newtable(state); lua_pushliteral(state, "unresolved"); lua_setfield(state, -2, "status");
        lua_pushliteral(state, "input binding has no selected dataset"); lua_setfield(state, -2, "message");
        return 1;
    }
    for (size_t i = 0; i < context->dataset->input_count; ++i) {
        const NNTensorSlot *slot = &context->dataset->inputs[i];
        if (strcmp(slot->name, binding)) continue;
        if (slot->shape.type != NN_VALUE_ARRAY) break;
        size_t count = slot->shape.as.array.count;
        char **dims = calloc(count ? count : 1, sizeof(*dims));
        if (!dims) break;
        bool valid = true;
        for (size_t d = 0; d < count; ++d) {
            const NNValue *value = &slot->shape.as.array.items[d];
            char text[32];
            if (value->type == NN_VALUE_INT) snprintf(text, sizeof(text), "%lld", value->as.integer);
            else if (value->type == NN_VALUE_STRING) snprintf(text, sizeof(text), "%s", value->as.string);
            else { valid = false; break; }
            dims[d] = copy_string(text); if (!dims[d]) valid = false;
        }
        if (!valid) { free_dimensions(dims, count); break; }
        lua_newtable(state); lua_pushliteral(state, "success"); lua_setfield(state, -2, "status");
        tensor_push(state, slot->dtype, (const char *const *)dims, count);
        lua_setfield(state, -2, "output"); free_dimensions(dims, count);
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
    lua_setglobal(state, "services");
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

static void set_parameters(lua_State *state, const NNNode *node)
{
    lua_createtable(state, 0, (int)node->parameter_count);
    for (size_t i = 0; i < node->parameter_count; ++i) {
        if (push_value(state, &node->parameters[i].value))
            lua_setfield(state, -2, node->parameters[i].key);
        else lua_pop(state, 1);
    }
}

static char *read_rule(const NNPackage *package, size_t *length)
{
    if (!package || !package->directory || !package->inference_file ||
        package->inference_file[0] == '/' || strstr(package->inference_file, "..")) return NULL;
    size_t a = strlen(package->directory), b = strlen(package->inference_file);
    if (a > SIZE_MAX - b - 2) return NULL;
    char *path = malloc(a + b + 2);
    if (!path) return NULL;
    snprintf(path, a + b + 2, "%s/%s", package->directory, package->inference_file);
    struct stat info;
    if (lstat(path, &info) || !S_ISREG(info.st_mode) || info.st_size < 0 ||
        (size_t)info.st_size > RULE_FILE_LIMIT) { free(path); return NULL; }
    FILE *file = fopen(path, "rb"); free(path);
    if (!file) return NULL;
    *length = (size_t)info.st_size;
    char *source = malloc(*length + 1);
    if (!source) { fclose(file); return NULL; }
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

static bool tensor_from_lua(lua_State *state, int index, Tensor *tensor)
{
    if (!tensor_read(state, index)) return false;
    lua_getfield(state, index, "dtype"); tensor->dtype = copy_string(lua_tostring(state, -1)); lua_pop(state, 1);
    char **dims = NULL;
    if (!tensor->dtype || !read_shape(state, index, &dims, &tensor->count)) return false;
    tensor->dimensions = dims;
    return true;
}

static void tensor_dispose(Tensor *tensor)
{
    free(tensor->dtype); free_dimensions(tensor->dimensions, tensor->count);
    memset(tensor, 0, sizeof(*tensor));
}

static NNInferenceStatus execute_rule(const NNProject *project, const NNNode *node,
                                      const NNPackage *package,
                                      const NNDataset *dataset,
                                      Tensor *inputs, size_t input_count,
                                      Tensor *output, char **message)
{
    size_t source_length = 0;
    char *source = read_rule(package, &source_length);
    if (!source) { *message = copy_string("inference rule unavailable or invalid"); return NN_INFERENCE_RUNTIME_FAULT; }
    LuaBudget budget = {0};
    lua_State *state = lua_newstate(limited_alloc, &budget, 0);
    if (!state) { free(source); *message = copy_string("unable to create bounded Lua state"); return NN_INFERENCE_RUNTIME_FAULT; }
    *(LuaBudget **)lua_getextraspace(state) = &budget;
    open_safe_libraries(state);
    lua_sethook(state, instruction_hook, LUA_MASKCOUNT, 1000);
    int status = luaL_loadbufferx(state, source, source_length, package->id, "t");
    free(source);
    if (status == LUA_OK) status = lua_pcall(state, 0, 1, 0);
    if (status == LUA_OK && !lua_isfunction(state, -1)) status = LUA_ERRRUN;
    LuaContext context = { .dataset = dataset, .node = node };
    if (status == LUA_OK) {
        lua_newtable(state);
        lua_createtable(state, (int)input_count, 0);
        for (size_t i = 0; i < input_count; ++i) {
            tensor_push(state, inputs[i].dtype, (const char *const *)inputs[i].dimensions, inputs[i].count);
            lua_seti(state, -2, (lua_Integer)i + 1);
        }
        lua_setfield(state, -2, "inputs");
        set_parameters(state, node);
        set_services(state, &context);
        lua_getglobal(state, "services");
        status = lua_pcall(state, 3, 1, 0);
    }
    NNInferenceStatus result = NN_INFERENCE_RUNTIME_FAULT;
    if (status == LUA_OK && lua_istable(state, -1)) {
        lua_getfield(state, -1, "status"); const char *kind = lua_tostring(state, -1); lua_pop(state, 1);
        if (kind && !strcmp(kind, "success")) {
            lua_getfield(state, -1, "output");
            if (tensor_from_lua(state, -1, output)) result = NN_INFERENCE_SUCCESS;
            else *message = copy_string("rule returned an invalid tensor");
            lua_pop(state, 1);
        } else if (kind && (!strcmp(kind, "error") || !strcmp(kind, "unresolved"))) {
            lua_getfield(state, -1, "message"); *message = copy_string(lua_tostring(state, -1)); lua_pop(state, 1);
            if (!*message) *message = copy_string("rule returned a diagnostic without a message");
            result = !strcmp(kind, "error") ? NN_INFERENCE_SEMANTIC_ERROR : NN_INFERENCE_UNRESOLVED;
        } else *message = copy_string("rule returned an invalid result");
    } else if (status != LUA_OK) {
        const char *error = lua_tostring(state, -1);
        *message = copy_string(error ? error : "Lua inference failed");
    }
    lua_close(state);
    (void)project;
    return result;
}

static size_t find_node_index(const NNModel *model, const char *id)
{
    for (size_t i = 0; i < nn_model_node_count(model); ++i)
        if (!strcmp(nn_model_node_at(model, i)->id, id)) return i;
    return (size_t)-1;
}

static void result_set(Result *result, const NNNode *node, NNInferenceStatus status,
                       char *message, Tensor *tensor)
{
    result->id = copy_string(node->id); result->message = message;
    result->view.node_id = result->id; result->view.status = status;
    result->view.message = result->message;
    if (status == NN_INFERENCE_SUCCESS && tensor) {
        result->dtype = copy_string(tensor->dtype);
        result->dimensions = calloc(tensor->count ? tensor->count : 1, sizeof(*result->dimensions));
        for (size_t i = 0; i < tensor->count && result->dimensions; ++i)
            result->dimensions[i] = copy_string(tensor->dimensions[i]);
        result->view.dtype = result->dtype;
        result->view.dimensions = (const char *const *)result->dimensions;
        result->view.dimension_count = tensor->count;
    }
}

static bool join_handle_order(const char *handle, size_t *order)
{
    if (!handle || strncmp(handle, "in-", 3) || !handle[3]) return false;
    size_t value = 0;
    for (const unsigned char *p = (const unsigned char *)handle + 3; *p; ++p) {
        if (*p < '0' || *p > '9' || value > ((size_t)-1 - (*p - '0')) / 10) return false;
        value = value * 10 + (*p - '0');
    }
    if (!value) return false;
    *order = value;
    return true;
}

static int incoming_compare(const void *left, const void *right)
{
    const Incoming *a = left, *b = right;
    return a->order < b->order ? -1 : a->order > b->order ? 1 : 0;
}

NNInferenceReport *nn_infer_project(const NNProject *project)
{
    if (!project) return NULL;
    const NNModel *model = nn_project_model((NNProject *)project);
    const NNCatalog *catalog = nn_project_catalog(project);
    size_t count = nn_model_node_count(model);
    NNInferenceReport *report = calloc(1, sizeof(*report));
    Tensor *outputs = calloc(count ? count : 1, sizeof(*outputs));
    size_t *indegree = calloc(count ? count : 1, sizeof(*indegree));
    bool *done = calloc(count ? count : 1, sizeof(*done));
    if (!report || !outputs || !indegree || !done) goto fail;
    report->items = calloc(count ? count : 1, sizeof(*report->items));
    if (!report->items) goto fail;
    const NNDataset *dataset = nn_project_active_dataset(project);
    for (size_t e = 0; e < nn_model_edge_count(model); ++e) {
        const NNEdge *edge = nn_model_edge_at(model, e);
        size_t target = find_node_index(model, edge->target_id);
        if (target != (size_t)-1) ++indegree[target];
    }
    for (size_t step = 0; step < count; ++step) {
        size_t index = count;
        for (size_t i = 0; i < count; ++i) if (!done[i] && !indegree[i]) { index = i; break; }
        if (index == count) break;
        done[index] = true;
        const NNNode *node = nn_model_node_at(model, index);
        Result *result = &report->items[report->count++];
        const NNPackage *package = nn_catalog_find(catalog, node->package_id, node->package_version);
        size_t incoming_count = 0;
        for (size_t e = 0; e < nn_model_edge_count(model); ++e)
            if (!strcmp(nn_model_edge_at(model, e)->target_id, node->id)) ++incoming_count;
        Tensor *inputs = calloc(incoming_count ? incoming_count : 1, sizeof(*inputs));
        Incoming *incoming = calloc(incoming_count ? incoming_count : 1, sizeof(*incoming));
        size_t used = 0; bool missing = inputs == NULL || incoming == NULL;
        bool malformed_join_handle = false;
        if (inputs && incoming) for (size_t e = 0; e < nn_model_edge_count(model); ++e) {
            const NNEdge *edge = nn_model_edge_at(model, e);
            if (strcmp(edge->target_id, node->id)) continue;
            size_t order = 0;
            if (package && !strcmp(package->kind, "join")) {
                if (!join_handle_order(edge->target_handle_id, &order)) {
                    malformed_join_handle = true; break;
                }
            } else order = e;
            incoming[used++] = (Incoming){ .edge = edge, .order = order };
        }
        if (package && !strcmp(package->kind, "join") && !malformed_join_handle) {
            qsort(incoming, used, sizeof(*incoming), incoming_compare);
            for (size_t i = 1; i < used; ++i)
                if (incoming[i - 1].order == incoming[i].order) malformed_join_handle = true;
        }
        size_t edge_count = used;
        used = 0;
        if (inputs && incoming && !malformed_join_handle) for (size_t i = 0; i < edge_count; ++i) {
            const NNEdge *edge = incoming[i].edge;
            size_t source = find_node_index(model, edge->source_id);
            if (source == (size_t)-1 || !outputs[source].dtype) { missing = true; break; }
            inputs[used++] = outputs[source];
        }
        char *message = NULL; Tensor output = {0}; NNInferenceStatus status;
        if (malformed_join_handle) {
            status = NN_INFERENCE_SEMANTIC_ERROR;
            message = copy_string("join target handle must be in-<positive integer>");
        } else if (missing) {
            status = NN_INFERENCE_UNRESOLVED;
            message = copy_string("an upstream tensor is unresolved");
        } else if (!package) {
            status = NN_INFERENCE_RUNTIME_FAULT; message = copy_string("package is absent from active catalog");
        } else status = execute_rule(project, node, package, dataset, inputs, used, &output, &message);
        result_set(result, node, status, message, status == NN_INFERENCE_SUCCESS ? &output : NULL);
        if (status == NN_INFERENCE_SUCCESS) outputs[index] = output;
        else tensor_dispose(&output);
        free(inputs); free(incoming);
        for (size_t e = 0; e < nn_model_edge_count(model); ++e) {
            const NNEdge *edge = nn_model_edge_at(model, e);
            if (!strcmp(edge->source_id, node->id)) {
                size_t target = find_node_index(model, edge->target_id);
                if (target != (size_t)-1 && indegree[target]) --indegree[target];
            }
        }
    }
    for (size_t i = 0; i < count; ++i) tensor_dispose(&outputs[i]);
    free(outputs); free(indegree); free(done);
    return report;
fail:
    if (outputs) { for (size_t i = 0; i < count; ++i) tensor_dispose(&outputs[i]); }
    free(outputs); free(indegree); free(done); nn_inference_free(report); return NULL;
}

void nn_inference_free(NNInferenceReport *report)
{
    if (!report) return;
    for (size_t i = 0; i < report->count; ++i) {
        Result *item = &report->items[i];
        free(item->id); free(item->message); free(item->dtype);
        free_dimensions(item->dimensions, item->view.dimension_count);
    }
    free(report->items); free(report);
}

size_t nn_inference_count(const NNInferenceReport *report) { return report ? report->count : 0; }
const NNInferenceResult *nn_inference_at(const NNInferenceReport *report, size_t index)
{ return report && index < report->count ? &report->items[index].view : NULL; }
