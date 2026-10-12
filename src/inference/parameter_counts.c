#define _POSIX_C_SOURCE 200809L
#include "inference/parameter_counts.h"
#include "inference/inference.h"
#include "inference/inference_internal.h"
#include "model/value_json.h"
#include "utils/utils.h"
#include "yyjson.h"

#include <lauxlib.h>
#include <lua.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#define COUNT_INVOCATION_LIMIT 256u
#define COUNT_INTEGER_LIMIT ((uint64_t)INT64_MAX)

typedef struct {
    char *id;
    char *status;
    char *message;
    uint64_t total;
    uint64_t trainable;
    bool has_counts;
    bool complete;
} CountRow;

typedef struct {
    CountRow *rows;
    size_t count;
    size_t capacity;
    size_t invocations;
    bool allocation_failed;
    const NNProject *project;
    const NNModel *model;
    const NNCatalog *catalog;
    const NNInferenceReport *analysis;
} CountEvaluation;

typedef struct {
    CountEvaluation *evaluation;
    const NNNode *node;
    const NNPackage *package;
    size_t depth;
    char *message;
} CountContext;

typedef struct {
    const NNEdge *edge;
    size_t order;
} CountIncoming;

typedef struct {
    CountEvaluation *evaluation;
    const NNNode *node;
    const NNPackage *package;
    size_t depth;
} CountCall;

typedef struct {
    uint64_t total;
    uint64_t trainable;
    bool complete;
} Counts;

static bool add_count(uint64_t *target, uint64_t value)
{
    if (value > COUNT_INTEGER_LIMIT - *target) return false;
    *target += value;
    return true;
}

static void row_dispose(CountRow *row)
{
    free(row->id);
    free(row->status);
    free(row->message);
    memset(row, 0, sizeof(*row));
}

static bool row_add(CountEvaluation *evaluation, CountRow row)
{
    if (evaluation->count == evaluation->capacity) {
        size_t capacity = evaluation->capacity ? evaluation->capacity * 2 : 16;
        if (capacity < evaluation->capacity || capacity > SIZE_MAX / sizeof(*evaluation->rows))
            return false;
        CountRow *grown = realloc(evaluation->rows, capacity * sizeof(*grown));
        if (!grown) return false;
        evaluation->rows = grown;
        evaluation->capacity = capacity;
    }
    evaluation->rows[evaluation->count++] = row;
    return true;
}

static void count_rows_dispose(CountEvaluation *evaluation)
{
    for (size_t i = 0; i < evaluation->count; ++i) row_dispose(&evaluation->rows[i]);
    free(evaluation->rows);
}

static const NNInferenceResult *analysis_result(const CountEvaluation *evaluation,
                                                const char *node_id)
{
    if (!evaluation->analysis || !node_id) return NULL;
    for (size_t i = 0; i < nn_inference_count(evaluation->analysis); ++i) {
        const NNInferenceResult *result = nn_inference_at(evaluation->analysis, i);
        if (result && result->node_id && !strcmp(result->node_id, node_id)) return result;
    }
    return NULL;
}

static int count_incoming_compare(const void *left, const void *right)
{
    const CountIncoming *a = left, *b = right;
    return a->order < b->order ? -1 : a->order > b->order ? 1 : 0;
}

static bool push_value(lua_State *state, const NNValue *value, unsigned depth)
{
    if (!value || depth > 64) return false;
    switch (value->type) {
    case NN_VALUE_BOOL: lua_pushboolean(state, value->as.boolean); return true;
    case NN_VALUE_INT: lua_pushinteger(state, (lua_Integer)value->as.integer); return true;
    case NN_VALUE_REAL: lua_pushnumber(state, value->as.real); return true;
    case NN_VALUE_STRING: lua_pushstring(state, value->as.string); return value->as.string != NULL;
    case NN_VALUE_ARRAY:
        if (value->as.array.count > 1024) return false;
        lua_createtable(state, (int)value->as.array.count, 0);
        for (size_t i = 0; i < value->as.array.count; ++i) {
            if (!push_value(state, &value->as.array.items[i], depth + 1)) return false;
            lua_seti(state, -2, (lua_Integer)i + 1);
        }
        return true;
    case NN_VALUE_OBJECT:
        if (value->as.object.count > 1024) return false;
        lua_createtable(state, 0, (int)value->as.object.count);
        for (size_t i = 0; i < value->as.object.count; ++i) {
            const NNParameter *item = &value->as.object.items[i];
            if (!item->key || !push_value(state, &item->value, depth + 1)) return false;
            lua_setfield(state, -2, item->key);
        }
        return true;
    default: return false;
    }
}

static bool push_default(lua_State *state, const NNParameterDef *definition)
{
    switch (definition->default_value.type) {
    case NN_PARAMETER_BOOLEAN: lua_pushboolean(state, definition->default_value.as.boolean); return true;
    case NN_PARAMETER_INTEGER: lua_pushinteger(state, (lua_Integer)definition->default_value.as.integer); return true;
    case NN_PARAMETER_NUMBER: lua_pushnumber(state, definition->default_value.as.number); return true;
    case NN_PARAMETER_STRING: lua_pushstring(state, definition->default_value.as.string); return true;
    case NN_PARAMETER_JSON: {
        const char *text = definition->default_value.as.string;
        yyjson_doc *document = text ? yyjson_read(text, strlen(text), 0) : NULL;
        NNValue value = {0};
        bool okay = document && nn_value_from_json(yyjson_doc_get_root(document), &value) &&
                    push_value(state, &value, 0);
        nn_value_dispose(&value);
        if (document) yyjson_doc_free(document);
        return okay;
    }
    default: return false;
    }
}

static bool push_parameters(lua_State *state, const NNNode *node,
                            const NNPackage *package)
{
    lua_createtable(state, 0, (int)package->parameter_count);
    for (size_t i = 0; i < package->parameter_count; ++i) {
        const NNParameterDef *definition = &package->parameters[i];
        const NNValue *provided = NULL;
        for (size_t j = 0; j < node->parameter_count; ++j)
            if (!strcmp(node->parameters[j].key, definition->key)) {
                provided = &node->parameters[j].value;
                break;
            }
        bool okay = provided ? push_value(state, provided, 0) :
                    definition->has_default ? push_default(state, definition) : false;
        if (!okay) return false;
        lua_setfield(state, -2, definition->key);
    }
    return true;
}

static bool push_tensor_shape(lua_State *state, const NNInferenceTensor *tensor)
{
    if (!tensor) return false;
    lua_createtable(state, 0, 2);
    lua_pushstring(state, tensor->dtype ? tensor->dtype : "");
    lua_setfield(state, -2, "dtype");
    lua_createtable(state, (int)tensor->dimension_count, 0);
    for (size_t i = 0; i < tensor->dimension_count; ++i) {
        const char *text = tensor->dimensions[i];
        char *end = NULL;
        long long value = text ? strtoll(text, &end, 10) : 0;
        if (text && *text && end && !*end) lua_pushinteger(state, (lua_Integer)value);
        else lua_pushstring(state, text ? text : "");
        lua_seti(state, -2, (lua_Integer)i + 1);
    }
    lua_setfield(state, -2, "shape");
    return true;
}

static bool push_context(lua_State *state, CountEvaluation *evaluation,
                         const NNNode *node)
{
    const NNNode *graph_node = nn_model_find_node(evaluation->model, node->id);
    if (graph_node) node = graph_node;
    const NNInferenceResult *result = analysis_result(evaluation, node->id);
    lua_createtable(state, 0, 3);
    int context_index = lua_absindex(state, -1);
    lua_pushstring(state, node->id);
    lua_setfield(state, -2, "node_id");
    const NNPackage *package = nn_catalog_find(evaluation->catalog,
        node->package_id, node->package_version);
    bool join = package && package->kind && !strcmp(package->kind, "join");
    size_t edge_count = 0;
    for (size_t i = 0; i < nn_model_edge_count(evaluation->model); ++i) {
        const NNEdge *edge = nn_model_edge_at(evaluation->model, i);
        if (!edge || !edge->target_id || strcmp(edge->target_id, node->id) ||
            strcmp(edge->scope_id ? edge->scope_id : "",
                   node->scope_id ? node->scope_id : "")) continue;
        ++edge_count;
    }
    CountIncoming *incoming = calloc(edge_count ? edge_count : 1, sizeof(*incoming));
    if (!incoming) {
        evaluation->allocation_failed = true;
        return false;
    }
    size_t used = 0, array_size = edge_count;
    for (size_t i = 0; i < nn_model_edge_count(evaluation->model); ++i) {
        const NNEdge *edge = nn_model_edge_at(evaluation->model, i);
        if (!edge || !edge->target_id || strcmp(edge->target_id, node->id) ||
            strcmp(edge->scope_id ? edge->scope_id : "",
                   node->scope_id ? node->scope_id : "")) continue;
        size_t order = i;
        if (join && !nn_join_handle_order(edge->target_handle_id, &order)) {
            free(incoming);
            return false;
        }
        incoming[used++] = (CountIncoming){ .edge = edge, .order = order };
        if (join && order > array_size) array_size = order;
    }
    if (join && used > 1) {
        qsort(incoming, used, sizeof(*incoming), count_incoming_compare);
        for (size_t i = 1; i < used; ++i) {
            if (incoming[i - 1].order == incoming[i].order) {
                free(incoming);
                return false;
            }
        }
    }
    if (array_size > (size_t)INT_MAX) {
        free(incoming);
        return false;
    }
    int context_inputs_index;
    lua_createtable(state, (int)array_size, 0);
    context_inputs_index = lua_absindex(state, -1);
    for (size_t i = 0; i < used; ++i) {
        const NNEdge *edge = incoming[i].edge;
        const NNInferenceResult *source = analysis_result(evaluation, edge->source_id);
        if (!source) continue;
        for (size_t j = 0; j < source->output_count; ++j) {
            const NNInferenceTensor *tensor = &source->outputs[j];
            if (edge->source_handle_id && tensor->handle_id &&
                strcmp(edge->source_handle_id, tensor->handle_id)) continue;
            if (!push_tensor_shape(state, tensor)) { free(incoming); return false; }
            lua_seti(state, context_inputs_index,
                     (lua_Integer)(join ? incoming[i].order : i + 1));
            break;
        }
    }
    free(incoming);
    lua_pushvalue(state, context_inputs_index);
    lua_setfield(state, context_index, "inputs");
    lua_pop(state, 1);
    if (result && result->output_count) {
        if (!push_tensor_shape(state, &result->outputs[0])) return false;
        lua_setfield(state, -2, "output");
    }
    return true;
}

static bool value_from_lua(lua_State *state, int index, NNValue *value, unsigned depth)
{
    if (depth > 64) return false;
    index = lua_absindex(state, index);
    if (lua_isboolean(state, index)) {
        value->type = NN_VALUE_BOOL; value->as.boolean = lua_toboolean(state, index);
    } else if (lua_isinteger(state, index)) {
        value->type = NN_VALUE_INT; value->as.integer = (long long)lua_tointeger(state, index);
    } else if (lua_type(state, index) == LUA_TNUMBER) {
        value->type = NN_VALUE_REAL; value->as.real = lua_tonumber(state, index);
    } else if (lua_type(state, index) == LUA_TSTRING) {
        value->type = NN_VALUE_STRING;
        value->as.string = nn_text_copy(lua_tostring(state, index));
        return value->as.string != NULL;
    } else if (lua_istable(state, index)) {
        size_t length = lua_rawlen(state, index);
        if (length) {
            if (length > 1024) return false;
            value->type = NN_VALUE_ARRAY;
            value->as.array.items = calloc(length, sizeof(*value->as.array.items));
            if (!value->as.array.items) return false;
            value->as.array.count = length;
            for (size_t i = 0; i < length; ++i) {
                lua_rawgeti(state, index, (lua_Integer)i + 1);
                bool okay = value_from_lua(state, -1, &value->as.array.items[i], depth + 1);
                lua_pop(state, 1);
                if (!okay) return false;
            }
        } else {
            value->type = NN_VALUE_OBJECT;
            lua_pushnil(state);
            while (lua_next(state, index)) {
                if (lua_type(state, -2) != LUA_TSTRING || value->as.object.count >= 1024) {
                    lua_pop(state, 2); return false;
                }
                size_t n = value->as.object.count;
                NNParameter *grown = realloc(value->as.object.items, (n + 1) * sizeof(*grown));
                if (!grown) { lua_pop(state, 2); return false; }
                value->as.object.items = grown;
                NNParameter *entry = &grown[n];
                memset(entry, 0, sizeof(*entry));
                ++value->as.object.count;
                entry->key = nn_text_copy(lua_tostring(state, -2));
                bool okay = entry->key && value_from_lua(state, -1, &entry->value, depth + 1);
                lua_pop(state, 1);
                if (!okay) { lua_pop(state, 1); return false; }
            }
        }
    } else return false;
    return true;
}

static bool read_rule(const NNPackage *package, char **source, size_t *length)
{
    if (!package || !package->parameters_file) return false;
    size_t a = strlen(package->directory), b = strlen(package->parameters_file);
    if (b > SIZE_MAX - 2 || a > SIZE_MAX - b - 2) return false;
    char *path = malloc(a + b + 2);
    if (!path) return false;
    snprintf(path, a + b + 2, "%s/%s", package->directory, package->parameters_file);
    struct stat info;
    if (lstat(path, &info) || !S_ISREG(info.st_mode) || info.st_size < 0 ||
        (size_t)info.st_size > RULE_FILE_LIMIT) { free(path); return false; }
    FILE *file = fopen(path, "rb");
    free(path);
    if (!file) return false;
    *length = (size_t)info.st_size;
    *source = malloc(*length + 1);
    if (!*source) { fclose(file); return false; }
    bool okay = fread(*source, 1, *length, file) == *length && !ferror(file);
    fclose(file);
    if (!okay) { free(*source); *source = NULL; return false; }
    (*source)[*length] = '\0';
    return true;
}

static Counts count_scope(CountEvaluation *evaluation, const char *scope,
                          size_t depth);
static bool invoke_package(CountEvaluation *evaluation, const NNNode *node,
                           const NNPackage *package, size_t depth,
                           Counts *counts, char **message, bool *has_rule,
                           bool *unresolved);
static int count_subflow_service(lua_State *state);
static int count_stereotype_service(lua_State *state);

static void push_count_result(lua_State *state, Counts counts)
{
    lua_newtable(state);
    lua_pushliteral(state, "success"); lua_setfield(state, -2, "status");
    lua_pushinteger(state, (lua_Integer)counts.total); lua_setfield(state, -2, "total");
    lua_pushinteger(state, (lua_Integer)counts.trainable); lua_setfield(state, -2, "trainable");
    lua_pushboolean(state, counts.complete); lua_setfield(state, -2, "complete");
}

static int invoke_count_package(lua_State *state)
{
    const CountCall *call = lua_touserdata(state, 2);
    if (!call || !call->evaluation || !call->node || !call->package)
        return luaL_error(state, "invalid parameter count call context");
    CountContext context = { .evaluation = call->evaluation, .node = call->node,
        .package = call->package, .depth = call->depth };
    if (!push_context(state, call->evaluation, call->node))
        return luaL_error(state, "unable to build parameter count context");
    if (!push_parameters(state, call->node, call->package))
        return luaL_error(state, "unable to build parameter count parameters");
    lua_newtable(state);
    if (call->package->kind && !strcmp(call->package->kind, "subflow")) {
        lua_pushlightuserdata(state, &context);
        lua_pushcclosure(state, count_subflow_service, 1);
        lua_setfield(state, -2, "count_subflow");
    }
    lua_pushlightuserdata(state, &context);
    lua_pushcclosure(state, count_stereotype_service, 1);
    lua_setfield(state, -2, "count_stereotype");
    lua_pushvalue(state, -1);
    lua_setglobal(state, "services");
    lua_pushvalue(state, 1);
    lua_insert(state, -4);
    lua_call(state, 3, 1);
    return 1;
}

static int count_subflow_service(lua_State *state)
{
    CountContext *context = lua_touserdata(state, lua_upvalueindex(1));
    if (!context || !context->package->kind || strcmp(context->package->kind, "subflow"))
        return luaL_error(state, "count_subflow is only available to subflow packages");
    if (context->depth >= 32 || ++context->evaluation->invocations > COUNT_INVOCATION_LIMIT)
        return luaL_error(state, "parameter count recursion limit exceeded");
    Counts counts = count_scope(context->evaluation, context->node->id, context->depth + 1);
    push_count_result(state, counts);
    return 1;
}

static int count_stereotype_service(lua_State *state)
{
    CountContext *context = lua_touserdata(state, lua_upvalueindex(1));
    if (!context || context->depth >= 32 ||
        ++context->evaluation->invocations > COUNT_INVOCATION_LIMIT)
        return luaL_error(state, "stereotype parameter count limit exceeded");
    NNValue reference = {0};
    NNParameter *parameters = NULL;
    size_t parameter_count = 0;
    char error[256] = "";
    if (!value_from_lua(state, 1, &reference, 0)) {
        nn_value_dispose(&reference);
        return luaL_error(state, "invalid stereotype reference");
    }
    const NNPackage *package = nn_catalog_reference(context->evaluation->catalog,
        &reference, NULL, &parameters, &parameter_count, error, sizeof(error));
    nn_value_dispose(&reference);
    if (!package) return luaL_error(state, "%s", error[0] ? error : "inactive stereotype reference");
    NNNode synthetic = { .id = context->node->id, .package_id = package->id,
        .package_version = package->version, .parameters = parameters,
        .parameter_count = parameter_count };
    Counts counts = { .complete = true };
    char *message = NULL;
    bool has_rule = false;
    bool unresolved = false;
    bool okay = invoke_package(context->evaluation, &synthetic, package,
                               context->depth + 1, &counts, &message, &has_rule,
                               &unresolved);
    nn_catalog_parameters_free(parameters, parameter_count);
    if (!okay && has_rule && !unresolved) {
        lua_pushstring(state, message ? message : "stereotype count failed");
        free(message);
        return lua_error(state);
    }
    free(message);
    if (!has_rule || unresolved) counts.complete = false;
    push_count_result(state, counts);
    return 1;
}

static bool invoke_package(CountEvaluation *evaluation, const NNNode *node,
                           const NNPackage *package, size_t depth,
                           Counts *counts, char **message, bool *has_rule,
                           bool *unresolved)
{
    *unresolved = false;
    *has_rule = package->parameters_file != NULL;
    *counts = (Counts){ .complete = true };
    if (!*has_rule) return true;
    char *source = NULL;
    size_t length = 0;
    if (!read_rule(package, &source, &length)) {
        *message = nn_text_copy("parameter count rule is unavailable or invalid");
        return false;
    }
    LuaBudget budget = {0};
    lua_State *state = lua_newstate(nn_inference_limited_alloc, &budget, 0);
    if (!state) { free(source); *message = nn_text_copy("unable to create bounded Lua state"); return false; }
    *(LuaBudget **)lua_getextraspace(state) = &budget;
    if (!nn_inference_protected_lua_initialize(state)) {
        lua_close(state); free(source);
        *message = nn_text_copy("unable to initialize bounded Lua state");
        return false;
    }
    lua_sethook(state, nn_inference_instruction_hook, LUA_MASKCOUNT, 1000);
    int status = luaL_loadbufferx(state, source, length, package->id, "t");
    free(source);
    if (status == LUA_OK) status = lua_pcall(state, 0, 1, 0);
    if (status == LUA_OK && !lua_isfunction(state, -1)) status = LUA_ERRRUN;
    CountCall call = { .evaluation = evaluation, .node = node,
                       .package = package, .depth = depth };
    if (status == LUA_OK) {
        lua_pushcfunction(state, invoke_count_package);
        lua_pushvalue(state, 1);
        lua_pushlightuserdata(state, &call);
        status = lua_pcall(state, 2, 1, 0);
    }
    if (status == LUA_OK && lua_istable(state, -1)) {
        lua_getfield(state, -1, "status");
        const char *result_status = lua_type(state, -1) == LUA_TSTRING
            ? lua_tostring(state, -1) : NULL;
        if (result_status && !strcmp(result_status, "success")) {
            lua_pop(state, 1);
            lua_getfield(state, -1, "total");
            bool valid = lua_isinteger(state, -1) && lua_tointeger(state, -1) >= 0;
            if (valid) counts->total = (uint64_t)lua_tointeger(state, -1);
            lua_pop(state, 1);
            lua_getfield(state, -1, "trainable");
            valid = valid && lua_isinteger(state, -1) && lua_tointeger(state, -1) >= 0;
            if (valid) counts->trainable = (uint64_t)lua_tointeger(state, -1);
            lua_pop(state, 1);
            valid = valid && counts->trainable <= counts->total;
            lua_getfield(state, -1, "complete");
            if (!lua_isnil(state, -1)) {
                valid = valid && lua_isboolean(state, -1);
                if (valid) counts->complete = lua_toboolean(state, -1);
            }
            lua_pop(state, 1);
            if (!valid) {
                status = LUA_ERRRUN;
                counts->total = 0;
                counts->trainable = 0;
                counts->complete = false;
                *message = nn_text_copy("parameter count rule returned malformed or out-of-range counts");
            }
        } else {
            *unresolved = result_status && !strcmp(result_status, "unresolved");
            const char *text = result_status && !strcmp(result_status, "unresolved")
                ? "parameter count is unresolved" : "parameter count rule returned an error";
            lua_pop(state, 1);
            lua_getfield(state, -1, "message");
            const char *detail = lua_type(state, -1) == LUA_TSTRING ? lua_tostring(state, -1) : NULL;
            *message = nn_text_copy(detail ? detail : text);
            lua_pop(state, 1);
            counts->complete = false;
            status = LUA_ERRRUN;
        }
    } else if (status == LUA_OK) {
        status = LUA_ERRRUN;
        *message = nn_text_copy("parameter count rule must return a result table");
    }
    if (status != LUA_OK && !*message) {
        const char *text = lua_type(state, -1) == LUA_TSTRING ? lua_tostring(state, -1) : NULL;
        *message = nn_text_copy(text ? text : "Lua parameter count failed");
    }
    if (budget.host_allocation_failed) evaluation->allocation_failed = true;
    lua_close(state);
    nn_inference_lua_temp_cleanup(&budget);
    return status == LUA_OK;
}

static Counts count_scope(CountEvaluation *evaluation, const char *scope,
                          size_t depth)
{
    Counts result = { .complete = true };
    if (depth > 32) return (Counts){ .complete = false };
    for (size_t i = 0; i < nn_model_node_count(evaluation->model); ++i) {
        const NNNode *node = nn_model_node_at(evaluation->model, i);
        if (!node || strcmp(node->scope_id ? node->scope_id : "", scope ? scope : "")) continue;
        const NNPackage *package = nn_catalog_find(evaluation->catalog,
            node->package_id, node->package_version);
        CountRow row = { .id = nn_text_copy(node->id), .complete = true };
        if (!row.id) { evaluation->allocation_failed = true; result.complete = false; continue; }
        if (!package) {
            row.status = nn_text_copy("error");
            row.message = nn_text_copy("node package is not active");
            row.complete = false;
        } else if (!package->parameters_file) {
            row.status = nn_text_copy("unknown");
            row.message = nn_text_copy("package has no parameter count rule");
            row.complete = false;
        } else {
            Counts counts = { .complete = true };
            char *message = NULL;
            bool has_rule = false;
            bool unresolved = false;
            bool okay = invoke_package(evaluation, node, package, depth,
                                       &counts, &message, &has_rule, &unresolved);
            row.total = counts.total;
            row.trainable = counts.trainable;
            row.has_counts = okay || counts.total || counts.trainable;
            row.complete = okay && counts.complete;
            row.status = nn_text_copy(okay ? (counts.complete ? "success" : "unknown") :
                unresolved ? "unresolved" : "error");
            row.message = message ? message : row.complete ? NULL :
                nn_text_copy("parameter count is incomplete");
        }
        if (!row.status || (row.message == NULL && !row.complete)) {
            row_dispose(&row); evaluation->allocation_failed = true;
            result.complete = false; continue;
        }
        if (row.has_counts) {
            if (!add_count(&result.total, row.total) ||
                !add_count(&result.trainable, row.trainable)) {
                row.has_counts = false; row.complete = false;
                free(row.status); row.status = nn_text_copy("error");
                free(row.message); row.message = nn_text_copy("aggregate parameter count overflow");
            }
        }
        if (!row.complete) result.complete = false;
        if (!row_add(evaluation, row)) {
            row_dispose(&row); evaluation->allocation_failed = true;
            result.complete = false;
        }
    }
    if (!scope || !*scope) {
        for (size_t i = 0; i < nn_model_node_count(evaluation->model); ++i) {
            const NNNode *node = nn_model_node_at(evaluation->model, i);
            if (!node || !node->scope_id || !*node->scope_id) continue;
            const NNNode *owner = nn_model_find_node(evaluation->model, node->scope_id);
            const NNPackage *owner_package = owner ? nn_catalog_find(evaluation->catalog,
                owner->package_id, owner->package_version) : NULL;
            if (owner_package && owner_package->kind &&
                !strcmp(owner_package->kind, "subflow")) continue;
            CountRow row = { .id = nn_text_copy(node->id),
                .status = nn_text_copy("error"),
                .message = nn_text_copy("node scope has no active subflow owner"),
                .complete = false };
            if (!row.id || !row.status || !row.message || !row_add(evaluation, row)) {
                row_dispose(&row);
                evaluation->allocation_failed = true;
            }
            result.complete = false;
        }
    }
    return result;
}

char *nn_parameter_counts_json(const NNProject *project,
                               const NNInferenceReport *analysis,
                               char *error, size_t capacity)
{
    if (!project) { nn_fail(error, capacity, "no active project"); return NULL; }
    CountEvaluation evaluation = { .project = project,
        .model = nn_project_model((NNProject *)project),
        .catalog = nn_project_catalog(project), .analysis = analysis };
    Counts counts = count_scope(&evaluation, "", 0);
    if (evaluation.allocation_failed) {
        count_rows_dispose(&evaluation);
        nn_fail(error, capacity, "unable to allocate parameter count report");
        return NULL;
    }
    yyjson_mut_doc *doc = yyjson_mut_doc_new(NULL);
    yyjson_mut_val *root = doc ? yyjson_mut_obj(doc) : NULL;
    yyjson_mut_val *rows = doc ? yyjson_mut_arr(doc) : NULL;
    if (!doc || !root || !rows) {
        yyjson_mut_doc_free(doc); count_rows_dispose(&evaluation);
        nn_fail(error, capacity, "unable to allocate parameter count JSON"); return NULL;
    }
    bool okay = yyjson_mut_obj_add_int(doc, root, "total", (int64_t)counts.total) &&
        yyjson_mut_obj_add_int(doc, root, "trainable", (int64_t)counts.trainable) &&
        yyjson_mut_obj_add_bool(doc, root, "complete", counts.complete);
    for (size_t i = 0; i < evaluation.count; ++i) {
        if (!okay) break;
        const CountRow *row = &evaluation.rows[i];
        yyjson_mut_val *item = yyjson_mut_obj(doc);
        if (!item || !yyjson_mut_obj_add_str(doc, item, "node_id", row->id) ||
            !yyjson_mut_obj_add_str(doc, item, "status", row->status) ||
            (row->message && !yyjson_mut_obj_add_str(doc, item, "message", row->message))) {
            okay = false;
            break;
        }
        if (row->has_counts) {
            okay = yyjson_mut_obj_add_int(doc, item, "total", (int64_t)row->total) &&
                yyjson_mut_obj_add_int(doc, item, "trainable", (int64_t)row->trainable);
        }
        if (okay) okay = yyjson_mut_arr_append(rows, item);
    }
    if (okay) okay = yyjson_mut_obj_add_val(doc, root, "rows", rows);
    if (okay) yyjson_mut_doc_set_root(doc, root);
    char *json = okay ? yyjson_mut_write(doc, YYJSON_WRITE_PRETTY, NULL) : NULL;
    yyjson_mut_doc_free(doc);
    count_rows_dispose(&evaluation);
    if (!json) { nn_fail(error, capacity, "unable to serialize parameter count report"); return NULL; }
    nn_error_set(error, capacity, "");
    return json;
}
