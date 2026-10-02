#define _POSIX_C_SOURCE 200809L
#include "inference_internal.h"
#include "utils/utils.h"

#include <lauxlib.h>
#include <lua.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static int inference_subflow(lua_State *state);
static bool set_parameters(lua_State *state, const NNNode *node);

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
        nn_inference_tensor_push(state, context->inherited->dtype,
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
        char **dims = nn_inference_lua_temp_calloc(state, count ? count : 1, sizeof(*dims));
        if (!dims) break;
        bool valid = true;
        for (size_t d = 0; d < count; ++d) {
            const NNValue *value = &slot->shape.as.array.items[d];
            char text[32];
            if (value->type == NN_VALUE_INT) snprintf(text, sizeof(text), "%lld", value->as.integer);
            else if (value->type == NN_VALUE_STRING) snprintf(text, sizeof(text), "%s", value->as.string);
            else { valid = false; break; }
            dims[d] = nn_inference_lua_temp_copy(state, text);
            if (!dims[d]) { nn_inference_note_host_allocation_failure(state); valid = false; }
        }
        if (!valid) { nn_inference_lua_temp_dimensions_free(state, dims, count); break; }
        lua_newtable(state); lua_pushliteral(state, "success"); lua_setfield(state, -2, "status");
        nn_inference_tensor_push(state, slot->dtype, (const char *const *)dims, count);
        lua_setfield(state, -2, "output"); nn_inference_lua_temp_dimensions_free(state, dims, count);
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
    const char *kind = nn_inference_package_kind(context->evaluation, context->node);
    if (kind && !strcmp(kind, "subflow")) {
        lua_pushlightuserdata(state, context);
        lua_pushcclosure(state, inference_subflow, 1);
        lua_setfield(state, -2, "infer_subflow");
    }
    lua_setglobal(state, "services");
}

int nn_inference_invoke_rule(lua_State *state)
{
    RuleCall *call = lua_touserdata(state, 2);
    if (!call || !call->context) return luaL_error(state, "invalid inference call context");
    lua_pushvalue(state, 1);
    lua_newtable(state);
    lua_createtable(state, (int)call->input_count, 0);
    for (size_t i = 0; i < call->input_count; ++i) {
        nn_inference_tensor_push(state, call->inputs[i].dtype,
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

NNInferenceStatus nn_inference_execute_rule(Evaluation *evaluation, const NNNode *node,
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
    lua_State *state = lua_newstate(nn_inference_limited_alloc, &budget, 0);
    if (!state) { free(source); *message = nn_text_copy("unable to create bounded Lua state"); return NN_INFERENCE_RUNTIME_FAULT; }
    *(LuaBudget **)lua_getextraspace(state) = &budget;
    if (!nn_inference_protected_lua_initialize(state)) {
        lua_close(state);
        free(source);
        *message = nn_text_copy("unable to initialize bounded Lua state");
        return NN_INFERENCE_RUNTIME_FAULT;
    }
    lua_sethook(state, nn_inference_instruction_hook, LUA_MASKCOUNT, 1000);
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
        lua_pushlightuserdata(state, &nn_inference_invoke_rule_registry_key);
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
    nn_inference_lua_temp_detach(state, context.inherited_scratch.dtype);
    nn_inference_lua_temp_detach(state, context.inherited_scratch.dimensions);
    for (size_t i = 0; i < context.inherited_scratch.count; ++i)
        nn_inference_lua_temp_detach(state, context.inherited_scratch.dimensions[i]);
    nn_inference_tensor_dispose(&context.inherited_scratch);
    nn_inference_tensor_dispose(&context.output_scratch[0]);
    nn_inference_tensor_dispose(&context.output_scratch[1]);
    free(context.message_scratch);
    NNInferenceStatus result = compilation_error ? NN_INFERENCE_COMPILATION_ERROR
                                                  : NN_INFERENCE_RUNTIME_FAULT;
    if (status == LUA_OK && lua_istable(state, -1)) {
        ExtractResult extracted = { .outputs = output,
                                    .output_count = package->output_count,
                                    .package = package,
                                    .terminal = package->kind &&
                                        (!strcmp(package->kind, "output") ||
                                         !strcmp(package->kind, "loss-output")),
                                    .message = message,
                                    .status = NN_INFERENCE_RUNTIME_FAULT };
        lua_pushlightuserdata(state, &nn_inference_extract_result_registry_key);
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
            for (size_t i = 0; i < (extracted.terminal ? 1 : package->output_count); ++i) {
                nn_inference_tensor_detach_lua_temporaries(state, &output[i]);
                nn_inference_tensor_dispose(&output[i]);
            }
        }
        if (result != NN_INFERENCE_SUCCESS) {
            for (size_t i = 0; i < (extracted.terminal ? 1 : package->output_count); ++i) {
                nn_inference_tensor_detach_lua_temporaries(state, &output[i]);
                nn_inference_tensor_dispose(&output[i]);
            }
        }
    } else if (status != LUA_OK && !compilation_error) {
        const char *error = lua_type(state, -1) == LUA_TSTRING
            ? lua_tostring(state, -1) : NULL;
        *message = nn_text_copy(error ? error : "Lua inference failed");
    }
    if (budget.host_allocation_failed) evaluation->allocation_failed = true;
    lua_close(state);
    nn_inference_lua_temp_cleanup(&budget);
    return result;
}

static int inference_subflow(lua_State *state)
{
    LuaContext *context = lua_touserdata(state, lua_upvalueindex(1));
    const char *kind = context ? nn_inference_package_kind(context->evaluation, context->node) : NULL;
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
    nn_inference_tensor_dispose(&context->inherited_scratch);
    nn_inference_tensor_dispose(&context->output_scratch[0]);
    nn_inference_tensor_dispose(&context->output_scratch[1]);
    free(context->message_scratch);
    context->message_scratch = NULL;
    free(context->cause_node_id);
    context->cause_node_id = NULL;
    if (!nn_inference_tensor_from_lua(state, 1, &context->inherited_scratch, true)) {
        return luaL_error(state, "infer_subflow expects a tensor");
    }
    NNInferenceStatus status = nn_inference_evaluate_scope(context->evaluation, context->node->id,
                                               &context->inherited_scratch, context->depth + 1,
                                                context->output_scratch,
                                               &context->message_scratch,
                                               &context->cause_node_id);
    nn_inference_tensor_dispose(&context->inherited_scratch);
    if (status == NN_INFERENCE_RUNTIME_FAULT) {
        lua_pushstring(state, context->message_scratch
                           ? context->message_scratch : "nested inference runtime fault");
        return lua_error(state);
    }
    lua_newtable(state);
    if (status == NN_INFERENCE_SUCCESS) {
        lua_pushliteral(state, "success"); lua_setfield(state, -2, "status");
        const NNPackage *package = nn_catalog_find(context->evaluation->catalog,
            context->node->package_id, context->node->package_version);
        if (package && package->output_count == 1) {
            nn_inference_tensor_push(state, context->output_scratch[0].dtype,
                        (const char *const *)context->output_scratch[0].dimensions,
                        context->output_scratch[0].count);
            lua_setfield(state, -2, "output");
        } else if (package && package->output_count == 2) {
            lua_newtable(state);
            for (size_t i = 0; i < package->output_count; ++i) {
                nn_inference_tensor_push(state, context->output_scratch[i].dtype,
                            (const char *const *)context->output_scratch[i].dimensions,
                            context->output_scratch[i].count);
                lua_setfield(state, -2, package->outputs[i].id);
            }
            lua_setfield(state, -2, "outputs");
        }
    } else {
        lua_pushstring(state, status == NN_INFERENCE_SEMANTIC_ERROR ? "error" : "unresolved");
        lua_setfield(state, -2, "status");
        lua_pushstring(state, context->message_scratch
                           ? context->message_scratch : "nested scope is unresolved");
        lua_setfield(state, -2, "message");
    }
    free(context->message_scratch);
    context->message_scratch = NULL;
    nn_inference_tensor_dispose(&context->output_scratch[0]);
    nn_inference_tensor_dispose(&context->output_scratch[1]);
    return 1;
}