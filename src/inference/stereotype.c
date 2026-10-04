#define _POSIX_C_SOURCE 200809L
#include "inference_internal.h"
#include "utils/utils.h"

#include <lauxlib.h>
#include <lua.h>
#include <stdlib.h>
#include <string.h>

static bool read_lua_value(lua_State *state, int index, const char *type,
                           NNValue *value, unsigned depth, LuaContext *context);

static bool read_lua_table(lua_State *state, int index, NNValue *value,
                           bool force_object, unsigned depth, LuaContext *context)
{
    if (depth > 64) return false;
    index = lua_absindex(state, index);
    size_t count = force_object ? 0 : lua_rawlen(state, index);
    if (!force_object) {
        lua_pushnil(state);
        while (lua_next(state, index)) {
            if (!lua_isinteger(state, -2) || lua_tointeger(state, -2) < 1 ||
                (size_t)lua_tointeger(state, -2) > count) {
                count = 0;
                lua_pop(state, 2);
                force_object = true;
                break;
            }
            lua_pop(state, 1);
        }
    }
    memset(value, 0, sizeof(*value));
    if (!force_object) {
        value->type = NN_VALUE_ARRAY;
        value->as.array.items = calloc(count ? count : 1, sizeof(NNValue));
        if (!value->as.array.items) { context->evaluation->allocation_failed = true; return false; }
        value->as.array.count = count;
        for (size_t i = 0; i < count; ++i) {
            lua_rawgeti(state, index, (lua_Integer)i + 1);
            if (!read_lua_value(state, -1, NULL, &value->as.array.items[i], depth + 1, context)) {
                lua_pop(state, 1);
                nn_value_dispose(value);
                return false;
            }
            lua_pop(state, 1);
        }
        return true;
    }
    size_t fields = 0;
    lua_pushnil(state);
    while (lua_next(state, index)) {
        size_t key_length = 0;
        const char *key = lua_type(state, -2) == LUA_TSTRING
            ? lua_tolstring(state, -2, &key_length) : NULL;
        if (!key || !key[0] || strlen(key) != key_length ||
            ++fields > 1024) {
            lua_pop(state, 2);
            return false;
        }
        lua_pop(state, 1);
    }
    value->type = NN_VALUE_OBJECT;
    value->as.object.items = calloc(fields ? fields : 1, sizeof(NNParameter));
    if (!value->as.object.items) { context->evaluation->allocation_failed = true; return false; }
    lua_pushnil(state);
    while (lua_next(state, index)) {
        size_t at = value->as.object.count;
        const char *key = lua_tostring(state, -2);
        value->as.object.items[at].key = nn_text_copy(key);
        if (!value->as.object.items[at].key) context->evaluation->allocation_failed = true;
        bool parameters = !strcmp(key, "parameters");
        if (!value->as.object.items[at].key ||
            !read_lua_value(state, -1, parameters ? "json-object" : NULL,
                            &value->as.object.items[at].value, depth + 1, context)) {
            value->as.object.count = at + 1;
            lua_pop(state, 2);
            nn_value_dispose(value);
            return false;
        }
        value->as.object.count = at + 1;
        lua_pop(state, 1);
    }
    return true;
}

static bool read_lua_value(lua_State *state, int index, const char *type,
                           NNValue *value, unsigned depth, LuaContext *context)
{
    if (depth > 64) return false;
    memset(value, 0, sizeof(*value));
    if (lua_isboolean(state, index)) {
        value->type = NN_VALUE_BOOL;
        value->as.boolean = lua_toboolean(state, index);
    } else if (lua_isinteger(state, index)) {
        value->type = NN_VALUE_INT;
        value->as.integer = (long long)lua_tointeger(state, index);
    } else if (lua_type(state, index) == LUA_TNUMBER) {
        value->type = NN_VALUE_REAL;
        value->as.real = (double)lua_tonumber(state, index);
    } else if (lua_type(state, index) == LUA_TSTRING) {
        size_t length = 0;
        const char *text = lua_tolstring(state, index, &length);
        if (strlen(text) != length) return false;
        value->type = NN_VALUE_STRING;
        value->as.string = nn_text_copy(text);
        if (!value->as.string) context->evaluation->allocation_failed = true;
        return value->as.string != NULL;
    } else if (lua_istable(state, index)) {
        return read_lua_table(state, index, value,
            type && (!strcmp(type, "stereotype") || !strcmp(type, "json-object")), depth, context);
    } else return false;
    if (type && !strcmp(type, "integer") && value->type != NN_VALUE_INT) return false;
    if (type && (!strcmp(type, "string") || !strcmp(type, "dtype")) &&
        value->type != NN_VALUE_STRING) return false;
    if (type && !strcmp(type, "number") && value->type != NN_VALUE_REAL) return false;
    if (type && !strcmp(type, "boolean") && value->type != NN_VALUE_BOOL) return false;
    return true;
}

static bool read_parameter_table(lua_State *state, int table,
                                 const NNPackage *package,
                                 LuaContext *context)
{
    table = lua_absindex(state, table);
    size_t fields = 0;
    lua_pushnil(state);
    while (lua_next(state, table)) {
        size_t key_length = 0;
        const char *key = lua_type(state, -2) == LUA_TSTRING
            ? lua_tolstring(state, -2, &key_length) : NULL;
        if (!key || strlen(key) != key_length || ++fields > package->parameter_count) {
            lua_pop(state, 2);
            return false;
        }
        lua_pop(state, 1);
    }
    NNParameter *items = calloc(fields ? fields : 1, sizeof(NNParameter));
    if (!items) { context->evaluation->allocation_failed = true; return false; }
    context->stereotype_values = items;
    context->stereotype_value_count = fields;
    size_t at = 0;
    lua_pushnil(state);
    while (lua_next(state, table)) {
        const char *key = lua_tostring(state, -2);
        const NNParameterDef *definition = NULL;
        for (size_t i = 0; i < package->parameter_count; ++i)
            if (!strcmp(package->parameters[i].key, key)) {
                definition = &package->parameters[i];
                break;
            }
        items[at].key = nn_text_copy(key);
        if (!items[at].key) context->evaluation->allocation_failed = true;
        if (!definition || !items[at].key ||
            !read_lua_value(state, -1, definition->type,
                            &items[at].value, 0, context)) {
            lua_pop(state, 2);
            return false;
        }
        ++at;
        lua_pop(state, 1);
    }
    return true;
}

void nn_inference_stereotype_cleanup(LuaContext *context)
{
    if (!context) return;
    nn_catalog_parameters_free(context->stereotype_values,
                               context->stereotype_value_count);
    context->stereotype_values = NULL;
    context->stereotype_value_count = 0;
    nn_catalog_parameters_free(context->stereotype_parameters,
                               context->stereotype_parameter_count);
    context->stereotype_parameters = NULL;
    context->stereotype_parameter_count = 0;
    for (size_t i = 0; i < context->stereotype_input_count; ++i)
        nn_inference_tensor_dispose(&context->stereotype_inputs[i]);
    free(context->stereotype_inputs);
    context->stereotype_inputs = NULL;
    context->stereotype_input_count = 0;
    for (size_t i = 0; i < 2; ++i)
        nn_inference_tensor_dispose(&context->stereotype_outputs[i]);
    free(context->stereotype_message);
    context->stereotype_message = NULL;
    free(context->stereotype_source_file);
    context->stereotype_source_file = NULL;
    free(context->stereotype_cause_node_id);
    context->stereotype_cause_node_id = NULL;
}

int nn_inference_stereotype(lua_State *state)
{
    LuaContext *context = lua_touserdata(state, lua_upvalueindex(1));
    /* A rule may catch the first Lua error. Keep its owned diagnostics intact. */
    if (context && context->stereotype_failed)
        return luaL_error(state, "%s", context->stereotype_message
            ? context->stereotype_message : "nested stereotype rule failed");
    if (!context || ++context->evaluation->invocations > 256 || context->depth >= 32)
        return luaL_error(state, "stereotype inference limit exceeded");
    if (!lua_istable(state, 1) || !lua_istable(state, 2))
        return luaL_error(state, "infer_stereotype expects a reference and ordered tensor list");

    const char *id = NULL, *version = NULL;
    size_t id_length = 0, version_length = 0, reference_fields = 0;
    lua_pushliteral(state, "id"); lua_rawget(state, 1);
    if (lua_type(state, -1) == LUA_TSTRING) id = lua_tolstring(state, -1, &id_length);
    lua_pop(state, 1);
    lua_pushliteral(state, "version"); lua_rawget(state, 1);
    if (lua_type(state, -1) == LUA_TSTRING) version = lua_tolstring(state, -1, &version_length);
    lua_pop(state, 1);
    lua_pushnil(state);
    while (lua_next(state, 1)) {
        size_t key_length = 0;
        const char *key = lua_type(state, -2) == LUA_TSTRING
            ? lua_tolstring(state, -2, &key_length) : NULL;
        if (!key || strlen(key) != key_length ||
            (strcmp(key, "id") && strcmp(key, "version") &&
                     strcmp(key, "parameters"))) {
            lua_pop(state, 2);
            id = version = NULL;
            break;
        }
        ++reference_fields;
        lua_pop(state, 1);
    }
    if (!id || !version || !id_length || !version_length ||
        strlen(id) != id_length || strlen(version) != version_length || reference_fields != 3) {
        lua_newtable(state);
        lua_pushliteral(state, "error"); lua_setfield(state, -2, "status");
        lua_pushliteral(state, "invalid stereotype reference"); lua_setfield(state, -2, "message");
        return 1;
    }
    const NNPackage *package = nn_catalog_resolve(context->evaluation->catalog, id, version);
    if (!package || !package->kind || !strcmp(package->kind, "subflow") ||
        !strcmp(package->kind, "input") || !strcmp(package->kind, "output") ||
        !strcmp(package->kind, "loss-output") || !package->output_count || package->output_count > 2) {
        lua_newtable(state);
        lua_pushliteral(state, "error"); lua_setfield(state, -2, "status");
        lua_pushliteral(state, "stereotype reference does not identify an inferable package");
        lua_setfield(state, -2, "message");
        return 1;
    }

    lua_pushliteral(state, "parameters"); lua_rawget(state, 1);
    char error[256] = {0};
    bool valid = lua_istable(state, -1) &&
        read_parameter_table(state, -1, package, context) &&
        nn_catalog_parameters(context->evaluation->catalog, package,
            context->stereotype_values, context->stereotype_value_count,
            &context->stereotype_parameters, &context->stereotype_parameter_count,
            error, sizeof(error));
    lua_pop(state, 1);
    if (!valid && (strstr(error, "out of memory") || strstr(error, "normalize")))
        context->evaluation->allocation_failed = true;
    if (!valid) {
        lua_newtable(state);
        lua_pushliteral(state, "error"); lua_setfield(state, -2, "status");
        lua_pushstring(state, error[0] ? error : "invalid stereotype parameters");
        lua_setfield(state, -2, "message");
        nn_inference_stereotype_cleanup(context);
        return 1;
    }

    size_t input_count = lua_rawlen(state, 2);
    size_t input_fields = 0;
    bool ordered_inputs = true;
    lua_pushnil(state);
    while (lua_next(state, 2)) {
        if (!lua_isinteger(state, -2) || lua_tointeger(state, -2) < 1 ||
            (size_t)lua_tointeger(state, -2) > input_count) ordered_inputs = false;
        ++input_fields;
        lua_pop(state, 1);
    }
    ordered_inputs = ordered_inputs && input_fields == input_count;
    if (!ordered_inputs || input_count > 1024 ||
        (!strcmp(package->kind, "join") ? input_count < 2 : input_count != 1)) {
        lua_newtable(state);
        lua_pushliteral(state, "error"); lua_setfield(state, -2, "status");
        lua_pushliteral(state, "stereotype input count does not match package kind");
        lua_setfield(state, -2, "message");
        nn_inference_stereotype_cleanup(context);
        return 1;
    }
    context->stereotype_inputs = calloc(input_count ? input_count : 1,
                                        sizeof(*context->stereotype_inputs));
    context->stereotype_input_count = input_count;
    if (!context->stereotype_inputs) {
        context->stereotype_input_count = 0;
        nn_inference_stereotype_cleanup(context);
        context->evaluation->allocation_failed = true;
        return luaL_error(state, "out of memory preparing stereotype inputs");
    }
    bool tensors_valid = true;
    for (size_t i = 0; i < input_count; ++i) {
        lua_rawgeti(state, 2, (lua_Integer)i + 1);
        if (!nn_inference_tensor_from_lua(state, -1, &context->stereotype_inputs[i], false))
            tensors_valid = false;
        lua_pop(state, 1);
        if (!tensors_valid) break;
    }
    if (!tensors_valid) {
        lua_newtable(state);
        lua_pushliteral(state, "error"); lua_setfield(state, -2, "status");
        lua_pushliteral(state, "stereotype inputs must be tensors"); lua_setfield(state, -2, "message");
        nn_inference_stereotype_cleanup(context);
        return 1;
    }

    NNNode synthetic = *context->node;
    synthetic.package_id = package->id;
    synthetic.package_version = package->version;
    synthetic.parameters = context->stereotype_parameters;
    synthetic.parameter_count = context->stereotype_parameter_count;
    NNInferenceStatus status = nn_inference_execute_rule(context->evaluation, &synthetic,
        package, context->depth + 1, NULL, context->stereotype_inputs, input_count,
        context->stereotype_outputs, &context->stereotype_message,
        &context->stereotype_source_file, &context->stereotype_source_line,
        &context->stereotype_cause_node_id);
    if (status == NN_INFERENCE_RUNTIME_FAULT || status == NN_INFERENCE_COMPILATION_ERROR) {
        if (!context->stereotype_source_file && package->inference_file) {
            context->stereotype_source_file = nn_text_copy(package->inference_file);
            if (!context->stereotype_source_file)
                context->evaluation->allocation_failed = true;
        }
        if (!context->stereotype_source_line)
            context->stereotype_source_line = nn_inference_error_line(
                context->stereotype_message);
        context->stereotype_failed = true;
        context->stereotype_status = status;
        return luaL_error(state, "%s", context->stereotype_message
            ? context->stereotype_message : "nested stereotype rule failed");
    }

    lua_newtable(state);
    if (status == NN_INFERENCE_SUCCESS) {
        lua_pushliteral(state, "success"); lua_setfield(state, -2, "status");
        if (package->output_count == 1) {
            nn_inference_tensor_push(state, context->stereotype_outputs[0].dtype,
                (const char *const *)context->stereotype_outputs[0].dimensions,
                context->stereotype_outputs[0].count);
            lua_setfield(state, -2, "output");
        } else {
            lua_newtable(state);
            for (size_t i = 0; i < package->output_count; ++i) {
                nn_inference_tensor_push(state, context->stereotype_outputs[i].dtype,
                    (const char *const *)context->stereotype_outputs[i].dimensions,
                    context->stereotype_outputs[i].count);
                lua_setfield(state, -2, package->outputs[i].id);
            }
            lua_setfield(state, -2, "outputs");
        }
    } else {
        lua_pushstring(state, status == NN_INFERENCE_UNRESOLVED ? "unresolved" : "error");
        lua_setfield(state, -2, "status");
        lua_pushstring(state, context->stereotype_message
            ? context->stereotype_message : "stereotype inference failed");
        lua_setfield(state, -2, "message");
    }
    nn_inference_stereotype_cleanup(context);
    return 1;
}
