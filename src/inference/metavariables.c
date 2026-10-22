#include "inference_internal.h"

#include <lauxlib.h>
#include <lua.h>
#include <math.h>
#include <limits.h>
#include <stdio.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "model/value_json.h"
#include "utils/utils.h"

#define PARAMETER_RESOLVE_DEPTH_LIMIT 64u
#define PARAMETER_BINDING_LIMIT (64u * 1024u)
#define VALUE_ITEM_LIMIT 1024u

typedef struct Resolver Resolver;
typedef struct {
    const NNNode *node;
    const NNPackage *package;
    NNParameter *values;
    unsigned char *state;
    size_t count;
} ParameterFrame;

struct Resolver {
    const NNCatalog *catalog;
    const NNMetavariable *meta;
    size_t meta_count;
    unsigned resolve_depth;
    NNParameterResolutionStatus status;
    char error[512];
};

static void set_error_status(Resolver *resolver, NNParameterResolutionStatus status,
                             const char *message)
{
    if (!resolver->error[0]) {
        resolver->status = status;
        nn_error_set(resolver->error, sizeof(resolver->error), message);
    }
}

static void set_error(Resolver *resolver, const char *message)
{
    set_error_status(resolver, NN_PARAMETER_RESOLUTION_SEMANTIC, message);
}

static void set_errorf(Resolver *resolver, NNParameterResolutionStatus status,
                       const char *format, ...)
{
    if (resolver->error[0]) return;
    resolver->status = status;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(resolver->error, sizeof(resolver->error), format, arguments);
    va_end(arguments);
}

static const NNValue *object_value(const NNValue *object, const char *key)
{
    return nn_value_member(object, key);
}

static bool value_to_lua(lua_State *state, const NNValue *value, unsigned depth)
{
    if (!value || depth > 64 || !lua_checkstack(state, 3)) return false;
    switch (value->type) {
    case NN_VALUE_BOOL: lua_pushboolean(state, value->as.boolean); return true;
    case NN_VALUE_INT: lua_pushinteger(state, (lua_Integer)value->as.integer); return true;
    case NN_VALUE_REAL: lua_pushnumber(state, value->as.real); return true;
    case NN_VALUE_STRING: if (!value->as.string) return false; lua_pushstring(state, value->as.string); return true;
    case NN_VALUE_ARRAY:
        if (value->as.array.count > VALUE_ITEM_LIMIT) return false;
        lua_createtable(state, (int)value->as.array.count, 0);
        for (size_t i = 0; i < value->as.array.count; ++i) {
            if (!value_to_lua(state, &value->as.array.items[i], depth + 1)) return false;
            lua_seti(state, -2, (lua_Integer)i + 1);
        }
        return true;
    case NN_VALUE_OBJECT:
        if (value->as.object.count > VALUE_ITEM_LIMIT) return false;
        lua_createtable(state, 0, (int)value->as.object.count);
        for (size_t i = 0; i < value->as.object.count; ++i) {
            if (!value->as.object.items[i].key ||
                !value_to_lua(state, &value->as.object.items[i].value, depth + 1)) return false;
            lua_setfield(state, -2, value->as.object.items[i].key);
        }
        return true;
    default: return false;
    }
}

typedef struct { size_t items; size_t bytes; bool allocation_failed; } LuaConversion;

static bool value_from_lua(lua_State *state, int index, NNValue *out, unsigned depth,
                           LuaConversion *budget)
{
    if (depth > 64 || ++budget->items > 4096 || !lua_checkstack(state, 3)) return false;
    index = lua_absindex(state, index);
    memset(out, 0, sizeof(*out));
    if (lua_isboolean(state, index)) {
        out->type = NN_VALUE_BOOL; out->as.boolean = lua_toboolean(state, index);
    } else if (lua_isinteger(state, index)) {
        out->type = NN_VALUE_INT; out->as.integer = (long long)lua_tointeger(state, index);
    } else if (lua_type(state, index) == LUA_TNUMBER) {
        out->type = NN_VALUE_REAL; out->as.real = lua_tonumber(state, index);
        if (!isfinite(out->as.real)) return false;
    } else if (lua_type(state, index) == LUA_TSTRING) {
        size_t length = 0;
        const char *text = lua_tolstring(state, index, &length);
        if (length > 1024 * 1024 - budget->bytes || memchr(text, '\0', length)) return false;
        budget->bytes += length;
        out->type = NN_VALUE_STRING; out->as.string = nn_text_copy(text);
        if (!out->as.string) { budget->allocation_failed = true; return false; }
    } else if (lua_istable(state, index)) {
        size_t raw_length = lua_rawlen(state, index);
        if (raw_length > VALUE_ITEM_LIMIT || raw_length > 4096 - budget->items) return false;
        bool array = true;
        size_t keys = 0;
        lua_pushnil(state);
        while (lua_next(state, index)) {
            ++keys;
            if (!lua_isinteger(state, -2) || lua_tointeger(state, -2) < 1 ||
                (size_t)lua_tointeger(state, -2) > raw_length) array = false;
            lua_pop(state, 1);
        }
        if (array && raw_length == keys) {
            out->type = NN_VALUE_ARRAY;
            out->as.array.items = calloc(raw_length ? raw_length : 1, sizeof(NNValue));
            if (!out->as.array.items) { budget->allocation_failed = true; return false; }
            out->as.array.count = raw_length;
            for (size_t i = 0; i < raw_length; ++i) {
                lua_rawgeti(state, index, (lua_Integer)i + 1);
                bool okay = value_from_lua(state, -1, &out->as.array.items[i], depth + 1, budget);
                lua_pop(state, 1);
                if (!okay) { nn_value_dispose(out); return false; }
            }
        } else {
            if (keys > VALUE_ITEM_LIMIT || keys > 4096 - budget->items) return false;
            out->type = NN_VALUE_OBJECT;
            out->as.object.items = calloc(keys ? keys : 1, sizeof(NNParameter));
            if (!out->as.object.items) { budget->allocation_failed = true; return false; }
            lua_pushnil(state);
            while (lua_next(state, index)) {
                if (lua_type(state, -2) != LUA_TSTRING) { lua_pop(state, 2); nn_value_dispose(out); return false; }
                size_t key_length = 0;
                const char *key = lua_tolstring(state, -2, &key_length);
                if (!key_length || key_length > 1024 * 1024 - budget->bytes || memchr(key, '\0', key_length)) {
                    lua_pop(state, 2); nn_value_dispose(out); return false;
                }
                budget->bytes += key_length;
                size_t i = out->as.object.count++;
                NNParameter *entry = &out->as.object.items[i];
                entry->key = nn_text_copy(key);
                if (!entry->key) budget->allocation_failed = true;
                bool okay = entry->key && value_from_lua(state, -1, &entry->value, depth + 1, budget);
                lua_pop(state, 1);
                if (!okay) { lua_pop(state, 1); nn_value_dispose(out); return false; }
            }
        }
    } else return false;
    return true;
}

static const NNMetavariable *find_meta(const Resolver *resolver, const char *name)
{
    for (size_t i = 0; i < resolver->meta_count; ++i)
        if (!strcmp(resolver->meta[i].name, name)) return &resolver->meta[i];
    return NULL;
}

static size_t frame_index(const ParameterFrame *frame, const char *key)
{
    for (size_t i = 0; i < frame->count; ++i)
        if (!strcmp(frame->package->parameters[i].key, key)) return i;
    return SIZE_MAX;
}

static bool resolve_parameter(Resolver *resolver, ParameterFrame *frame, size_t index,
                              unsigned depth);

static int read_only(lua_State *state)
{
    (void)state;
    return luaL_error(state, "parameter binding views are read-only");
}

static int meta_lookup(lua_State *state)
{
    Resolver *resolver = lua_touserdata(state, lua_upvalueindex(1));
    const char *key = lua_type(state, 2) == LUA_TSTRING ? lua_tostring(state, 2) : NULL;
    const NNMetavariable *variable = key ? find_meta(resolver, key) : NULL;
    if (!variable) {
        set_errorf(resolver, NN_PARAMETER_RESOLUTION_UNRESOLVED,
                   "unknown metavariable '%s'", key ? key : "?");
        return luaL_error(state, "%s", resolver->error);
    }
    if (!value_to_lua(state, &variable->value, 0))
        return luaL_error(state, "cannot expose metavariable '%s'", key);
    return 1;
}

static int parameter_lookup(lua_State *state)
{
    ParameterFrame *frame = lua_touserdata(state, lua_upvalueindex(1));
    Resolver *resolver = lua_touserdata(state, lua_upvalueindex(2));
    const char *key = lua_type(state, 2) == LUA_TSTRING ? lua_tostring(state, 2) : NULL;
    size_t index = key ? frame_index(frame, key) : SIZE_MAX;
    if (index == SIZE_MAX) {
        set_errorf(resolver, NN_PARAMETER_RESOLUTION_UNRESOLVED,
                   "unknown parameter field '%s'", key ? key : "?");
        return luaL_error(state, "%s", resolver->error);
    }
    if (!resolve_parameter(resolver, frame, index, resolver->resolve_depth + 1))
        return luaL_error(state, "%s", resolver->error[0] ? resolver->error : "parameter resolution failed");
    if (!value_to_lua(state, &frame->values[index].value, 0))
        return luaL_error(state, "cannot expose parameter field '%s'", key);
    return 1;
}

static void push_readonly_proxy(lua_State *state, lua_CFunction lookup, void *owner,
                                Resolver *resolver)
{
    lua_newtable(state);
    lua_newtable(state);
    lua_pushlightuserdata(state, owner);
    if (resolver) {
        lua_pushlightuserdata(state, resolver);
        lua_pushcclosure(state, lookup, 2);
    } else {
        lua_pushcclosure(state, lookup, 1);
    }
    lua_setfield(state, -2, "__index");
    lua_pushcfunction(state, read_only); lua_setfield(state, -2, "__newindex");
    lua_pushboolean(state, 0); lua_setfield(state, -2, "__metatable");
    lua_setmetatable(state, -2);
}

typedef struct {
    Resolver *resolver;
    ParameterFrame *frame;
    LuaBudget *budget;
    const char *source;
    size_t length;
    NNValue result;
    LuaConversion conversion;
    NNParameterResolutionStatus status;
} LuaBindingCall;

static int run_binding(lua_State *state)
{
    LuaBindingCall *call = lua_touserdata(state, 1);
    lua_pushnil(state); lua_setglobal(state, "rawset");
    lua_pushnil(state); lua_setglobal(state, "setmetatable");
    lua_pushnil(state); lua_setglobal(state, "print");
    lua_pushnil(state); lua_setglobal(state, "warn");
    push_readonly_proxy(state, meta_lookup, call->resolver, NULL);
    lua_setglobal(state, "meta");
    push_readonly_proxy(state, parameter_lookup, call->frame, call->resolver);
    lua_setglobal(state, "params");
    size_t expression_length = call->length + sizeof("return ") - 1;
    char *expression = malloc(expression_length + 1);
    if (!expression) {
        call->status = NN_PARAMETER_RESOLUTION_ALLOCATION;
        call->budget->host_allocation_failed = true;
        return luaL_error(state, "out of memory compiling parameter binding");
    }
    memcpy(expression, "return ", 7);
    memcpy(expression + 7, call->source, call->length);
    expression[expression_length] = '\0';
    int status = luaL_loadbufferx(state, expression, expression_length,
                                  "parameter binding", "t");
    free(expression);
    if (status != LUA_OK) {
        call->status = status == LUA_ERRSYNTAX
            ? NN_PARAMETER_RESOLUTION_COMPILATION : NN_PARAMETER_RESOLUTION_RUNTIME;
        return lua_error(state);
    }
    status = lua_pcall(state, 0, 1, 0);
    if (status != LUA_OK) {
        call->status = call->resolver->status == NN_PARAMETER_RESOLUTION_SUCCESS
            ? NN_PARAMETER_RESOLUTION_RUNTIME : call->resolver->status;
        return lua_error(state);
    }
    if (!value_from_lua(state, -1, &call->result, 0, &call->conversion)) {
        call->status = call->conversion.allocation_failed
            ? NN_PARAMETER_RESOLUTION_ALLOCATION : NN_PARAMETER_RESOLUTION_SEMANTIC;
        if (call->conversion.allocation_failed) call->budget->host_allocation_failed = true;
        return luaL_error(state, "parameter Lua result is not a bounded NNValue");
    }
    return 0;
}

static bool evaluate_lua(Resolver *resolver, ParameterFrame *frame, const char *source,
                         NNValue *result)
{
    size_t length = strlen(source);
    if (!length || length > PARAMETER_BINDING_LIMIT) {
        set_error(resolver, "parameter Lua binding is empty or exceeds 64 KiB");
        return false;
    }
    LuaBudget budget = {0};
    lua_State *state = lua_newstate(nn_inference_limited_alloc, &budget, 0);
    if (!state) {
        set_error_status(resolver, budget.host_allocation_failed ? NN_PARAMETER_RESOLUTION_ALLOCATION : NN_PARAMETER_RESOLUTION_RUNTIME,
                         "unable to create bounded Lua state"); return false;
    }
    *(LuaBudget **)lua_getextraspace(state) = &budget;
    if (!nn_inference_protected_lua_initialize(state)) {
        lua_close(state); set_error_status(resolver, budget.host_allocation_failed ? NN_PARAMETER_RESOLUTION_ALLOCATION : NN_PARAMETER_RESOLUTION_RUNTIME,
                                           "unable to initialize bounded Lua state"); return false;
    }
    lua_sethook(state, nn_inference_instruction_hook, LUA_MASKCOUNT, 1000);
    LuaBindingCall call = { .resolver = resolver, .frame = frame, .budget = &budget,
                            .source = source, .length = length };
    if (!lua_checkstack(state, 2)) {
        set_error_status(resolver, budget.host_allocation_failed ? NN_PARAMETER_RESOLUTION_ALLOCATION : NN_PARAMETER_RESOLUTION_RUNTIME,
                         "unable to grow bounded Lua stack");
        lua_close(state); nn_inference_lua_temp_cleanup(&budget); return false;
    }
    lua_pushcfunction(state, run_binding);
    lua_pushlightuserdata(state, &call);
    int status = lua_pcall(state, 1, 0, 0);
    bool okay = status == LUA_OK;
    if (!okay) {
        const char *message = lua_type(state, -1) == LUA_TSTRING ? lua_tostring(state, -1) : NULL;
        set_error_status(resolver, call.status ? call.status : NN_PARAMETER_RESOLUTION_RUNTIME,
                         message ? message : "parameter Lua evaluation failed");
    }
    if (budget.host_allocation_failed) {
        if (!resolver->error[0])
            set_error_status(resolver, NN_PARAMETER_RESOLUTION_ALLOCATION,
                             "out of memory converting parameter Lua result");
        else resolver->status = NN_PARAMETER_RESOLUTION_ALLOCATION;
        okay = false;
    }
    if (okay) *result = call.result;
    else nn_value_dispose(&call.result);
    lua_close(state);
    nn_inference_lua_temp_cleanup(&budget);
    return okay;
}

static bool default_value(const NNParameterDef *definition, NNValue *value)
{
    memset(value, 0, sizeof(*value));
    if (definition->has_default) {
        switch (definition->default_value.type) {
        case NN_PARAMETER_BOOLEAN: value->type = NN_VALUE_BOOL; value->as.boolean = definition->default_value.as.boolean; return true;
        case NN_PARAMETER_INTEGER: value->type = NN_VALUE_INT; value->as.integer = definition->default_value.as.integer; return true;
        case NN_PARAMETER_NUMBER: value->type = NN_VALUE_REAL; value->as.real = definition->default_value.as.number; return true;
        case NN_PARAMETER_STRING: value->type = NN_VALUE_STRING; value->as.string = nn_text_copy(definition->default_value.as.string); return value->as.string != NULL;
        case NN_PARAMETER_JSON: {
            const char *text = definition->default_value.as.string;
            yyjson_doc *doc = text ? yyjson_read(text, strlen(text), 0) : NULL;
            bool okay = doc && nn_value_from_json(yyjson_doc_get_root(doc), value);
            if (doc) yyjson_doc_free(doc);
            return okay;
        }
        default: return false;
        }
    }
    return false;
}

static bool exact_integer(double number, long long *integer)
{
    if (!isfinite(number) || trunc(number) != number || number < (double)LLONG_MIN ||
        number >= -(double)LLONG_MIN) return false;
    long long converted = (long long)number;
    if ((double)converted != number) return false;
    *integer = converted;
    return true;
}

static bool resolve_reference(Resolver *resolver, const NNNode *node,
                              const NNParameterDef *destination, const NNValue *raw,
                              NNValue *resolved, unsigned depth)
{
    const NNValue *id = object_value(raw, "id"), *version = object_value(raw, "version");
    const NNValue *given = object_value(raw, "parameters");
    bool empty_parameters = given && given->type == NN_VALUE_ARRAY && given->as.array.count == 0;
    if (!id || id->type != NN_VALUE_STRING || !version || version->type != NN_VALUE_STRING ||
        !given || (given->type != NN_VALUE_OBJECT && !empty_parameters) || raw->as.object.count != 3) {
        set_error(resolver, "stereotype binding requires id, version and parameters"); return false;
    }
    const NNPackage *package = nn_catalog_resolve(resolver->catalog, id->as.string, version->as.string);
    if (!package) {
        set_error_status(resolver, NN_PARAMETER_RESOLUTION_UNRESOLVED,
                         "stereotype binding package is not active"); return false;
    }
    if ((package->kind && !strcmp(package->kind, "subflow")) ||
        (destination->kind && (!package->kind || strcmp(package->kind, destination->kind)))) {
        set_error(resolver, "stereotype binding does not resolve to a compatible active package"); return false;
    }
    if (depth >= PARAMETER_RESOLVE_DEPTH_LIMIT) { set_error_status(resolver, NN_PARAMETER_RESOLUTION_RUNTIME, "parameter resolution depth limit exceeded (64)"); return false; }
    ParameterFrame frame = { .node = node, .package = package, .count = package->parameter_count };
    frame.values = calloc(frame.count ? frame.count : 1, sizeof(NNParameter));
    frame.state = calloc(frame.count ? frame.count : 1, sizeof(unsigned char));
    if (!frame.values || !frame.state) { free(frame.values); free(frame.state); set_error_status(resolver, NN_PARAMETER_RESOLUTION_ALLOCATION, "out of memory resolving stereotype parameters"); return false; }
    for (size_t i = 0; !empty_parameters && i < given->as.object.count; ++i)
        if (frame_index(&frame, given->as.object.items[i].key) == SIZE_MAX) {
            set_errorf(resolver, NN_PARAMETER_RESOLUTION_UNRESOLVED,
                       "unknown referenced parameter '%s'", given->as.object.items[i].key); goto fail;
        }
    for (size_t i = 0; i < frame.count; ++i) {
        const NNParameterDef *def = &package->parameters[i];
        const NNValue *input = empty_parameters ? NULL : nn_value_member(given, def->key);
        frame.values[i].key = nn_text_copy(def->key);
        if (!frame.values[i].key) { set_error_status(resolver, NN_PARAMETER_RESOLUTION_ALLOCATION, "out of memory resolving stereotype parameters"); goto fail; }
        if (!input && !default_value(def, &frame.values[i].value)) {
            if (def->has_default)
                set_error_status(resolver, NN_PARAMETER_RESOLUTION_ALLOCATION, "out of memory copying referenced parameter default");
            else
                set_errorf(resolver, NN_PARAMETER_RESOLUTION_UNRESOLVED,
                           "missing referenced parameter '%s' at node '%s'", def->key, node->id);
            goto fail;
        }
        if (input && !nn_value_copy(&frame.values[i].value, input)) { set_error_status(resolver, NN_PARAMETER_RESOLUTION_ALLOCATION, "out of memory copying stereotype parameter"); goto fail; }
    }
    /* Each frame tracks its own lazy sibling state, including cycle detection. */
    for (size_t i = 0; i < frame.count; ++i)
        if (!resolve_parameter(resolver, &frame, i, depth + 1)) goto fail;
    resolved->type = NN_VALUE_OBJECT;
    resolved->as.object.items = calloc(3, sizeof(NNParameter));
    if (!resolved->as.object.items) { set_error_status(resolver, NN_PARAMETER_RESOLUTION_ALLOCATION, "out of memory resolving stereotype reference"); goto fail; }
    resolved->as.object.count = 3;
    resolved->as.object.items[0].key = nn_text_copy("id");
    resolved->as.object.items[0].value.type = NN_VALUE_STRING;
    resolved->as.object.items[0].value.as.string = nn_text_copy(id->as.string);
    resolved->as.object.items[1].key = nn_text_copy("version");
    resolved->as.object.items[1].value.type = NN_VALUE_STRING;
    resolved->as.object.items[1].value.as.string = nn_text_copy(version->as.string);
    resolved->as.object.items[2].key = nn_text_copy("parameters");
    resolved->as.object.items[2].value.type = NN_VALUE_OBJECT;
    resolved->as.object.items[2].value.as.object.items = frame.values;
    resolved->as.object.items[2].value.as.object.count = frame.count;
    frame.values = NULL;
    free(frame.state);
    if (!resolved->as.object.items[0].key || !resolved->as.object.items[0].value.as.string ||
        !resolved->as.object.items[1].key || !resolved->as.object.items[1].value.as.string ||
        !resolved->as.object.items[2].key) {
        nn_value_dispose(resolved); set_error_status(resolver, NN_PARAMETER_RESOLUTION_ALLOCATION, "out of memory building stereotype reference"); return false;
    }
    return true;
fail:
    free(frame.state);
    nn_catalog_parameters_free(frame.values, frame.count);
    return false;
}

static bool resolve_parameter(Resolver *resolver, ParameterFrame *frame, size_t index,
                              unsigned depth)
{
    if (depth > PARAMETER_RESOLVE_DEPTH_LIMIT) { set_error_status(resolver, NN_PARAMETER_RESOLUTION_RUNTIME, "parameter resolution depth limit exceeded (64)"); return false; }
    if (frame->state[index] == 2) return true;
    const NNParameterDef *definition = &frame->package->parameters[index];
    if (frame->state[index] == 1) {
        set_errorf(resolver, NN_PARAMETER_RESOLUTION_SEMANTIC,
                   "parameter dependency cycle at node '%s' field '%s'", frame->node->id, definition->key); return false;
    }
    frame->state[index] = 1;
    unsigned previous_depth = resolver->resolve_depth;
    resolver->resolve_depth = depth;
    NNValue *value = &frame->values[index].value;
    const NNValue *tag = value->type == NN_VALUE_OBJECT && value->as.object.count == 1 && value->as.object.items
        ? &value->as.object.items[0].value : NULL;
    const char *tag_name = value->type == NN_VALUE_OBJECT && value->as.object.count == 1 && value->as.object.items
        ? value->as.object.items[0].key : NULL;
    if (tag_name && (!strcmp(tag_name, "$ref") || !strcmp(tag_name, "$lua"))) {
        if (!tag || tag->type != NN_VALUE_STRING || !tag->as.string ||
            strlen(tag->as.string) > PARAMETER_BINDING_LIMIT) {
            set_errorf(resolver, NN_PARAMETER_RESOLUTION_SEMANTIC,
                       "invalid binding at node '%s' field '%s'", frame->node->id, definition->key); resolver->resolve_depth = previous_depth; return false;
        }
        NNValue result = {0};
        bool okay = false;
        if (!strcmp(tag_name, "$ref")) {
            const NNMetavariable *variable = find_meta(resolver, tag->as.string);
            if (!variable) set_errorf(resolver, NN_PARAMETER_RESOLUTION_UNRESOLVED,
                                      "unknown metavariable '%s' at node '%s' field '%s'", tag->as.string, frame->node->id, definition->key);
            else if (!strcmp(definition->type, "stereotype")) okay = resolve_reference(resolver, frame->node, definition, &variable->value, &result, depth + 1);
            else okay = nn_value_copy(&result, &variable->value);
        } else if (!strcmp(definition->type, "stereotype")) {
            NNValue expression = {0};
            okay = evaluate_lua(resolver, frame, tag->as.string, &expression);
            if (okay) okay = resolve_reference(resolver, frame->node, definition, &expression, &result, depth + 1);
            nn_value_dispose(&expression);
        } else okay = evaluate_lua(resolver, frame, tag->as.string, &result);
        if (!okay) {
            if (!resolver->error[0]) set_errorf(resolver, NN_PARAMETER_RESOLUTION_SEMANTIC,
                                               "unable to resolve node '%s' field '%s'", frame->node->id, definition->key);
            else if (!strstr(resolver->error, "node '")) {
                char detail[512]; snprintf(detail, sizeof(detail), "%s", resolver->error);
                snprintf(resolver->error, sizeof(resolver->error), "node '%s' field '%s': %.400s", frame->node->id, definition->key, detail);
            }
            nn_value_dispose(&result); resolver->resolve_depth = previous_depth; return false;
        }
        if (!strcmp(definition->type, "integer") && result.type == NN_VALUE_REAL) {
            long long integer;
            if (!exact_integer(result.as.real, &integer)) {
                nn_value_dispose(&result); set_errorf(resolver, NN_PARAMETER_RESOLUTION_SEMANTIC,
                    "fractional or out-of-range integer at node '%s' field '%s'", frame->node->id, definition->key); resolver->resolve_depth = previous_depth; return false;
            }
            result.type = NN_VALUE_INT; result.as.integer = integer;
        } else if (!strcmp(definition->type, "number") && result.type == NN_VALUE_INT) {
            double number = (double)result.as.integer;
            if (number < (double)LLONG_MIN || number >= -(double)LLONG_MIN || (long long)number != result.as.integer) { nn_value_dispose(&result); set_error(resolver, "integer cannot be represented exactly as a number"); resolver->resolve_depth = previous_depth; return false; }
            result.type = NN_VALUE_REAL; result.as.real = number;
        }
        char validation[256] = "";
        if (!nn_catalog_validate_value(resolver->catalog, definition, &result, validation, sizeof(validation))) {
            nn_value_dispose(&result);
            set_errorf(resolver, NN_PARAMETER_RESOLUTION_SEMANTIC,
                       "node '%s' field '%s': %s", frame->node->id, definition->key, validation[0] ? validation : "wrong resolved type");
            resolver->resolve_depth = previous_depth; return false;
        }
        nn_value_dispose(value); *value = result;
    } else if (!strcmp(definition->type, "stereotype")) {
        NNValue reference = {0};
        if (!resolve_reference(resolver, frame->node, definition, value, &reference, depth + 1)) { resolver->resolve_depth = previous_depth; return false; }
        nn_value_dispose(value); *value = reference;
    } else {
        if (!strcmp(definition->type, "number") && value->type == NN_VALUE_INT) {
            double number = (double)value->as.integer;
            if (number < (double)LLONG_MIN || number >= -(double)LLONG_MIN ||
                (long long)number != value->as.integer) {
                set_error(resolver, "integer cannot be represented exactly as a number");
                resolver->resolve_depth = previous_depth; return false;
            }
            value->type = NN_VALUE_REAL; value->as.real = number;
        }
        if (!nn_catalog_validate_value(resolver->catalog, definition, value,
                                       resolver->error, sizeof(resolver->error))) {
        char detail[512]; snprintf(detail, sizeof(detail), "%s", resolver->error[0] ? resolver->error : "invalid literal");
            resolver->error[0] = '\0';
            if (strstr(detail, "node '")) set_errorf(resolver, NN_PARAMETER_RESOLUTION_SEMANTIC, "%s", detail);
            else set_errorf(resolver, NN_PARAMETER_RESOLUTION_SEMANTIC,
                            "node '%s' field '%s': %.400s", frame->node->id, definition->key, detail);
            resolver->resolve_depth = previous_depth; return false;
        }
    }
    frame->state[index] = 2;
    resolver->resolve_depth = previous_depth;
    return true;
}

bool nn_inference_resolve_reference(const NNProject *project, const NNNode *node,
                                    const NNParameterDef *definition,
                                    const NNValue *value, NNValue *resolved,
                                    char *error, size_t capacity)
{
    if (error && capacity) error[0] = '\0';
    if (!project || !node || !definition || !value || !resolved)
        return nn_fail(error, capacity, "invalid stereotype parameter resolution request");
    Resolver resolver = { .catalog = nn_project_catalog(project) };
    resolver.meta = nn_project_metavariables(project, &resolver.meta_count);
    if (!resolve_reference(&resolver, node, definition, value, resolved, 0)) {
        nn_error_set(error, capacity, resolver.error[0] ? resolver.error : "unable to resolve stereotype parameters");
        return false;
    }
    return true;
}

NNParameterResolutionStatus nn_inference_resolve_parameters_typed(
    const NNProject *project, const char *node_id, NNParameter **parameters,
    size_t *count, char *error, size_t capacity)
{
    if (error && capacity) error[0] = '\0';
    if (parameters) *parameters = NULL;
    if (count) *count = 0;
    if (!project || !node_id || !parameters || !count)
    {
        nn_fail(error, capacity, "invalid parameter resolution request");
        return NN_PARAMETER_RESOLUTION_SEMANTIC;
    }
    const NNModel *model = nn_project_model((NNProject *)project);
    const NNNode *node = nn_model_find_node(model, node_id);
    const NNCatalog *catalog = nn_project_catalog(project);
    const NNPackage *package = node ? nn_catalog_find(catalog, node->package_id, node->package_version) : NULL;
    if (!node || !package) {
        nn_fail(error, capacity, "node package is not active");
        return NN_PARAMETER_RESOLUTION_UNRESOLVED;
    }
    Resolver resolver = { .catalog = catalog };
    resolver.meta = nn_project_metavariables(project, &resolver.meta_count);
    ParameterFrame frame = { .node = node, .package = package, .count = package->parameter_count };
    frame.values = calloc(frame.count ? frame.count : 1, sizeof(NNParameter));
    frame.state = calloc(frame.count ? frame.count : 1, sizeof(unsigned char));
    if (!frame.values || !frame.state) {
        free(frame.values); free(frame.state);
        nn_fail(error, capacity, "out of memory resolving parameters");
        return NN_PARAMETER_RESOLUTION_ALLOCATION;
    }
    for (size_t i = 0; i < frame.count; ++i) {
        const NNParameterDef *definition = &package->parameters[i];
        const NNValue *provided = NULL;
        for (size_t j = 0; j < node->parameter_count; ++j)
            if (!strcmp(node->parameters[j].key, definition->key)) { provided = &node->parameters[j].value; break; }
        frame.values[i].key = nn_text_copy(definition->key);
        bool okay = frame.values[i].key != NULL;
        if (!okay) set_error_status(&resolver, NN_PARAMETER_RESOLUTION_ALLOCATION, "out of memory allocating parameter key");
        else if (provided && !nn_value_copy(&frame.values[i].value, provided)) {
            set_error_status(&resolver, NN_PARAMETER_RESOLUTION_ALLOCATION, "out of memory copying parameter"); okay = false;
        } else if (!provided && !default_value(definition, &frame.values[i].value)) {
            if (definition->has_default) set_error_status(&resolver, NN_PARAMETER_RESOLUTION_ALLOCATION, "out of memory copying parameter default");
            else set_errorf(&resolver, NN_PARAMETER_RESOLUTION_UNRESOLVED,
                "missing required parameter '%s' at node '%s' field '%s'",
                definition->key, node->id, definition->key);
            okay = false;
        }
        if (!okay) break;
    }
    if (!resolver.error[0]) {
        for (size_t i = 0; i < frame.count; ++i)
            if (!resolve_parameter(&resolver, &frame, i, 0)) break;
    }
    free(frame.state);
    if (resolver.error[0]) {
        nn_catalog_parameters_free(frame.values, frame.count);
        nn_error_set(error, capacity, resolver.error);
        return resolver.status;
    }
    *parameters = frame.values;
    *count = frame.count;
    return NN_PARAMETER_RESOLUTION_SUCCESS;
}

bool nn_inference_resolve_parameters(const NNProject *project, const char *node_id,
                                     NNParameter **parameters, size_t *count,
                                     char *error, size_t capacity)
{
    return nn_inference_resolve_parameters_typed(project, node_id, parameters,
        count, error, capacity) == NN_PARAMETER_RESOLUTION_SUCCESS;
}
