#include "inference_internal.h"
#include "utils/utils.h"

#include <lauxlib.h>
#include <lua.h>
#include <lualib.h>
#include <stdlib.h>
#include <string.h>

char nn_inference_invoke_rule_registry_key;
char nn_inference_extract_result_registry_key;

static void open_safe_libraries(lua_State *state)
{
    luaL_requiref(state, LUA_GNAME, luaopen_base, 1); lua_pop(state, 1);
    lua_pushnil(state); lua_setglobal(state, "dofile");
    lua_pushnil(state); lua_setglobal(state, "loadfile");
    lua_pushnil(state); lua_setglobal(state, "load");
    lua_pushnil(state); lua_setglobal(state, "collectgarbage");
    luaL_requiref(state, LUA_MATHLIBNAME, luaopen_math, 1); lua_pop(state, 1);
    nn_inference_set_tensor_functions(state);
}

static int initialize_lua(lua_State *state)
{
    open_safe_libraries(state);
    lua_pushlightuserdata(state, &nn_inference_invoke_rule_registry_key);
    lua_pushcfunction(state, nn_inference_invoke_rule);
    lua_rawset(state, LUA_REGISTRYINDEX);
    lua_pushlightuserdata(state, &nn_inference_extract_result_registry_key);
    lua_pushcfunction(state, nn_inference_extract_rule_result);
    lua_rawset(state, LUA_REGISTRYINDEX);
    return 0;
}

bool nn_inference_protected_lua_initialize(lua_State *state)
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
    lua_State *state = lua_newstate(nn_inference_limited_alloc, &budget, 0);
    if (!state) {
        nn_error_set(error, capacity, "unable to create bounded Lua state");
        return false;
    }
    *(LuaBudget **)lua_getextraspace(state) = &budget;
    if (!nn_inference_protected_lua_initialize(state)) {
        nn_error_set(error, capacity, "unable to initialize bounded Lua state");
        lua_close(state);
        return false;
    }
    lua_sethook(state, nn_inference_instruction_hook, LUA_MASKCOUNT, 1000);
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