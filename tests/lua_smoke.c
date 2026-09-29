#include <lauxlib.h>
#include <lua.h>
#include <lualib.h>

int main(void)
{
    lua_State *state = luaL_newstate();
    if (state == NULL)
        return 1;

    int status = luaL_loadstring(state, "return 6 * 7");
    if (status == LUA_OK)
        status = lua_pcall(state, 0, 1, 0);

    int result = status == LUA_OK && lua_isinteger(state, -1)
        && lua_tointeger(state, -1) == 42 ? 0 : 1;
    lua_close(state);
    return result;
}
