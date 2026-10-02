#define _POSIX_C_SOURCE 200809L
#include "inference_internal.h"
#include "utils/utils.h"

#include <lua.h>
#include <stdlib.h>

void *nn_inference_limited_alloc(void *data, void *ptr, size_t old_size, size_t new_size)
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

void nn_inference_instruction_hook(lua_State *state, lua_Debug *debug)
{
    (void)debug;
    LuaBudget *budget = *(LuaBudget **)lua_getextraspace(state);
    budget->instructions += 1000;
    if (budget->instructions > LUA_INSTRUCTION_LIMIT) {
        lua_pushliteral(state, "inference instruction limit exceeded");
        lua_error(state);
    }
}

void nn_inference_note_host_allocation_failure(lua_State *state)
{
    LuaBudget *budget = *(LuaBudget **)lua_getextraspace(state);
    if (budget) budget->host_allocation_failed = true;
}

static LuaBudget *nn_inference_lua_budget(lua_State *state)
{
    return *(LuaBudget **)lua_getextraspace(state);
}

void *nn_inference_lua_temp_calloc(lua_State *state, size_t count, size_t size)
{
    if (size && count > SIZE_MAX / size) {
        nn_inference_note_host_allocation_failure(state);
        return NULL;
    }
    LuaBudget *budget = nn_inference_lua_budget(state);
    if (!budget || budget->temporary_count == LUA_TEMP_ALLOCATION_LIMIT) {
        nn_inference_note_host_allocation_failure(state);
        return NULL;
    }
    void *memory = calloc(count ? count : 1, size ? size : 1);
    if (!memory) {
        nn_inference_note_host_allocation_failure(state);
        return NULL;
    }
    budget->temporary[budget->temporary_count++] = memory;
    return memory;
}

void *nn_inference_lua_temp_copy(lua_State *state, const char *text)
{
    if (!text) return NULL;
    void *copy = nn_text_copy(text);
    if (!copy) {
        nn_inference_note_host_allocation_failure(state);
        return NULL;
    }
    LuaBudget *budget = nn_inference_lua_budget(state);
    if (!budget || budget->temporary_count == LUA_TEMP_ALLOCATION_LIMIT) {
        free(copy);
        nn_inference_note_host_allocation_failure(state);
        return NULL;
    }
    budget->temporary[budget->temporary_count++] = copy;
    return copy;
}

void nn_inference_lua_temp_free(lua_State *state, void *memory)
{
    if (!memory) return;
    LuaBudget *budget = nn_inference_lua_budget(state);
    if (budget) {
        for (size_t i = 0; i < budget->temporary_count; ++i) {
            if (budget->temporary[i] != memory) continue;
            budget->temporary[i] = budget->temporary[--budget->temporary_count];
            break;
        }
    }
    free(memory);
}

void nn_inference_lua_temp_detach(lua_State *state, void *memory)
{
    LuaBudget *budget = nn_inference_lua_budget(state);
    if (!memory || !budget) return;
    for (size_t i = 0; i < budget->temporary_count; ++i) {
        if (budget->temporary[i] != memory) continue;
        budget->temporary[i] = budget->temporary[--budget->temporary_count];
        return;
    }
}

void nn_inference_lua_temp_dimensions_free(lua_State *state, char **dimensions, size_t count)
{
    if (!dimensions) return;
    for (size_t i = 0; i < count; ++i) nn_inference_lua_temp_free(state, dimensions[i]);
    nn_inference_lua_temp_free(state, dimensions);
}

void nn_inference_lua_temp_cleanup(LuaBudget *budget)
{
    while (budget->temporary_count)
        free(budget->temporary[--budget->temporary_count]);
}