#include "inference_internal.h"
#include "utils/utils.h"

#include <lauxlib.h>
#include <lua.h>
#include <stdlib.h>
#include <string.h>

bool nn_inference_tensor_from_lua(lua_State *state, int index, Tensor *tensor, bool tracked)
{
    if (!nn_inference_tensor_read(state, index)) return false;
    nn_inference_raw_getfield(state, index, "dtype");
    tensor->dtype = tracked ? nn_inference_lua_temp_copy(state, lua_tostring(state, -1))
                            : nn_text_copy(lua_tostring(state, -1));
    lua_pop(state, 1);
    if (!tensor->dtype) {
        if (!tracked) nn_inference_note_host_allocation_failure(state);
        return false;
    }
    char **dims = NULL;
    size_t count = 0;
    if (!nn_inference_read_shape(state, index, &dims, &count, tracked)) {
        if (tracked) nn_inference_lua_temp_free(state, tensor->dtype);
        else free(tensor->dtype);
        memset(tensor, 0, sizeof(*tensor));
        return false;
    }
    tensor->dimensions = dims;
    tensor->count = count;
    if (tracked) {
        nn_inference_lua_temp_detach(state, tensor->dtype);
        nn_inference_lua_temp_detach(state, dims);
        for (size_t i = 0; i < count; ++i) nn_inference_lua_temp_detach(state, dims[i]);
    }
    return true;
}

void nn_inference_tensor_detach_lua_temporaries(lua_State *state, Tensor *tensor)
{
    nn_inference_lua_temp_detach(state, tensor->dtype);
    nn_inference_lua_temp_detach(state, tensor->dimensions);
    for (size_t i = 0; i < tensor->count; ++i)
        nn_inference_lua_temp_detach(state, tensor->dimensions[i]);
}

static void extraction_message(lua_State *state, ExtractResult *result, const char *message)
{
    *result->message = nn_text_copy(message);
    if (!*result->message) nn_inference_note_host_allocation_failure(state);
}

int nn_inference_extract_rule_result(lua_State *state)
{
    ExtractResult *result = lua_touserdata(state, 2);
    if (!result || !result->outputs || !result->message)
        return luaL_error(state, "invalid inference result extraction context");
    nn_inference_raw_getfield(state, 1, "status");
    const char *kind = lua_type(state, -1) == LUA_TSTRING ? lua_tostring(state, -1) : NULL;
    lua_pop(state, 1);
    result->status = NN_INFERENCE_RUNTIME_FAULT;
    if (kind && !strcmp(kind, "success")) {
        nn_inference_raw_getfield(state, 1, "output");
        bool has_output = !lua_isnil(state, -1);
        lua_pop(state, 1);
        nn_inference_raw_getfield(state, 1, "outputs");
        bool has_outputs = !lua_isnil(state, -1);
        if (result->terminal) {
            lua_pop(state, 1);
            if (!has_output || has_outputs) {
                result->status = NN_INFERENCE_SEMANTIC_ERROR;
                extraction_message(state, result, "terminal rule must return only its consumed output tensor");
            } else {
                nn_inference_raw_getfield(state, 1, "output");
                if (nn_inference_tensor_from_lua(state, -1, &result->outputs[0], true)) {
                    nn_inference_tensor_detach_lua_temporaries(state, &result->outputs[0]);
                    result->status = NN_INFERENCE_SUCCESS;
                } else {
                    result->status = NN_INFERENCE_SEMANTIC_ERROR;
                    extraction_message(state, result, "rule returned an invalid tensor");
                }
                lua_pop(state, 1);
            }
        } else if (has_output && has_outputs) {
            lua_pop(state, 1);
            result->status = NN_INFERENCE_SEMANTIC_ERROR;
            extraction_message(state, result, "rule must return either output or outputs, not both");
        } else if (has_output) {
            lua_pop(state, 1);
            nn_inference_raw_getfield(state, 1, "output");
            if (result->output_count != 1) {
                lua_pop(state, 1);
                result->status = NN_INFERENCE_SEMANTIC_ERROR;
                extraction_message(state, result, "multi-output rule cannot use output shorthand");
            } else if (nn_inference_tensor_from_lua(state, -1, &result->outputs[0], true)) {
                nn_inference_tensor_detach_lua_temporaries(state, &result->outputs[0]);
                result->status = NN_INFERENCE_SUCCESS;
                lua_pop(state, 1);
            } else {
                result->status = NN_INFERENCE_SEMANTIC_ERROR;
                extraction_message(state, result, "rule returned an invalid tensor");
                lua_pop(state, 1);
            }
        } else if (has_outputs && lua_istable(state, -1)) {
            int map = lua_absindex(state, -1);
            size_t seen = 0;
            bool valid = true;
            for (size_t i = 0; i < result->package->output_count; ++i) {
                nn_inference_raw_getfield(state, map, result->package->outputs[i].id);
                if (lua_isnil(state, -1)) valid = false;
                else if (!nn_inference_tensor_from_lua(state, -1, &result->outputs[i], true)) valid = false;
                else nn_inference_tensor_detach_lua_temporaries(state, &result->outputs[i]);
                lua_pop(state, 1);
            }
            lua_pushnil(state);
            while (lua_next(state, map)) {
                size_t key_length = 0;
                const char *key = lua_type(state, -2) == LUA_TSTRING
                    ? lua_tolstring(state, -2, &key_length) : NULL;
                bool known = false;
                for (size_t i = 0; key && i < result->package->output_count; ++i) {
                    size_t id_length = strlen(result->package->outputs[i].id);
                    if (key_length == id_length && !memcmp(key,
                        result->package->outputs[i].id, id_length)) known = true;
                }
                if (!known) valid = false;
                ++seen;
                lua_pop(state, 1);
            }
            if (seen != result->package->output_count) valid = false;
            lua_pop(state, 1);
            if (valid) result->status = NN_INFERENCE_SUCCESS;
            else {
                for (size_t i = 0; i < result->package->output_count; ++i) {
                    nn_inference_tensor_detach_lua_temporaries(state, &result->outputs[i]);
                    nn_inference_tensor_dispose(&result->outputs[i]);
                }
                result->status = NN_INFERENCE_SEMANTIC_ERROR;
                extraction_message(state, result, "rule output map does not match declared handles");
            }
        } else {
            lua_pop(state, 1);
            result->status = NN_INFERENCE_SEMANTIC_ERROR;
            extraction_message(state, result, "rule result is missing a valid output or outputs map");
        }
        if (result->status != NN_INFERENCE_SUCCESS && !*result->message)
            extraction_message(state, result, "rule output is invalid");
    } else if (kind && (!strcmp(kind, "error") || !strcmp(kind, "unresolved"))) {
        nn_inference_raw_getfield(state, 1, "message");
        if (lua_type(state, -1) == LUA_TSTRING) {
            *result->message = nn_text_copy(lua_tostring(state, -1));
            if (!*result->message) nn_inference_note_host_allocation_failure(state);
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
