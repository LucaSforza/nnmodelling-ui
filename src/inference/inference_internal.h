#ifndef NN_INFERENCE_INTERNAL_H
#define NN_INFERENCE_INTERNAL_H
#include <stdbool.h>
#include <stddef.h>
#include <lua.h>
#include "inference/inference.h"
#include "catalog/catalog.h"
#include "model/model.h"
#include "project/project.h"
#define RULE_FILE_LIMIT (1024u * 1024u)
#define LUA_MEMORY_LIMIT (16u * 1024u * 1024u)
#define LUA_INSTRUCTION_LIMIT 200000u
#define INFERENCE_WORK_LIMIT 100000u
#define INFERENCE_LUA_INSTRUCTION_LIMIT 10000000u
#define INFERENCE_SCOPE_CACHE_ENTRY_LIMIT 256u
#define INFERENCE_SCOPE_CACHE_BYTE_LIMIT (16u * 1024u * 1024u)
#define LUA_TEMP_ALLOCATION_LIMIT 512
typedef struct Evaluation Evaluation;
typedef enum {
    NN_PARAMETER_RESOLUTION_SUCCESS,
    NN_PARAMETER_RESOLUTION_UNRESOLVED,
    NN_PARAMETER_RESOLUTION_SEMANTIC,
    NN_PARAMETER_RESOLUTION_COMPILATION,
    NN_PARAMETER_RESOLUTION_RUNTIME,
    NN_PARAMETER_RESOLUTION_ALLOCATION
} NNParameterResolutionStatus;
typedef struct LuaBudget {
    size_t used;
    unsigned instructions;
    Evaluation *evaluation;
    bool host_allocation_failed;
    size_t temporary_count;
    void *temporary[LUA_TEMP_ALLOCATION_LIMIT];
} LuaBudget;
void *nn_inference_limited_alloc(void *data, void *ptr, size_t old_size, size_t new_size);
void nn_inference_instruction_hook(lua_State *state, lua_Debug *debug);
void nn_inference_lua_temp_cleanup(LuaBudget *budget);
typedef struct { char *dtype; char **dimensions; size_t count; } Tensor;
typedef struct { char *handle_id; char *type; Tensor tensor; } OutputTensor;
typedef struct { OutputTensor *items; size_t count; } NodeOutputs;
typedef struct {
    NNInferenceResult view;
    char *id, *message, *dtype, *cause_node_id, *source_file;
    char **dimensions;
    NNInferenceTensor *outputs;
    OutputTensor *owned_outputs;
} Result;
struct NNInferenceReport {
    Result *items;
    size_t count;
    NNInferenceStatus root_status;
    char *root_message;
    bool failed;
};
typedef struct Evaluation {
    const NNProject *project;
    const NNModel *model;
    const NNCatalog *catalog;
    NNInferenceReport *report;
    const NNDataset *dataset;
    size_t work_units;
    size_t lua_instruction_units;
    struct ScopeCacheEntry *scope_cache[INFERENCE_SCOPE_CACHE_ENTRY_LIMIT];
    size_t scope_cache_count;
    size_t scope_cache_bytes;
    bool work_budget_exceeded;
    bool allocation_failed;
} Evaluation;
typedef struct { const NNEdge *edge; size_t order; } Incoming;
typedef struct {
    Evaluation *evaluation;
    const NNNode *node;
    size_t depth;
    const Tensor *inherited;
    char *cause_node_id;
    Tensor inherited_scratch;
    Tensor output_scratch[2];
    char *message_scratch;
    NNValue reference_scratch;
    NNParameter *reference_parameters;
    size_t reference_parameter_count;
    Tensor *reference_inputs;
    size_t reference_input_count;
} LuaContext;
typedef struct {
    LuaContext *context;
    Tensor *inputs;
    size_t input_count;
    const NNParameter *parameters;
    size_t parameter_count;
} RuleCall;

typedef struct {
    Tensor *outputs;
    size_t output_count;
    const NNPackage *package;
    bool terminal;
    char **message;
    NNInferenceStatus status;
} ExtractResult;
bool nn_inference_tensor_from_lua(lua_State *state, int index, Tensor *tensor, bool tracked);
void nn_inference_tensor_detach_lua_temporaries(lua_State *state, Tensor *tensor);
int nn_inference_extract_rule_result(lua_State *state);
int nn_inference_invoke_rule(lua_State *state);
bool nn_inference_protected_lua_initialize(lua_State *state);
NNParameterResolutionStatus nn_inference_resolve_parameters_typed(
    const NNProject *project, const char *node_id, NNParameter **parameters,
    size_t *count, char *error, size_t capacity);
extern char nn_inference_invoke_rule_registry_key;
extern char nn_inference_extract_result_registry_key;
NNInferenceStatus nn_inference_evaluate_scope(Evaluation *evaluation, const char *scope,
    const Tensor *inherited, size_t depth, Tensor *output, char **message,
    char **cause_node_id);
size_t nn_inference_find_node_index(const NNModel *model, const char *id);
Result *nn_inference_report_result(Evaluation *evaluation, const NNNode *node);
const char *nn_inference_package_kind(Evaluation *evaluation, const NNNode *node);
void nn_inference_scope_set_status(Evaluation *evaluation, const char *scope,
    NNInferenceStatus status, const char *message);
NNInferenceStatus nn_inference_execute_rule(Evaluation *evaluation, const NNNode *node,
    const NNPackage *package, size_t depth, const Tensor *inherited, Tensor *inputs,
    size_t input_count, Tensor *output, char **message, char **source_file,
    size_t *source_line, char **cause_node_id);
bool nn_inference_result_set(Result *result, const NNNode *node,
    NNInferenceStatus status, char *message, Tensor *tensor, const char *cause,
    const char *source_file, size_t source_line, const char *code_override);
bool nn_inference_result_set_output_metadata(Result *result, const NNPackage *package,
    const Tensor *tensors);
void nn_inference_result_dispose(Result *result);
bool nn_inference_result_copy(Result *destination, const Result *source);
bool nn_inference_charge_work(Evaluation *evaluation, size_t amount);
bool nn_inference_charge_lua(Evaluation *evaluation, size_t amount);
bool nn_inference_scope_cache_lookup(Evaluation *evaluation, const NNNode *owner,
    const Tensor *inherited, size_t depth, Tensor *outputs, size_t output_count,
    bool *hit);
void nn_inference_scope_cache_store(Evaluation *evaluation, const NNNode *owner,
    const Tensor *inherited, size_t depth, const Tensor *outputs, size_t output_count);
void nn_inference_scope_cache_free(Evaluation *evaluation);
void nn_inference_free_dimensions(char **dimensions, size_t count);
void nn_inference_tensor_dispose(Tensor *tensor);
bool nn_inference_tensor_copy(Tensor *destination, const Tensor *source);
void nn_inference_node_outputs_dispose(NodeOutputs *outputs);
Tensor *nn_inference_node_output_find(NodeOutputs *outputs, const char *handle_id);
bool nn_inference_node_output_add(NodeOutputs *outputs, const NNPackage *package, const Tensor *tensors);
bool nn_inference_node_output_add_mapping(NodeOutputs *outputs, const NNNode *node, const NNPackage *package, const Tensor *tensor);
bool nn_inference_read_shape(lua_State *state, int tensor, char ***dims, size_t *count, bool temporary);
void nn_inference_raw_getfield(lua_State *state, int index, const char *key);
int nn_inference_tensor_read(lua_State *state, int index);
void nn_inference_tensor_push(lua_State *state, const char *dtype, const char *const *dimensions, size_t count);
void *nn_inference_lua_temp_calloc(lua_State *state, size_t count, size_t size);
void *nn_inference_lua_temp_copy(lua_State *state, const char *text);
void nn_inference_lua_temp_free(lua_State *state, void *memory);
void nn_inference_lua_temp_detach(lua_State *state, void *memory);
void nn_inference_lua_temp_dimensions_free(lua_State *state, char **dimensions, size_t count);
void nn_inference_note_host_allocation_failure(lua_State *state);
void nn_inference_set_tensor_functions(lua_State *state);
bool nn_inference_resolve_reference(const NNProject *project, const NNNode *node,
                                    const NNParameterDef *definition,
                                    const NNValue *value, NNValue *resolved,
                                    char *error, size_t capacity);
#endif
