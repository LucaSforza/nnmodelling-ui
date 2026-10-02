#include "inference_internal.h"
#include "utils/utils.h"

#include <stdlib.h>
#include <string.h>

bool nn_inference_result_set(Result *result, const NNNode *node, NNInferenceStatus status,
                       char *message, Tensor *tensor, const char *cause,
                       const char *source_file, size_t source_line,
                       const char *code_override)
{
    free(result->id); free(result->message); free(result->dtype);
    free(result->cause_node_id); free(result->source_file);
    for (size_t i = 0; i < result->view.output_count; ++i) {
        free(result->owned_outputs ? result->owned_outputs[i].handle_id : NULL);
        free(result->owned_outputs ? result->owned_outputs[i].type : NULL);
        if (result->owned_outputs) nn_inference_tensor_dispose(&result->owned_outputs[i].tensor);
    }
    free(result->outputs); free(result->owned_outputs);
    nn_inference_free_dimensions(result->dimensions, result->view.dimension_count);
    memset(result, 0, sizeof(*result));
    result->id = nn_text_copy(node->id); result->message = message;
    result->view.node_id = result->id; result->view.status = status;
    result->view.message = result->message;
    result->cause_node_id = nn_text_copy(cause);
    result->source_file = nn_text_copy(source_file);
    result->view.cause_node_id = result->cause_node_id;
    result->view.source_file = result->source_file;
    result->view.source_line = source_line;
    if (code_override) result->view.code = code_override;
    else if (status == NN_INFERENCE_COMPILATION_ERROR) result->view.code = "lua.compile";
    else if (status == NN_INFERENCE_SEMANTIC_ERROR) result->view.code = "model.semantic";
    else if (status == NN_INFERENCE_UNRESOLVED) result->view.code = "model.incomplete";
    else if (status == NN_INFERENCE_RUNTIME_FAULT) result->view.code = "analysis.internal";
    else result->view.code = NULL;
    if (status == NN_INFERENCE_SUCCESS && tensor) {
        result->dtype = nn_text_copy(tensor->dtype);
        result->dimensions = calloc(tensor->count ? tensor->count : 1, sizeof(*result->dimensions));
        for (size_t i = 0; i < tensor->count && result->dimensions; ++i)
            result->dimensions[i] = nn_text_copy(tensor->dimensions[i]);
        result->view.dtype = result->dtype;
        result->view.dimensions = (const char *const *)result->dimensions;
        result->view.dimension_count = tensor->count;
    }
    if (!result->id || (status != NN_INFERENCE_SUCCESS && (!message || !result->view.code)) ||
        (cause && !result->cause_node_id) || (source_file && !result->source_file) ||
        (status == NN_INFERENCE_COMPILATION_ERROR && !result->source_file)) return false;
    if (status == NN_INFERENCE_SUCCESS && tensor) {
        if (!result->dtype || !result->dimensions) return false;
        for (size_t i = 0; i < tensor->count; ++i)
            if (!result->dimensions[i]) return false;
    }
    return true;
}

bool nn_inference_result_set_output_metadata(Result *result, const NNPackage *package,
                                       const Tensor *tensors)
{
    if (!result || !package || !package->outputs || !package->output_count ||
        package->output_count > 2 || !result->dtype) return false;
    result->outputs = calloc(package->output_count, sizeof(*result->outputs));
    result->owned_outputs = calloc(package->output_count, sizeof(*result->owned_outputs));
    if (!result->outputs || !result->owned_outputs) return false;
    result->view.output_count = package->output_count;
    for (size_t i = 0; i < package->output_count; ++i) {
        OutputTensor *owned = &result->owned_outputs[i];
        owned->handle_id = nn_text_copy(package->outputs[i].id);
        owned->type = nn_text_copy(package->outputs[i].type);
        if (!owned->handle_id || !owned->type) return false;
        if (!tensors || !nn_inference_tensor_copy(&owned->tensor, &tensors[i])) return false;
        result->outputs[i] = (NNInferenceTensor){
            .handle_id = owned->handle_id, .type = owned->type,
            .dtype = owned->tensor.dtype,
            .dimensions = (const char *const *)owned->tensor.dimensions,
            .dimension_count = owned->tensor.count
        };
    }
    result->view.outputs = result->outputs;
    return true;
}
