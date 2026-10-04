#include "inference_internal.h"
#include "catalog/catalog.h"
#include "utils/utils.h"

#include <stdlib.h>
#include <string.h>

void nn_inference_free_dimensions(char **dimensions, size_t count)
{
    if (!dimensions) return;
    for (size_t i = 0; i < count; ++i) free(dimensions[i]);
    free(dimensions);
}

void nn_inference_tensor_dispose(Tensor *tensor)
{
    free(tensor->dtype); nn_inference_free_dimensions(tensor->dimensions, tensor->count);
    memset(tensor, 0, sizeof(*tensor));
}

void nn_inference_node_outputs_dispose(NodeOutputs *outputs)
{
    if (!outputs) return;
    for (size_t i = 0; i < outputs->count; ++i) {
        free(outputs->items[i].handle_id);
        free(outputs->items[i].type);
        nn_inference_tensor_dispose(&outputs->items[i].tensor);
    }
    free(outputs->items);
    memset(outputs, 0, sizeof(*outputs));
}

Tensor *nn_inference_node_output_find(NodeOutputs *outputs, const char *handle_id)
{
    if (!outputs || !handle_id) return NULL;
    for (size_t i = 0; i < outputs->count; ++i)
        if (outputs->items[i].handle_id &&
            !strcmp(outputs->items[i].handle_id, handle_id))
            return &outputs->items[i].tensor;
    return NULL;
}

bool nn_inference_node_output_add(NodeOutputs *outputs, const NNPackage *package,
                            const Tensor *tensors)
{
    if (!package || !package->output_count || package->output_count > 2 ||
        !package->outputs || !tensors)
        return false;
    OutputTensor *items = calloc(package->output_count, sizeof(*items));
    if (!items) return false;
    for (size_t i = 0; i < package->output_count; ++i) {
        items[i].handle_id = nn_text_copy(package->outputs[i].id);
        items[i].type = nn_text_copy(package->outputs[i].type);
        if (!items[i].handle_id || !items[i].type ||
            !nn_inference_tensor_copy(&items[i].tensor, &tensors[i])) {
            NodeOutputs partial = { .items = items, .count = package->output_count };
            nn_inference_node_outputs_dispose(&partial);
            return false;
        }
    }
    outputs->items = items;
    outputs->count = package->output_count;
    return true;
}

bool nn_inference_node_output_add_mapping(NodeOutputs *outputs, const NNNode *node,
                                    const NNPackage *package, const Tensor *tensor)
{
    if (!node || !node->boundary_handle_id || !package || !tensor) return true;
    OutputTensor *item = calloc(1, sizeof(*item));
    if (!item) return false;
    item->handle_id = nn_text_copy(node->boundary_handle_id);
    item->type = nn_text_copy(nn_catalog_package_is_kind(package, "loss-output")
                                  ? "loss" : "output");
    if (!item->handle_id || !item->type || !nn_inference_tensor_copy(&item->tensor, tensor)) {
        free(item->handle_id); free(item->type); nn_inference_tensor_dispose(&item->tensor);
        free(item); return false;
    }
    outputs->items = item;
    outputs->count = 1;
    return true;
}

bool nn_inference_tensor_copy(Tensor *destination, const Tensor *source)
{
    destination->dtype = nn_text_copy(source->dtype);
    destination->dimensions = calloc(source->count ? source->count : 1,
                                     sizeof(*destination->dimensions));
    if (!destination->dtype || !destination->dimensions) { nn_inference_tensor_dispose(destination); return false; }
    destination->count = source->count;
    for (size_t i = 0; i < source->count; ++i) {
        destination->dimensions[i] = nn_text_copy(source->dimensions[i]);
        if (!destination->dimensions[i]) { nn_inference_tensor_dispose(destination); return false; }
    }
    return true;
}
