#include "inference_internal.h"
#include "utils/utils.h"

#include <stdlib.h>
#include <string.h>

size_t nn_inference_find_node_index(const NNModel *model, const char *id)
{
    for (size_t i = 0; i < nn_model_node_count(model); ++i)
        if (!strcmp(nn_model_node_at(model, i)->id, id)) return i;
    return (size_t)-1;
}

Result *nn_inference_report_result(Evaluation *evaluation, const NNNode *node)
{
    size_t index = nn_inference_find_node_index(evaluation->model, node->id);
    if (index == (size_t)-1) return NULL;
    return &evaluation->report->items[index];
}

const char *nn_inference_package_kind(Evaluation *evaluation, const NNNode *node)
{
    const NNPackage *package = nn_catalog_find(evaluation->catalog,
                                               node->package_id, node->package_version);
    return package ? package->kind : NULL;
}

void nn_inference_scope_set_status(Evaluation *evaluation, const char *scope,
                             NNInferenceStatus status, const char *message)
{
    for (size_t i = 0; i < nn_model_node_count(evaluation->model); ++i) {
        const NNNode *node = nn_model_node_at(evaluation->model, i);
        if (strcmp(node->scope_id ? node->scope_id : "", scope ? scope : "")) continue;
        Result *result = nn_inference_report_result(evaluation, node);
        if (result && !nn_inference_result_set(result, node, status, nn_text_copy(message), NULL,
                                  NULL, NULL, 0, NULL)) evaluation->report->failed = true;
    }
}
