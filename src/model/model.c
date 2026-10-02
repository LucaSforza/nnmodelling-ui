#include "model_internal.h"
#include "model_utils.h"

#include <stdlib.h>

NNModel *nn_model_new(void) { return calloc(1, sizeof(NNModel)); }

void nn_model_free(NNModel *model) {
    if (!model) return;
    for (size_t i = 0; i < model->node_count; ++i) nn_model_node_dispose(&model->nodes[i].view);
    for (size_t i = 0; i < model->edge_count; ++i) nn_model_edge_dispose(&model->edges[i].view);
    free(model->nodes); free(model->edges); free(model);
}

size_t nn_model_node_count(const NNModel *model) { return model ? model->node_count : 0; }
size_t nn_model_edge_count(const NNModel *model) { return model ? model->edge_count : 0; }
const NNNode *nn_model_node_at(const NNModel *model, size_t index) {
    return model && index < model->node_count ? &model->nodes[index].view : NULL;
}
const NNEdge *nn_model_edge_at(const NNModel *model, size_t index) {
    return model && index < model->edge_count ? &model->edges[index].view : NULL;
}
const NNNode *nn_model_find_node(const NNModel *model, const char *id) {
    size_t i = nn_model_node_index(model, id);
    return i == (size_t)-1 ? NULL : &model->nodes[i].view;
}
