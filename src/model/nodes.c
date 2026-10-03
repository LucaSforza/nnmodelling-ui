#include "model_internal.h"
#include "model_utils.h"
#include "utils/utils.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static bool normalize_position(double x, double y, int32_t *normalized_x,
                               int32_t *normalized_y, char *error, size_t cap) {
    if (!isfinite(x) || !isfinite(y))
        return nn_fail(error, cap, "node position must be finite");

    const double min_grid = (double)(INT32_MIN / NN_MODEL_GRID_SPACING);
    const double max_grid = (double)(INT32_MAX / NN_MODEL_GRID_SPACING);
    double grid_x = round(x / NN_MODEL_GRID_SPACING);
    double grid_y = round(y / NN_MODEL_GRID_SPACING);
    if (!isfinite(grid_x) || !isfinite(grid_y) ||
        grid_x < min_grid || grid_x > max_grid ||
        grid_y < min_grid || grid_y > max_grid)
        return nn_fail(error, cap, "node position is outside the supported range");

    *normalized_x = (int32_t)grid_x * NN_MODEL_GRID_SPACING;
    *normalized_y = (int32_t)grid_y * NN_MODEL_GRID_SPACING;
    return true;
}

void nn_model_node_dispose(NNNode *node) {
    free((char *)node->id); free((char *)node->label);
    free((char *)node->package_id); free((char *)node->package_version);
    free((char *)node->scope_id);
    free((char *)node->boundary_handle_id);
    for (size_t i = 0; i < node->parameter_count; ++i) {
        free(node->parameters[i].key);
        nn_value_dispose(&((NNParameter *)node->parameters)[i].value);
    }
    free((void *)node->parameters);
}

bool nn_model_add_node(NNModel *model, const char *id, const char *label,
                       const char *package_id, const char *package_version,
                       const char *scope_id, double x, double y,
                       char *error, size_t error_capacity) {
    int32_t normalized_x, normalized_y;
    if (!normalize_position(x, y, &normalized_x, &normalized_y, error, error_capacity))
        return false;
    if (!model || !nn_model_valid_id(id) || !label || !nn_model_valid_id(package_id) ||
        !nn_model_valid_id(package_version) || !scope_id)
        return nn_fail(error, error_capacity, "invalid node fields");
    if (nn_model_node_index(model, id) != (size_t)-1 || nn_model_edge_index(model, id) != (size_t)-1)
        return nn_fail(error, error_capacity, "duplicate stable ID");
    NNNode node = {0};
    node.id = nn_text_copy(id); node.label = nn_text_copy(label);
    node.package_id = nn_text_copy(package_id); node.package_version = nn_text_copy(package_version);
    node.scope_id = nn_text_copy(scope_id);
    node.x = normalized_x; node.y = normalized_y;
    if (!node.id || !node.label || !node.package_id || !node.package_version || !node.scope_id ||
        !nn_model_reserve((void **)&model->nodes, &model->node_capacity, model->node_count + 1, sizeof(*model->nodes))) {
        nn_model_node_dispose(&node);
        return nn_fail(error, error_capacity, "out of memory");
    }
    model->nodes[model->node_count++].view = node;
    nn_error_set(error, error_capacity, "");
    return true;
}

bool nn_model_remove_node(NNModel *model, const char *id, char *error, size_t cap) {
    size_t i = nn_model_node_index(model, id);
    if (i == (size_t)-1) return nn_fail(error, cap, "node not found");
    /* Remove edges while id is still valid; callers may pass node->id itself. */
    for (size_t e = 0; e < model->edge_count;) {
        NNEdge *edge = &model->edges[e].view;
        if (!strcmp(edge->source_id, id) || !strcmp(edge->target_id, id)) {
            nn_model_edge_dispose(edge);
            memmove(&model->edges[e], &model->edges[e + 1], (model->edge_count - e - 1) * sizeof(*model->edges));
            --model->edge_count;
        } else ++e;
    }
    nn_model_node_dispose(&model->nodes[i].view);
    memmove(&model->nodes[i], &model->nodes[i + 1], (model->node_count - i - 1) * sizeof(*model->nodes));
    --model->node_count;
    nn_error_set(error, cap, ""); return true;
}

bool nn_model_move_node(NNModel *model, const char *id, double x, double y, char *error, size_t cap) {
    int32_t normalized_x, normalized_y;
    if (!normalize_position(x, y, &normalized_x, &normalized_y, error, cap)) return false;
    size_t i = nn_model_node_index(model, id);
    if (i == (size_t)-1) return nn_fail(error, cap, "node not found");
    model->nodes[i].view.x = normalized_x; model->nodes[i].view.y = normalized_y;
    nn_error_set(error, cap, ""); return true;
}

bool nn_model_rename_node(NNModel *model, const char *id, const char *label, char *error, size_t cap) {
    size_t i = nn_model_node_index(model, id);
    if (i == (size_t)-1) return nn_fail(error, cap, "node not found");
    if (!label) return nn_fail(error, cap, "invalid label");
    char *copy = nn_text_copy(label);
    if (!copy) return nn_fail(error, cap, "out of memory");
    free((char *)model->nodes[i].view.label); model->nodes[i].view.label = copy;
    nn_error_set(error, cap, ""); return true;
}

bool nn_model_set_boundary_handle(NNModel *model, const char *node_id,
                                  const char *handle_id, char *error, size_t cap) {
    size_t i = nn_model_node_index(model, node_id);
    if (i == (size_t)-1) return nn_fail(error, cap, "node not found");
    if (!handle_id || !*handle_id) return nn_fail(error, cap, "invalid boundary handle");
    char *copy = nn_text_copy(handle_id);
    if (!copy) return nn_fail(error, cap, "out of memory");
    free((char *)model->nodes[i].view.boundary_handle_id);
    model->nodes[i].view.boundary_handle_id = copy;
    nn_error_set(error, cap, "");
    return true;
}
