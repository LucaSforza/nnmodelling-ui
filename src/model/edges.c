#include "model_internal.h"
#include "model_utils.h"
#include "utils/utils.h"

#include <stdlib.h>
#include <string.h>

void nn_model_edge_dispose(NNEdge *edge) {
    free((char *)edge->id); free((char *)edge->source_id);
    free((char *)edge->source_handle_id); free((char *)edge->target_id);
    free((char *)edge->target_handle_id); free((char *)edge->scope_id);
}

static bool path_exists(const NNModel *model, size_t from, size_t to, bool *seen) {
    if (from == to) return true;
    if (seen[from]) return false;
    seen[from] = true;
    const char *id = model->nodes[from].view.id;
    for (size_t e = 0; e < model->edge_count; ++e) {
        const NNEdge *edge = &model->edges[e].view;
        if (!strcmp(edge->source_id, id)) {
            size_t next = nn_model_node_index(model, edge->target_id);
            if (next != (size_t)-1 && path_exists(model, next, to, seen)) return true;
        }
    }
    return false;
}

bool nn_model_connect(NNModel *model, const char *id, const char *source_id, const char *source_handle_id,
                      const char *target_id, const char *target_handle_id, char *error, size_t cap) {
    if (!model || !nn_model_valid_id(id) || !nn_model_valid_id(source_handle_id) || !nn_model_valid_id(target_handle_id))
        return nn_fail(error, cap, "invalid edge fields");
    if (nn_model_edge_index(model, id) != (size_t)-1 || nn_model_node_index(model, id) != (size_t)-1)
        return nn_fail(error, cap, "duplicate stable ID");
    size_t s = nn_model_node_index(model, source_id), t = nn_model_node_index(model, target_id);
    if (s == (size_t)-1 || t == (size_t)-1) return nn_fail(error, cap, "edge endpoint not found");
    NNNode *source = &model->nodes[s].view, *target = &model->nodes[t].view;
    if (strcmp(source->scope_id, target->scope_id)) return nn_fail(error, cap, "edge endpoints have different scopes");
    for (size_t e = 0; e < model->edge_count; ++e) {
        const NNEdge *edge = &model->edges[e].view;
        if (!strcmp(edge->target_id, target_id) && !strcmp(edge->target_handle_id, target_handle_id))
            return nn_fail(error, cap, "target handle is occupied");
    }
    bool *seen = calloc(model->node_count, sizeof(bool));
    if (!seen) return nn_fail(error, cap, "out of memory");
    bool cycle = path_exists(model, t, s, seen); free(seen);
    if (cycle) return nn_fail(error, cap, "edge would create a cycle");
    NNEdge edge = {0};
    edge.id = nn_text_copy(id); edge.source_id = nn_text_copy(source_id);
    edge.source_handle_id = nn_text_copy(source_handle_id); edge.target_id = nn_text_copy(target_id);
    edge.target_handle_id = nn_text_copy(target_handle_id); edge.scope_id = nn_text_copy(source->scope_id);
    if (!edge.id || !edge.source_id || !edge.source_handle_id || !edge.target_id ||
        !edge.target_handle_id || !edge.scope_id ||
        !nn_model_reserve((void **)&model->edges, &model->edge_capacity, model->edge_count + 1, sizeof(*model->edges))) {
        nn_model_edge_dispose(&edge); return nn_fail(error, cap, "out of memory");
    }
    model->edges[model->edge_count++].view = edge;
    nn_error_set(error, cap, ""); return true;
}

bool nn_model_disconnect(NNModel *model, const char *id, char *error, size_t cap) {
    size_t i = nn_model_edge_index(model, id);
    if (i == (size_t)-1) return nn_fail(error, cap, "edge not found");
    nn_model_edge_dispose(&model->edges[i].view);
    memmove(&model->edges[i], &model->edges[i + 1], (model->edge_count - i - 1) * sizeof(*model->edges));
    --model->edge_count; nn_error_set(error, cap, ""); return true;
}
