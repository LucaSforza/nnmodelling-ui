#include "model_internal.h"
#include "model_utils.h"

#include <stdlib.h>
#include <string.h>

bool nn_model_reserve(void **items, size_t *capacity, size_t count, size_t item_size) {
    if (count <= *capacity) return true;
    size_t next = *capacity ? *capacity : 4;
    while (next < count) {
        if (next > (size_t)-1 / 2) { next = count; break; }
        next *= 2;
    }
    if (next > (size_t)-1 / item_size) return false;
    void *grown = realloc(*items, next * item_size);
    if (!grown) return false;
    *items = grown;
    *capacity = next;
    return true;
}

size_t nn_model_node_index(const NNModel *model, const char *id) {
    if (!model || !id) return (size_t)-1;
    for (size_t i = 0; i < model->node_count; ++i)
        if (!strcmp(model->nodes[i].view.id, id)) return i;
    return (size_t)-1;
}

size_t nn_model_edge_index(const NNModel *model, const char *id) {
    if (!model || !id) return (size_t)-1;
    for (size_t i = 0; i < model->edge_count; ++i)
        if (!strcmp(model->edges[i].view.id, id)) return i;
    return (size_t)-1;
}

bool nn_model_valid_id(const char *id) { return id && *id; }
