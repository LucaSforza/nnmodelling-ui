#include "model.h"
#include "utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { NNNode view; } NodeRecord;
typedef struct { NNEdge view; } EdgeRecord;

struct NNModel {
    NodeRecord *nodes;
    size_t node_count, node_capacity;
    EdgeRecord *edges;
    size_t edge_count, edge_capacity;
};

static bool reserve(void **items, size_t *capacity, size_t count, size_t item_size) {
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

void nn_value_dispose(NNValue *value) {
    if (!value) return;
    if (value->type == NN_VALUE_STRING) free(value->as.string);
    else if (value->type == NN_VALUE_ARRAY) {
        for (size_t i = 0; value->as.array.items && i < value->as.array.count; ++i)
            nn_value_dispose(&value->as.array.items[i]);
        free(value->as.array.items);
    }
    memset(value, 0, sizeof(*value));
}

bool nn_value_copy(NNValue *destination, const NNValue *source) {
    if (!destination || !source) return false;
    NNValue copy = { .type = source->type };
    switch (source->type) {
    case NN_VALUE_BOOL: copy.as.boolean = source->as.boolean; break;
    case NN_VALUE_INT: copy.as.integer = source->as.integer; break;
    case NN_VALUE_REAL: copy.as.real = source->as.real; break;
    case NN_VALUE_STRING:
        if (!source->as.string || !(copy.as.string = nn_text_copy(source->as.string))) return false;
        break;
    case NN_VALUE_ARRAY:
        if (source->as.array.count && !source->as.array.items) return false;
        if (source->as.array.count > (size_t)-1 / sizeof(NNValue)) return false;
        if (source->as.array.count) {
            copy.as.array.items = calloc(source->as.array.count, sizeof(NNValue));
            if (!copy.as.array.items) return false;
            for (size_t i = 0; i < source->as.array.count; ++i) {
                if (!nn_value_copy(&copy.as.array.items[i], &source->as.array.items[i])) {
                    copy.as.array.count = i;
                    nn_value_dispose(&copy);
                    return false;
                }
            }
            copy.as.array.count = source->as.array.count;
        }
        break;
    default: return false;
    }
    *destination = copy;
    return true;
}

static void node_dispose(NNNode *node) {
    free((char *)node->id); free((char *)node->label);
    free((char *)node->package_id); free((char *)node->package_version);
    free((char *)node->scope_id);
    for (size_t i = 0; i < node->parameter_count; ++i) {
        free(node->parameters[i].key);
        nn_value_dispose(&((NNParameter *)node->parameters)[i].value);
    }
    free((void *)node->parameters);
}

static void edge_dispose(NNEdge *edge) {
    free((char *)edge->id); free((char *)edge->source_id);
    free((char *)edge->source_handle_id); free((char *)edge->target_id);
    free((char *)edge->target_handle_id); free((char *)edge->scope_id);
}

NNModel *nn_model_new(void) { return calloc(1, sizeof(NNModel)); }

void nn_model_free(NNModel *model) {
    if (!model) return;
    for (size_t i = 0; i < model->node_count; ++i) node_dispose(&model->nodes[i].view);
    for (size_t i = 0; i < model->edge_count; ++i) edge_dispose(&model->edges[i].view);
    free(model->nodes); free(model->edges); free(model);
}

static size_t node_index(const NNModel *model, const char *id) {
    if (!model || !id) return (size_t)-1;
    for (size_t i = 0; i < model->node_count; ++i)
        if (!strcmp(model->nodes[i].view.id, id)) return i;
    return (size_t)-1;
}

static size_t edge_index(const NNModel *model, const char *id) {
    if (!model || !id) return (size_t)-1;
    for (size_t i = 0; i < model->edge_count; ++i)
        if (!strcmp(model->edges[i].view.id, id)) return i;
    return (size_t)-1;
}

static bool valid_id(const char *s) { return s && *s; }

bool nn_model_add_node(NNModel *model, const char *id, const char *label,
                       const char *package_id, const char *package_version,
                       const char *scope_id, double x, double y,
                       char *error, size_t error_capacity) {
    if (!model || !valid_id(id) || !label || !valid_id(package_id) ||
        !valid_id(package_version) || !scope_id)
        return nn_fail(error, error_capacity, "invalid node fields");
    if (node_index(model, id) != (size_t)-1 || edge_index(model, id) != (size_t)-1)
        return nn_fail(error, error_capacity, "duplicate stable ID");
    NNNode node = {0};
    node.id = nn_text_copy(id); node.label = nn_text_copy(label);
    node.package_id = nn_text_copy(package_id); node.package_version = nn_text_copy(package_version);
    node.scope_id = nn_text_copy(scope_id); node.x = x; node.y = y;
    if (!node.id || !node.label || !node.package_id || !node.package_version || !node.scope_id ||
        !reserve((void **)&model->nodes, &model->node_capacity, model->node_count + 1, sizeof(*model->nodes))) {
        node_dispose(&node);
        return nn_fail(error, error_capacity, "out of memory");
    }
    model->nodes[model->node_count++].view = node;
    nn_error_set(error, error_capacity, "");
    return true;
}

bool nn_model_remove_node(NNModel *model, const char *id, char *error, size_t cap) {
    size_t i = node_index(model, id);
    if (i == (size_t)-1) return nn_fail(error, cap, "node not found");
    /* Remove edges while id is still valid; callers may pass node->id itself. */
    for (size_t e = 0; e < model->edge_count;) {
        NNEdge *edge = &model->edges[e].view;
        if (!strcmp(edge->source_id, id) || !strcmp(edge->target_id, id)) {
            edge_dispose(edge);
            memmove(&model->edges[e], &model->edges[e + 1], (model->edge_count - e - 1) * sizeof(*model->edges));
            --model->edge_count;
        } else ++e;
    }
    node_dispose(&model->nodes[i].view);
    memmove(&model->nodes[i], &model->nodes[i + 1], (model->node_count - i - 1) * sizeof(*model->nodes));
    --model->node_count;
    nn_error_set(error, cap, ""); return true;
}

bool nn_model_move_node(NNModel *model, const char *id, double x, double y, char *error, size_t cap) {
    size_t i = node_index(model, id);
    if (i == (size_t)-1) return nn_fail(error, cap, "node not found");
    model->nodes[i].view.x = x; model->nodes[i].view.y = y;
    nn_error_set(error, cap, ""); return true;
}

bool nn_model_rename_node(NNModel *model, const char *id, const char *label, char *error, size_t cap) {
    size_t i = node_index(model, id);
    if (i == (size_t)-1) return nn_fail(error, cap, "node not found");
    if (!label) return nn_fail(error, cap, "invalid label");
    char *copy = nn_text_copy(label);
    if (!copy) return nn_fail(error, cap, "out of memory");
    free((char *)model->nodes[i].view.label); model->nodes[i].view.label = copy;
    nn_error_set(error, cap, ""); return true;
}

static bool path_exists(const NNModel *model, size_t from, size_t to, bool *seen) {
    if (from == to) return true;
    if (seen[from]) return false;
    seen[from] = true;
    const char *id = model->nodes[from].view.id;
    for (size_t e = 0; e < model->edge_count; ++e) {
        const NNEdge *edge = &model->edges[e].view;
        if (!strcmp(edge->source_id, id)) {
            size_t next = node_index(model, edge->target_id);
            if (next != (size_t)-1 && path_exists(model, next, to, seen)) return true;
        }
    }
    return false;
}

bool nn_model_connect(NNModel *model, const char *id, const char *source_id, const char *source_handle_id,
                      const char *target_id, const char *target_handle_id, char *error, size_t cap) {
    if (!model || !valid_id(id) || !valid_id(source_handle_id) || !valid_id(target_handle_id))
        return nn_fail(error, cap, "invalid edge fields");
    if (edge_index(model, id) != (size_t)-1 || node_index(model, id) != (size_t)-1)
        return nn_fail(error, cap, "duplicate stable ID");
    size_t s = node_index(model, source_id), t = node_index(model, target_id);
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
        !reserve((void **)&model->edges, &model->edge_capacity, model->edge_count + 1, sizeof(*model->edges))) {
        edge_dispose(&edge); return nn_fail(error, cap, "out of memory");
    }
    model->edges[model->edge_count++].view = edge;
    nn_error_set(error, cap, ""); return true;
}

bool nn_model_disconnect(NNModel *model, const char *id, char *error, size_t cap) {
    size_t i = edge_index(model, id);
    if (i == (size_t)-1) return nn_fail(error, cap, "edge not found");
    edge_dispose(&model->edges[i].view);
    memmove(&model->edges[i], &model->edges[i + 1], (model->edge_count - i - 1) * sizeof(*model->edges));
    --model->edge_count; nn_error_set(error, cap, ""); return true;
}

bool nn_model_set_parameter(NNModel *model, const char *node_id, const char *key,
                            const NNValue *value, char *error, size_t cap) {
    size_t i = node_index(model, node_id);
    if (i == (size_t)-1) return nn_fail(error, cap, "node not found");
    if (!valid_id(key) || !value) return nn_fail(error, cap, "invalid parameter");
    NNValue copy = {0};
    if (!nn_value_copy(&copy, value)) return nn_fail(error, cap, "invalid value or out of memory");
    NNNode *node = &model->nodes[i].view;
    for (size_t p = 0; p < node->parameter_count; ++p) {
        if (!strcmp(node->parameters[p].key, key)) {
            nn_value_dispose(&((NNParameter *)node->parameters)[p].value);
            ((NNParameter *)node->parameters)[p].value = copy;
            nn_error_set(error, cap, ""); return true;
        }
    }
    char *key_copy = nn_text_copy(key);
    if (!key_copy || node->parameter_count == (size_t)-1 / sizeof(NNParameter)) {
        free(key_copy); nn_value_dispose(&copy); return nn_fail(error, cap, "out of memory");
    }
    NNParameter *grown = realloc((void *)node->parameters, (node->parameter_count + 1) * sizeof(NNParameter));
    if (!grown) { free(key_copy); nn_value_dispose(&copy); return nn_fail(error, cap, "out of memory"); }
    grown[node->parameter_count++] = (NNParameter){ .key = key_copy, .value = copy };
    node->parameters = grown;
    nn_error_set(error, cap, ""); return true;
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
    size_t i = node_index(model, id);
    return i == (size_t)-1 ? NULL : &model->nodes[i].view;
}
