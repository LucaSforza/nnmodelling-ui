#include "model_internal.h"
#include "model_utils.h"
#include "utils/utils.h"

#include <stdlib.h>
#include <string.h>

NNModel *nn_model_new(void) { return calloc(1, sizeof(NNModel)); }

static bool copy_node(NNNode *destination, const NNNode *source)
{
    NNNode copy = {0};
    copy.id = nn_text_copy(source->id);
    copy.label = nn_text_copy(source->label);
    copy.package_id = nn_text_copy(source->package_id);
    copy.package_version = nn_text_copy(source->package_version);
    copy.scope_id = nn_text_copy(source->scope_id);
    copy.boundary_handle_id = source->boundary_handle_id
        ? nn_text_copy(source->boundary_handle_id) : NULL;
    copy.x = source->x;
    copy.y = source->y;
    if (!copy.id || !copy.label || !copy.package_id || !copy.package_version ||
        !copy.scope_id || (source->boundary_handle_id && !copy.boundary_handle_id) ||
        source->parameter_count > (size_t)-1 / sizeof(NNParameter) ||
        (source->parameter_count && !source->parameters)) {
        nn_model_node_dispose(&copy);
        return false;
    }
    NNParameter *parameters = source->parameter_count
        ? calloc(source->parameter_count, sizeof(*parameters)) : NULL;
    if (source->parameter_count && !parameters) {
        nn_model_node_dispose(&copy);
        return false;
    }
    copy.parameters = parameters;
    for (size_t i = 0; i < source->parameter_count; ++i) {
        parameters[i].key = nn_text_copy(source->parameters[i].key);
        if (!parameters[i].key || !nn_value_copy(&parameters[i].value, &source->parameters[i].value)) {
            copy.parameter_count = i + 1;
            nn_model_node_dispose(&copy);
            return false;
        }
        ++copy.parameter_count;
    }
    *destination = copy;
    return true;
}

static bool copy_edge(NNEdge *destination, const NNEdge *source)
{
    NNEdge copy = {0};
    copy.id = nn_text_copy(source->id);
    copy.source_id = nn_text_copy(source->source_id);
    copy.source_handle_id = nn_text_copy(source->source_handle_id);
    copy.target_id = nn_text_copy(source->target_id);
    copy.target_handle_id = nn_text_copy(source->target_handle_id);
    copy.scope_id = nn_text_copy(source->scope_id);
    if (!copy.id || !copy.source_id || !copy.source_handle_id || !copy.target_id ||
        !copy.target_handle_id || !copy.scope_id) {
        nn_model_edge_dispose(&copy);
        return false;
    }
    *destination = copy;
    return true;
}

NNModel *nn_model_copy(const NNModel *model)
{
    if (!model) return NULL;
    NNModel *copy = nn_model_new();
    if (!copy) return NULL;
    if (model->node_count > (size_t)-1 / sizeof(*copy->nodes) ||
        model->edge_count > (size_t)-1 / sizeof(*copy->edges)) {
        nn_model_free(copy);
        return NULL;
    }
    if (model->node_count) {
        copy->nodes = calloc(model->node_count, sizeof(*copy->nodes));
        if (!copy->nodes) { nn_model_free(copy); return NULL; }
        copy->node_capacity = model->node_count;
    }
    if (model->edge_count) {
        copy->edges = calloc(model->edge_count, sizeof(*copy->edges));
        if (!copy->edges) { nn_model_free(copy); return NULL; }
        copy->edge_capacity = model->edge_count;
    }
    for (size_t i = 0; i < model->node_count; ++i) {
        if (!copy_node(&copy->nodes[i].view, &model->nodes[i].view)) {
            nn_model_free(copy);
            return NULL;
        }
        ++copy->node_count;
    }
    for (size_t i = 0; i < model->edge_count; ++i) {
        if (!copy_edge(&copy->edges[i].view, &model->edges[i].view)) {
            nn_model_free(copy);
            return NULL;
        }
        ++copy->edge_count;
    }
    return copy;
}

static bool equal_text(const char *left, const char *right)
{
    return (!left || !right) ? left == right : !strcmp(left, right);
}

static bool equal_value(const NNValue *left, const NNValue *right)
{
    if (left->type != right->type) return false;
    switch (left->type) {
    case NN_VALUE_BOOL: return left->as.boolean == right->as.boolean;
    case NN_VALUE_INT: return left->as.integer == right->as.integer;
    case NN_VALUE_REAL: return left->as.real == right->as.real;
    case NN_VALUE_STRING: return equal_text(left->as.string, right->as.string);
    case NN_VALUE_ARRAY:
        if (left->as.array.count != right->as.array.count ||
            (left->as.array.count && (!left->as.array.items || !right->as.array.items))) return false;
        for (size_t i = 0; i < left->as.array.count; ++i)
            if (!equal_value(&left->as.array.items[i], &right->as.array.items[i])) return false;
        return true;
    case NN_VALUE_OBJECT:
        if (left->as.object.count != right->as.object.count ||
            (left->as.object.count && (!left->as.object.items || !right->as.object.items))) return false;
        for (size_t i = 0; i < left->as.object.count; ++i) {
            const NNParameter *item = &left->as.object.items[i];
            bool found = false;
            if (!item->key) return false;
            for (size_t j = 0; j < right->as.object.count; ++j) {
                const NNParameter *candidate = &right->as.object.items[j];
                if (equal_text(item->key, candidate->key)) {
                    if (!equal_value(&item->value, &candidate->value)) return false;
                    found = true;
                    break;
                }
            }
            if (!found) return false;
        }
        return true;
    default: return false;
    }
}

bool nn_model_equal(const NNModel *left, const NNModel *right)
{
    if (left == right) return true;
    if (!left || !right || left->node_count != right->node_count ||
        left->edge_count != right->edge_count) return false;
    for (size_t i = 0; i < left->node_count; ++i) {
        const NNNode *a = &left->nodes[i].view, *b = &right->nodes[i].view;
        if (!equal_text(a->id, b->id) || !equal_text(a->label, b->label) ||
            !equal_text(a->package_id, b->package_id) ||
            !equal_text(a->package_version, b->package_version) ||
            !equal_text(a->scope_id, b->scope_id) ||
            !equal_text(a->boundary_handle_id, b->boundary_handle_id) ||
            a->x != b->x || a->y != b->y || a->parameter_count != b->parameter_count ||
            (a->parameter_count && (!a->parameters || !b->parameters))) return false;
        for (size_t p = 0; p < a->parameter_count; ++p)
            if (!equal_text(a->parameters[p].key, b->parameters[p].key) ||
                !equal_value(&a->parameters[p].value, &b->parameters[p].value)) return false;
    }
    for (size_t i = 0; i < left->edge_count; ++i) {
        const NNEdge *a = &left->edges[i].view, *b = &right->edges[i].view;
        if (!equal_text(a->id, b->id) || !equal_text(a->source_id, b->source_id) ||
            !equal_text(a->source_handle_id, b->source_handle_id) ||
            !equal_text(a->target_id, b->target_id) ||
            !equal_text(a->target_handle_id, b->target_handle_id) ||
            !equal_text(a->scope_id, b->scope_id)) return false;
    }
    return true;
}

void nn_model_swap(NNModel *left, NNModel *right)
{
    if (!left || !right || left == right) return;
    NNModel temporary = *left;
    *left = *right;
    *right = temporary;
}

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
