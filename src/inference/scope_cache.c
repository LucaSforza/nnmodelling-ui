#include "inference_internal.h"
#include "utils/utils.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    size_t index;
    Result result;
} CachedResult;

struct ScopeCacheEntry {
    char *owner_id;
    Tensor input;
    size_t depth;
    Tensor outputs[2];
    size_t output_count;
    CachedResult *rows;
    size_t row_count;
};

bool nn_inference_charge_work(Evaluation *evaluation, size_t amount)
{
    if (!evaluation || evaluation->work_units > INFERENCE_WORK_LIMIT ||
        amount > INFERENCE_WORK_LIMIT - evaluation->work_units) {
        if (evaluation) evaluation->work_budget_exceeded = true;
        return false;
    }
    evaluation->work_units += amount;
    return true;
}

bool nn_inference_charge_lua(Evaluation *evaluation, size_t amount)
{
    if (!evaluation || evaluation->lua_instruction_units > INFERENCE_LUA_INSTRUCTION_LIMIT ||
        amount > INFERENCE_LUA_INSTRUCTION_LIMIT - evaluation->lua_instruction_units) {
        return false;
    }
    evaluation->lua_instruction_units += amount;
    return true;
}

static bool scope_is_descendant(const NNModel *model, const NNNode *node,
                                const char *owner_id)
{
    const char *scope = node->scope_id ? node->scope_id : "";
    for (size_t depth = 0; *scope && depth < 32; ++depth) {
        if (!strcmp(scope, owner_id)) return true;
        const NNNode *parent = nn_model_find_node(model, scope);
        if (!parent) return false;
        scope = parent->scope_id ? parent->scope_id : "";
    }
    return false;
}

static bool size_add(size_t *size, size_t amount)
{
    if (amount > SIZE_MAX - *size) return false;
    *size += amount;
    return true;
}

static bool size_array(size_t *size, size_t count, size_t item_size)
{
    return !item_size || count <= SIZE_MAX / item_size
        ? size_add(size, count * item_size) : false;
}

static bool size_text(size_t *size, const char *text)
{
    return !text || size_add(size, strlen(text) + 1);
}

static bool tensor_size(size_t *size, const Tensor *tensor)
{
    if (!size_text(size, tensor->dtype) ||
        !size_array(size, tensor->count ? tensor->count : 1,
                    sizeof(*tensor->dimensions))) return false;
    for (size_t i = 0; i < tensor->count; ++i)
        if (!size_text(size, tensor->dimensions[i])) return false;
    return true;
}

static bool result_size(size_t *size, const Result *result)
{
    if ((result->view.dimension_count && !result->dimensions) ||
        (result->view.output_count && (!result->outputs || !result->owned_outputs)))
        return false;
    if (!size_text(size, result->id) ||
        !size_text(size, result->message) || !size_text(size, result->dtype) ||
        !size_text(size, result->cause_node_id) || !size_text(size, result->source_file) ||
        !size_array(size, result->view.dimension_count ? result->view.dimension_count : 1,
                    sizeof(*result->dimensions)) ||
        !size_array(size, result->view.output_count, sizeof(*result->outputs)) ||
        !size_array(size, result->view.output_count, sizeof(*result->owned_outputs)))
        return false;
    for (size_t i = 0; i < result->view.dimension_count; ++i)
        if (!result->dimensions[i] || !size_text(size, result->dimensions[i])) return false;
    for (size_t i = 0; i < result->view.output_count; ++i) {
        const OutputTensor *output = &result->owned_outputs[i];
        if (!output->handle_id || !output->type || !output->tensor.dtype ||
            (output->tensor.count && !output->tensor.dimensions) ||
            !size_text(size, output->handle_id) || !size_text(size, output->type) ||
            !tensor_size(size, &output->tensor)) return false;
        for (size_t d = 0; d < output->tensor.count; ++d)
            if (!output->tensor.dimensions[d]) return false;
    }
    return true;
}

static bool tensor_equal(const Tensor *left, const Tensor *right)
{
    if (!left->dtype || !right->dtype || strcmp(left->dtype, right->dtype) ||
        left->count != right->count) return false;
    for (size_t i = 0; i < left->count; ++i)
        if (strcmp(left->dimensions[i], right->dimensions[i])) return false;
    return true;
}

static void entry_free(struct ScopeCacheEntry *entry)
{
    if (!entry) return;
    free(entry->owner_id);
    nn_inference_tensor_dispose(&entry->input);
    for (size_t i = 0; i < entry->output_count; ++i)
        nn_inference_tensor_dispose(&entry->outputs[i]);
    for (size_t i = 0; entry->rows && i < entry->row_count; ++i)
        nn_inference_result_dispose(&entry->rows[i].result);
    free(entry->rows);
    free(entry);
}

bool nn_inference_scope_cache_lookup(Evaluation *evaluation, const NNNode *owner,
    const Tensor *inherited, size_t depth, Tensor *outputs, size_t output_count,
    bool *hit)
{
    *hit = false;
    if (evaluation->allocation_failed || evaluation->report->failed) return false;
    for (size_t i = 0; i < evaluation->scope_cache_count; ++i) {
        struct ScopeCacheEntry *entry = evaluation->scope_cache[i];
        if (entry->depth != depth || entry->output_count != output_count ||
            strcmp(entry->owner_id, owner->id) || !tensor_equal(&entry->input, inherited))
            continue;
        if (!nn_inference_charge_work(evaluation, entry->row_count)) return false;
        *hit = true;
        for (size_t r = 0; r < entry->row_count; ++r) {
            Result *destination = &evaluation->report->items[entry->rows[r].index];
            if (!nn_inference_result_copy(destination, &entry->rows[r].result)) {
                evaluation->allocation_failed = true;
                return false;
            }
        }
        for (size_t o = 0; o < output_count; ++o) {
            if (!nn_inference_tensor_copy(&outputs[o], &entry->outputs[o])) {
                evaluation->allocation_failed = true;
                return false;
            }
        }
        return true;
    }
    return true;
}

void nn_inference_scope_cache_store(Evaluation *evaluation, const NNNode *owner,
    const Tensor *inherited, size_t depth, const Tensor *outputs, size_t output_count)
{
    if (evaluation->allocation_failed || evaluation->report->failed || evaluation->scope_cache_count >=
            INFERENCE_SCOPE_CACHE_ENTRY_LIMIT || !owner || !inherited ||
        !output_count || output_count > 2) return;

    size_t row_count = 0, bytes = sizeof(struct ScopeCacheEntry);
    if (!size_text(&bytes, owner->id) || !tensor_size(&bytes, inherited)) return;
    for (size_t i = 0; i < nn_model_node_count(evaluation->model); ++i) {
        const NNNode *node = nn_model_node_at(evaluation->model, i);
        if (!scope_is_descendant(evaluation->model, node, owner->id)) continue;
        const Result *result = &evaluation->report->items[i];
        if (!result->id || result->view.status != NN_INFERENCE_SUCCESS) return;
        if (!size_add(&bytes, sizeof(CachedResult)) || !result_size(&bytes, result)) return;
        ++row_count;
    }
    for (size_t i = 0; i < output_count; ++i)
        if (!tensor_size(&bytes, &outputs[i])) return;
    if (!row_count || bytes > INFERENCE_SCOPE_CACHE_BYTE_LIMIT - evaluation->scope_cache_bytes)
        return;

    struct ScopeCacheEntry *entry = calloc(1, sizeof(*entry));
    if (!entry) { evaluation->allocation_failed = true; return; }
    entry->owner_id = nn_text_copy(owner->id);
    entry->depth = depth;
    entry->output_count = output_count;
    entry->row_count = row_count;
    entry->rows = calloc(row_count, sizeof(*entry->rows));
    if (!entry->owner_id || !entry->rows ||
        !nn_inference_tensor_copy(&entry->input, inherited)) {
        entry_free(entry); evaluation->allocation_failed = true; return;
    }
    for (size_t i = 0; i < output_count; ++i)
        if (!nn_inference_tensor_copy(&entry->outputs[i], &outputs[i])) {
            entry_free(entry); evaluation->allocation_failed = true; return;
        }
    size_t row = 0;
    for (size_t i = 0; i < nn_model_node_count(evaluation->model); ++i) {
        const NNNode *node = nn_model_node_at(evaluation->model, i);
        if (!scope_is_descendant(evaluation->model, node, owner->id)) continue;
        entry->rows[row].index = i;
        if (!nn_inference_result_copy(&entry->rows[row].result,
                                      &evaluation->report->items[i])) {
            entry_free(entry); evaluation->allocation_failed = true; return;
        }
        ++row;
    }
    evaluation->scope_cache[evaluation->scope_cache_count++] = entry;
    evaluation->scope_cache_bytes += bytes;
}

void nn_inference_scope_cache_free(Evaluation *evaluation)
{
    if (!evaluation) return;
    for (size_t i = 0; i < evaluation->scope_cache_count; ++i)
        entry_free(evaluation->scope_cache[i]);
    evaluation->scope_cache_count = 0;
    evaluation->scope_cache_bytes = 0;
}
