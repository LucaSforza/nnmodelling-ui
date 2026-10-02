#include "inference_internal.h"
#include "utils/utils.h"

#include <stdlib.h>
#include <string.h>

NNInferenceReport *nn_infer_project(const NNProject *project)
{
    if (!project) return NULL;
    const NNModel *model = nn_project_model((NNProject *)project);
    size_t count = nn_model_node_count(model);
    NNInferenceReport *report = calloc(1, sizeof(*report));
    if (!report) return NULL;
    report->items = calloc(count ? count : 1, sizeof(*report->items));
    if (!report->items) { free(report); return NULL; }
    report->count = count;
    Evaluation evaluation = { .model = model,
        .catalog = nn_project_catalog(project), .report = report,
        .dataset = nn_project_active_dataset(project) };
    Tensor ignored = {0}; char *message = NULL;
    (void)nn_inference_evaluate_scope(&evaluation, "", NULL, 0, &ignored, &message, NULL);
    free(message); nn_inference_tensor_dispose(&ignored);
    size_t root_inputs = 0, root_outputs = 0, root_losses = 0;
    bool invalid_root_mapping = false;
    for (size_t i = 0; i < count; ++i) {
        const NNNode *node = nn_model_node_at(model, i);
        if (node->scope_id && *node->scope_id) continue;
        if (node->boundary_handle_id) invalid_root_mapping = true;
        const char *kind = nn_inference_package_kind(&evaluation, node);
        if (kind && !strcmp(kind, "input")) ++root_inputs;
        else if (kind && !strcmp(kind, "output")) ++root_outputs;
        else if (kind && !strcmp(kind, "loss-output")) ++root_losses;
    }
    if (invalid_root_mapping || root_outputs > 1 || root_losses > 1) {
        report->root_status = NN_INFERENCE_SEMANTIC_ERROR;
        report->root_message = nn_text_copy(invalid_root_mapping
            ? "root terminals cannot have boundary mappings"
            : "root must contain exactly one Output and one Loss Output terminal");
    } else if (!root_inputs || root_outputs != 1 || root_losses != 1) {
        report->root_status = NN_INFERENCE_UNRESOLVED;
        report->root_message = nn_text_copy(!root_inputs
            ? "root is missing an Input boundary"
            : root_outputs != 1 ? "root is missing an Output boundary"
                                : "root is missing a Loss Output boundary");
    } else {
        report->root_status = NN_INFERENCE_SUCCESS;
    }
    if (report->root_status != NN_INFERENCE_SUCCESS && !report->root_message)
        evaluation.allocation_failed = true;
    for (size_t i = 0; i < count; ++i) {
        const NNNode *node = nn_model_node_at(model, i);
        if (report->items[i].id) continue;
        const char *scope = node->scope_id ? node->scope_id : "";
        Result *result = nn_inference_report_result(&evaluation, node);
        if (!*scope) {
            if (result && !nn_inference_result_set(result, node, NN_INFERENCE_UNRESOLVED,
                                    nn_text_copy("node could not be evaluated"), NULL,
                                    NULL, NULL, 0, NULL)) report->failed = true;
        } else {
            const NNNode *owner = nn_model_find_node(model, scope);
            const char *owner_kind = owner ? nn_inference_package_kind(&evaluation, owner) : NULL;
            if (owner_kind && !strcmp(owner_kind, "subflow")) {
                /* Valid children skipped because their owner never delegated. */
                for (size_t child = 0; child < count; ++child) {
                    const NNNode *nested = nn_model_node_at(model, child);
                    if (strcmp(nested->scope_id ? nested->scope_id : "", scope)) continue;
                    Result *unseen = nn_inference_report_result(&evaluation, nested);
                    if (unseen && !unseen->id &&
                        !nn_inference_result_set(unseen, nested, NN_INFERENCE_UNRESOLVED,
                                    nn_text_copy("subflow was not invoked by its owner"), NULL,
                                    NULL, NULL, 0, NULL)) report->failed = true;
                }
            } else if (result && !nn_inference_result_set(result, node, NN_INFERENCE_UNRESOLVED,
                                            nn_text_copy("orphan scope has no subflow owner"), NULL,
                                            NULL, NULL, 0, NULL)) report->failed = true;
        }
    }
    /* Normalize skipped nested scopes only after all owner outcomes exist. */
    for (size_t i = 0; i < count; ++i) {
        Result *child_result = &report->items[i];
        if (!child_result->message ||
            strcmp(child_result->message, "subflow was not invoked by its owner")) continue;
        const NNNode *child = nn_model_node_at(model, i);
        const char *owner_id = child->scope_id ? child->scope_id : "";
        const NNNode *owner = *owner_id ? nn_model_find_node(model, owner_id) : NULL;
        const char *cause = NULL;
        for (size_t depth = 0; owner && depth < 32; ++depth) {
            Result *owner_result = nn_inference_report_result(&evaluation, owner);
            if (!owner_result || owner_result->view.status == NN_INFERENCE_SUCCESS) break;
            if (owner_result->cause_node_id) {
                cause = owner_result->cause_node_id;
                break;
            }
            if (owner_result->message &&
                !strcmp(owner_result->message, "subflow was not invoked by its owner")) {
                const char *parent_id = owner->scope_id ? owner->scope_id : "";
                owner = *parent_id ? nn_model_find_node(model, parent_id) : NULL;
                continue;
            }
            cause = owner->id;
            break;
        }
        if (cause && !nn_inference_result_set(child_result, child, NN_INFERENCE_UNRESOLVED,
                                 nn_text_copy("subflow was blocked by its owner"), NULL,
                                 cause, NULL, 0, "model.blocked"))
            report->failed = true;
    }
    if (report->failed || evaluation.allocation_failed) {
        nn_inference_free(report);
        return NULL;
    }
    return report;
}

void nn_inference_free(NNInferenceReport *report)
{
    if (!report) return;
    for (size_t i = 0; i < report->count; ++i) {
        Result *item = &report->items[i];
        free(item->id); free(item->message); free(item->dtype);
        free(item->cause_node_id); free(item->source_file);
        for (size_t o = 0; item->owned_outputs && o < item->view.output_count; ++o) {
            free(item->owned_outputs[o].handle_id); free(item->owned_outputs[o].type);
            nn_inference_tensor_dispose(&item->owned_outputs[o].tensor);
        }
        free(item->outputs); free(item->owned_outputs);
        nn_inference_free_dimensions(item->dimensions, item->view.dimension_count);
    }
    free(report->root_message); free(report->items); free(report);
}

size_t nn_inference_count(const NNInferenceReport *report) { return report ? report->count : 0; }

const NNInferenceResult *nn_inference_at(const NNInferenceReport *report, size_t index)
{ return report && index < report->count ? &report->items[index].view : NULL; }

NNInferenceStatus nn_inference_root_status(const NNInferenceReport *report)
{ return report ? report->root_status : NN_INFERENCE_RUNTIME_FAULT; }

const char *nn_inference_root_message(const NNInferenceReport *report)
{ return report ? report->root_message : NULL; }