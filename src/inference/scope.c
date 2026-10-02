#define _POSIX_C_SOURCE 200809L
#include "inference_internal.h"
#include "utils/utils.h"

#include <stdlib.h>
#include <string.h>

static int incoming_compare(const void *left, const void *right)
{
    const Incoming *a = left, *b = right;
    return a->order < b->order ? -1 : a->order > b->order ? 1 : 0;
}
NNInferenceStatus nn_inference_evaluate_scope(Evaluation *evaluation, const char *scope,
                                         const Tensor *inherited, size_t depth,
                                         Tensor *scope_output, char **scope_message,
                                         char **scope_cause)
{
    const NNModel *model = evaluation->model;
    size_t count = nn_model_node_count(model);
    NodeOutputs *outputs = calloc(count ? count : 1, sizeof(*outputs));
    size_t *indegree = calloc(count ? count : 1, sizeof(*indegree));
    bool *done = calloc(count ? count : 1, sizeof(*done));
    if (!outputs || !indegree || !done) {
        evaluation->allocation_failed = true;
        free(outputs); free(indegree); free(done);
        *scope_message = nn_text_copy("unable to allocate scope inference state");
        return NN_INFERENCE_RUNTIME_FAULT;
    }
    size_t scope_nodes = 0, inputs_n = 0, outputs_n = 0, losses_n = 0;
    size_t input_index = count, output_index = count;
    for (size_t i = 0; i < count; ++i) {
        const NNNode *node = nn_model_node_at(model, i);
        if (strcmp(node->scope_id ? node->scope_id : "", scope ? scope : "")) continue;
        ++scope_nodes;
        const char *kind = nn_inference_package_kind(evaluation, node);
        if (kind && !strcmp(kind, "input")) { ++inputs_n; input_index = i; }
        if (kind && !strcmp(kind, "output")) { ++outputs_n; output_index = i; }
        if (kind && !strcmp(kind, "loss-output")) ++losses_n;
    }
    bool nested_scope = scope && *scope;
    const NNNode *owner = nested_scope ? nn_model_find_node(model, scope) : NULL;
    const NNPackage *owner_package = owner ? nn_catalog_find(evaluation->catalog,
        owner->package_id, owner->package_version) : NULL;
    size_t mapped_index[2] = { count, count };
    bool invalid_mapping = false;
    if (nested_scope) {
        if (!owner_package || !owner_package->outputs || !owner_package->output_count ||
            owner_package->output_count > 2) invalid_mapping = true;
        for (size_t i = 0; i < count && !invalid_mapping; ++i) {
            const NNNode *terminal = nn_model_node_at(model, i);
            const char *kind = nn_inference_package_kind(evaluation, terminal);
            if (strcmp(terminal->scope_id ? terminal->scope_id : "", scope)) continue;
            if (terminal->boundary_handle_id &&
                (!kind || (strcmp(kind, "output") && strcmp(kind, "loss-output")))) {
                invalid_mapping = true;
                break;
            }
            if (!kind || (strcmp(kind, "output") && strcmp(kind, "loss-output"))) continue;
            if (!terminal->boundary_handle_id) continue;
            size_t handle = owner_package->output_count;
            for (size_t h = 0; h < owner_package->output_count; ++h)
                if (!strcmp(terminal->boundary_handle_id, owner_package->outputs[h].id)) {
                    handle = h; break;
                }
            if (handle == owner_package->output_count) { invalid_mapping = true; break; }
            const char *expected_kind = !strcmp(owner_package->outputs[handle].type, "loss")
                ? "loss-output" : "output";
            if (strcmp(kind, expected_kind) || mapped_index[handle] != count) {
                invalid_mapping = true; break;
            }
            mapped_index[handle] = i;
        }
        output_index = owner_package && owner_package->output_count ? mapped_index[0] : count;
    }
    if (nested_scope && invalid_mapping) {
        *scope_message = nn_text_copy("subflow terminal mapping is duplicate, unknown or wrong type");
        nn_inference_scope_set_status(evaluation, scope, NN_INFERENCE_SEMANTIC_ERROR, *scope_message);
        free(outputs); free(indegree); free(done); return NN_INFERENCE_SEMANTIC_ERROR;
    }
    if (!scope_nodes || (nested_scope && (!inputs_n || !(outputs_n + losses_n)))) {
        *scope_message = nn_text_copy("subflow scope is empty or missing a boundary");
        nn_inference_scope_set_status(evaluation, scope, NN_INFERENCE_UNRESOLVED, *scope_message);
        free(outputs); free(indegree); free(done); return NN_INFERENCE_UNRESOLVED;
    }
    if (nested_scope && inputs_n != 1) {
        *scope_message = nn_text_copy("subflow scope must contain exactly one immediate Input");
        nn_inference_scope_set_status(evaluation, scope, NN_INFERENCE_SEMANTIC_ERROR, *scope_message);
        free(outputs); free(indegree); free(done); return NN_INFERENCE_SEMANTIC_ERROR;
    }
    if (nested_scope && (outputs_n + losses_n > owner_package->output_count)) {
        *scope_message = nn_text_copy("subflow has extra terminal boundaries");
        nn_inference_scope_set_status(evaluation, scope, NN_INFERENCE_SEMANTIC_ERROR, *scope_message);
        free(outputs); free(indegree); free(done); return NN_INFERENCE_SEMANTIC_ERROR;
    }
    bool missing_mapping = false;
    if (nested_scope) for (size_t h = 0; h < owner_package->output_count; ++h)
        if (mapped_index[h] == count) missing_mapping = true;
    if (nested_scope && missing_mapping) {
        *scope_message = nn_text_copy("subflow output boundary is missing or unmapped");
        nn_inference_scope_set_status(evaluation, scope, NN_INFERENCE_UNRESOLVED, *scope_message);
        free(outputs); free(indegree); free(done); return NN_INFERENCE_UNRESOLVED;
    }
    for (size_t e = 0; e < nn_model_edge_count(model); ++e) {
        const NNEdge *edge = nn_model_edge_at(model, e);
        const NNNode *source = nn_model_find_node(model, edge->source_id);
        const NNNode *target = nn_model_find_node(model, edge->target_id);
        bool edge_claims_scope = !strcmp(edge->scope_id ? edge->scope_id : "", scope ? scope : "");
        bool touches_scope = (source && !strcmp(source->scope_id ? source->scope_id : "", scope ? scope : "")) ||
                             (target && !strcmp(target->scope_id ? target->scope_id : "", scope ? scope : ""));
        bool endpoint_mismatch = edge_claims_scope &&
            (!source || !target || strcmp(source->scope_id ? source->scope_id : "", scope ? scope : "") ||
             strcmp(target->scope_id ? target->scope_id : "", scope ? scope : ""));
        if (endpoint_mismatch || (touches_scope && !edge_claims_scope)) {
            *scope_message = nn_text_copy("scope contains a cross-scope or malformed edge");
            nn_inference_scope_set_status(evaluation, scope, NN_INFERENCE_SEMANTIC_ERROR, *scope_message);
            free(outputs); free(indegree); free(done); return NN_INFERENCE_SEMANTIC_ERROR;
        }
    }
    for (size_t e = 0; e < nn_model_edge_count(model); ++e) {
        const NNEdge *edge = nn_model_edge_at(model, e);
        if (strcmp(edge->scope_id ? edge->scope_id : "", scope ? scope : "")) continue;
        size_t target = nn_inference_find_node_index(model, edge->target_id);
        const NNNode *target_node = target != (size_t)-1 ? nn_model_node_at(model, target) : NULL;
        if (target_node && !strcmp(target_node->scope_id ? target_node->scope_id : "", scope ? scope : "")) ++indegree[target];
    }
    NNInferenceStatus output_status = NN_INFERENCE_UNRESOLVED;
    NNInferenceStatus scope_failure = NN_INFERENCE_UNRESOLVED;
    char *failure_message = NULL;
    for (size_t step = 0; step < scope_nodes; ++step) {
        size_t index = count;
        for (size_t i = 0; i < count; ++i) {
            const NNNode *candidate = nn_model_node_at(model, i);
            if (!done[i] && !strcmp(candidate->scope_id ? candidate->scope_id : "", scope ? scope : "") && !indegree[i]) { index = i; break; }
        }
        if (index == count) break;
        done[index] = true;
        const NNNode *node = nn_model_node_at(model, index);
        const NNPackage *package = nn_catalog_find(evaluation->catalog, node->package_id, node->package_version);
        size_t incoming_count = 0;
        for (size_t e = 0; e < nn_model_edge_count(model); ++e) {
            const NNEdge *edge = nn_model_edge_at(model, e);
            if (!strcmp(edge->target_id, node->id) && !strcmp(edge->scope_id ? edge->scope_id : "", scope ? scope : "")) ++incoming_count;
        }
        Tensor *inputs = calloc(incoming_count ? incoming_count : 1, sizeof(*inputs));
        Incoming *incoming = calloc(incoming_count ? incoming_count : 1, sizeof(*incoming));
        size_t used = 0; bool missing = !inputs || !incoming, malformed = false;
        if (!inputs || !incoming) evaluation->allocation_failed = true;
        if (inputs && incoming) for (size_t e = 0; e < nn_model_edge_count(model); ++e) {
            const NNEdge *edge = nn_model_edge_at(model, e);
            if (strcmp(edge->target_id, node->id) || strcmp(edge->scope_id ? edge->scope_id : "", scope ? scope : "")) continue;
            size_t order = e;
            if (package && !strcmp(package->kind, "join") && !nn_join_handle_order(edge->target_handle_id, &order)) { malformed = true; break; }
            incoming[used++] = (Incoming){ .edge = edge, .order = order };
        }
        if (package && !strcmp(package->kind, "join") && !malformed && used > 1) {
            qsort(incoming, used, sizeof(*incoming), incoming_compare);
            for (size_t i = 1; i < used; ++i) if (incoming[i-1].order == incoming[i].order) malformed = true;
        }
        size_t edge_count = used; used = 0;
        if (inputs && incoming && !malformed) for (size_t i = 0; i < edge_count; ++i) {
            size_t source = nn_inference_find_node_index(model, incoming[i].edge->source_id);
            Tensor *selected = source == (size_t)-1 ? NULL :
                nn_inference_node_output_find(&outputs[source], incoming[i].edge->source_handle_id);
            if (!selected || !selected->dtype) { missing = true; break; }
            inputs[used++] = *selected;
        }
        char *message = NULL, *source_file = NULL, *cause_node_id = NULL;
        size_t source_line = 0;
        Tensor output[2] = {{0}}; NNInferenceStatus status;
        if (malformed) { status = NN_INFERENCE_SEMANTIC_ERROR; message = nn_text_copy("join target handle must be in-<positive integer>"); }
        else if (missing) { status = NN_INFERENCE_UNRESOLVED; message = nn_text_copy("an upstream tensor is unresolved"); }
        else if (!package) { status = NN_INFERENCE_RUNTIME_FAULT; message = nn_text_copy("package is absent from active catalog"); }
        else if (incoming_count == 0 && (!package->kind || strcmp(package->kind, "input"))) {
            status = NN_INFERENCE_UNRESOLVED;
            message = nn_text_copy("node has no upstream tensor");
        }
        else status = nn_inference_execute_rule(evaluation, node, package, depth,
                                   node == nn_model_node_at(model, input_index) ? inherited : NULL,
                                    inputs, used, output, &message, &source_file,
                                   &source_line, &cause_node_id);
        if (status == NN_INFERENCE_UNRESOLVED && !cause_node_id && missing) {
            for (size_t e = 0; incoming && e < edge_count && !cause_node_id; ++e) {
                const NNEdge *edge = incoming[e].edge;
                const NNNode *source = nn_model_find_node(model, edge->source_id);
                Result *upstream = source ? nn_inference_report_result(evaluation, source) : NULL;
                if (!upstream) continue;
                if (upstream->cause_node_id) {
                    cause_node_id = nn_text_copy(upstream->cause_node_id);
                    if (!cause_node_id) evaluation->allocation_failed = true;
                }
                else if (upstream->view.status == NN_INFERENCE_SEMANTIC_ERROR ||
                         upstream->view.status == NN_INFERENCE_COMPILATION_ERROR ||
                         upstream->view.status == NN_INFERENCE_RUNTIME_FAULT ||
                         upstream->view.status == NN_INFERENCE_UNRESOLVED)
                    { cause_node_id = nn_text_copy(source->id);
                      if (!cause_node_id) evaluation->allocation_failed = true; }
            }
        }
        if (index == output_index && scope_cause && cause_node_id) {
            *scope_cause = nn_text_copy(cause_node_id);
            if (!*scope_cause) evaluation->allocation_failed = true;
        }
        Result *result = nn_inference_report_result(evaluation, node);
        bool result_added = !result || nn_inference_result_set(result, node, status,
            message, status == NN_INFERENCE_SUCCESS ? &output[0] : NULL,
            cause_node_id, source_file, source_line,
            cause_node_id ? "model.blocked" : NULL);
        free(cause_node_id);
        cause_node_id = NULL;
        if (!result_added) evaluation->report->failed = true;
        free(source_file);
        if (status == NN_INFERENCE_RUNTIME_FAULT) {
            scope_failure = status;
            free(failure_message);
            failure_message = nn_text_copy(message ? message : "nested inference runtime fault");
        } else if (status == NN_INFERENCE_SEMANTIC_ERROR && scope_failure == NN_INFERENCE_UNRESOLVED) {
            scope_failure = status;
            failure_message = nn_text_copy(message ? message : "nested semantic error");
        }
        if (status == NN_INFERENCE_SUCCESS) {
            bool terminal = package && package->kind &&
                (!strcmp(package->kind, "output") || !strcmp(package->kind, "loss-output"));
            if ((terminal && nested_scope &&
                 !nn_inference_node_output_add_mapping(&outputs[index], node, package, &output[0])) ||
                (!terminal && !nn_inference_node_output_add(&outputs[index], package, output))) {
                /* Host OOM discards the whole report. message already belongs to
                   result; do not replace/free it or allocate a secondary error. */
                evaluation->allocation_failed = true;
                status = NN_INFERENCE_RUNTIME_FAULT;
                for (size_t o = 0; o < 2; ++o) nn_inference_tensor_dispose(&output[o]);
            } else {
                if (!terminal && package && package->output_count) {
                    Result *typed = nn_inference_report_result(evaluation, node);
                    if (typed && !nn_inference_result_set_output_metadata(typed, package, output))
                        evaluation->report->failed = true;
                    for (size_t o = 0; o < 2; ++o) nn_inference_tensor_dispose(&output[o]);
                } else {
                    /* Terminal rules inspect the consumed tensor but publish no handle. */
                    for (size_t o = 0; o < 2; ++o) nn_inference_tensor_dispose(&output[o]);
                }
            }
        } else for (size_t o = 0; o < 2; ++o) nn_inference_tensor_dispose(&output[o]);
        if (index == output_index) {
            output_status = status;
            *scope_message = nn_text_copy(message ? message : "subflow output boundary is unresolved");
        }
        free(inputs); free(incoming);
        for (size_t e = 0; e < nn_model_edge_count(model); ++e) {
            const NNEdge *edge = nn_model_edge_at(model, e);
            if (strcmp(edge->source_id, node->id) || strcmp(edge->scope_id ? edge->scope_id : "", scope ? scope : "")) continue;
            size_t target = nn_inference_find_node_index(model, edge->target_id);
            if (target != (size_t)-1 && indegree[target]) --indegree[target];
        }
    }
    if (nested_scope) {
        output_status = NN_INFERENCE_SUCCESS;
        for (size_t h = 0; h < owner_package->output_count; ++h) {
            size_t terminal_index = mapped_index[h];
            if (terminal_index >= count || !done[terminal_index]) {
                output_status = NN_INFERENCE_UNRESOLVED;
                free(*scope_message);
                *scope_message = nn_text_copy("subflow output boundary is disconnected or cyclic");
                continue;
            }
            Result *terminal_result = nn_inference_report_result(evaluation, nn_model_node_at(model, terminal_index));
            if (!terminal_result || terminal_result->view.status != NN_INFERENCE_SUCCESS) {
                output_status = terminal_result ? terminal_result->view.status : NN_INFERENCE_UNRESOLVED;
                free(*scope_message);
                *scope_message = nn_text_copy(terminal_result && terminal_result->message
                    ? terminal_result->message : "subflow output boundary is unresolved");
                continue;
            }
            Tensor *mapped = nn_inference_node_output_find(&outputs[terminal_index],
                nn_model_node_at(model, terminal_index)->boundary_handle_id);
            if (!mapped || !nn_inference_tensor_copy(&scope_output[h], mapped)) {
                evaluation->allocation_failed = true;
                output_status = NN_INFERENCE_RUNTIME_FAULT;
                free(*scope_message);
                *scope_message = nn_text_copy("unable to copy subflow output tensor");
            }
        }
    }
    if (!nested_scope && output_index < count && !done[output_index]) {
        output_status = NN_INFERENCE_UNRESOLVED;
        *scope_message = nn_text_copy("root Output is disconnected or cyclic");
    }
    if (output_status != NN_INFERENCE_SUCCESS && scope_failure != NN_INFERENCE_UNRESOLVED) {
        output_status = scope_failure;
        free(*scope_message);
        *scope_message = failure_message;
        failure_message = NULL;
    }
    if (scope_cause && !*scope_cause && output_index < count &&
        output_status != NN_INFERENCE_SUCCESS) {
        const NNNode *output_node = nn_model_node_at(model, output_index);
        Result *output_result = nn_inference_report_result(evaluation, output_node);
        const char *root = output_result && output_result->cause_node_id
            ? output_result->cause_node_id : output_node->id;
        *scope_cause = nn_text_copy(root);
        if (!*scope_cause) evaluation->allocation_failed = true;
    }
    free(failure_message);
    for (size_t i = 0; i < count; ++i) nn_inference_node_outputs_dispose(&outputs[i]);
    free(outputs); free(indegree); free(done);
    return output_status;
}
