#include "application_internal.h"
#include "utils/utils.h"
#include "inference/inference.h"

#include <stdlib.h>
#include <string.h>

enum { NN_ANALYSIS_NODE_LIMIT = 50 };

void nn_app_event_begin(NNAppEvent *event, NNApplication *app, char *error, size_t cap)
{
    memset(event, 0, sizeof(*event));
    event->app = app;
    event->error = error;
    event->cap = cap;
    if (!app || !app->activity) return;
    event->doc = yyjson_mut_doc_new(NULL);
    event->data = event->doc ? yyjson_mut_obj(event->doc) : NULL;
    if (!event->data) { yyjson_mut_doc_free(event->doc); event->doc = NULL; return; }
    event->caller = error;
    event->caller_cap = cap;
    event->local[0] = '\0';
    event->error = event->local;
    event->cap = sizeof(event->local);
}

void nn_app_event_str(NNAppEvent *event, const char *key, const char *text)
{
    if (!event->doc) return;
    if (text) yyjson_mut_obj_add_strcpy(event->doc, event->data, key, text);
    else yyjson_mut_obj_add_null(event->doc, event->data, key);
}

void nn_app_event_path(NNAppEvent *event, const char *key, const char *path)
{
    if (!event->doc) return;
    char *redacted = nn_activity_redact_path(path);
    nn_app_event_str(event, key, redacted ? redacted : path);
    free(redacted);
}

void nn_app_event_num(NNAppEvent *event, const char *key, double number)
{
    if (event->doc) yyjson_mut_obj_add_real(event->doc, event->data, key, number);
}

void nn_app_event_bool(NNAppEvent *event, const char *key, bool value)
{
    if (event->doc) yyjson_mut_obj_add_bool(event->doc, event->data, key, value);
}

static void emit_doc(NNApplication *app, const char *name, bool ok, const char *error,
                     yyjson_mut_doc *doc, yyjson_mut_val *root)
{
    yyjson_mut_doc_set_root(doc, root);
    char *text = yyjson_mut_write(doc, 0, NULL);
    if (text) nn_activity_emit(app->activity, app->activity_source, name, ok, error, text);
    free(text);
}

static yyjson_mut_val *snapshot_value(yyjson_mut_doc *doc, const NNValue *value)
{
    yyjson_mut_val *json = nn_app_value_json(doc, value);
    return json ? json : yyjson_mut_null(doc);
}

static void emit_snapshot(NNApplication *app)
{
    const NNProject *project = app->project;
    const NNModel *model = project ? nn_project_model((NNProject *)project) : NULL;
    if (!model) return;
    yyjson_mut_doc *doc = yyjson_mut_doc_new(NULL);
    yyjson_mut_val *root = doc ? yyjson_mut_obj(doc) : NULL;
    yyjson_mut_val *nodes = root ? yyjson_mut_arr(doc) : NULL;
    yyjson_mut_val *edges = nodes ? yyjson_mut_arr(doc) : NULL;
    if (!edges) { yyjson_mut_doc_free(doc); return; }
    const NNDataset *active = nn_project_active_dataset(project);
    yyjson_mut_val *dataset = NULL;
    if (active) {
        dataset = yyjson_mut_obj(doc);
        if (dataset) {
            yyjson_mut_obj_add_strcpy(doc, dataset, "id", active->id);
            yyjson_mut_obj_add_strcpy(doc, dataset, "version", active->version);
        }
    } else dataset = yyjson_mut_null(doc);
    yyjson_mut_obj_add_strcpy(doc, root, "project", nn_project_id(project));
    yyjson_mut_obj_add_bool(doc, root, "dirty", nn_project_dirty(project));
    if (dataset) yyjson_mut_obj_add_val(doc, root, "active_dataset", dataset);
    for (size_t i = 0; i < nn_model_node_count(model); ++i) {
        const NNNode *node = nn_model_node_at(model, i);
        yyjson_mut_val *item = yyjson_mut_obj(doc), *params = yyjson_mut_obj(doc);
        if (!item || !params) continue;
        yyjson_mut_obj_add_strcpy(doc, item, "id", node->id);
        yyjson_mut_obj_add_strcpy(doc, item, "scope", node->scope_id);
        yyjson_mut_obj_add_strcpy(doc, item, "package", node->package_id);
        yyjson_mut_obj_add_strcpy(doc, item, "version", node->package_version);
        yyjson_mut_obj_add_strcpy(doc, item, "name", node->label);
        for (size_t p = 0; p < node->parameter_count; ++p)
            yyjson_mut_obj_add_val(doc, params, node->parameters[p].key,
                                   snapshot_value(doc, &node->parameters[p].value));
        yyjson_mut_obj_add_val(doc, item, "params", params);
        yyjson_mut_arr_append(nodes, item);
    }
    yyjson_mut_obj_add_val(doc, root, "nodes", nodes);
    for (size_t i = 0; i < nn_model_edge_count(model); ++i) {
        const NNEdge *edge = nn_model_edge_at(model, i);
        yyjson_mut_val *item = yyjson_mut_obj(doc);
        if (!item) continue;
        yyjson_mut_obj_add_strcpy(doc, item, "id", edge->id);
        yyjson_mut_obj_add_strcpy(doc, item, "source", edge->source_id);
        yyjson_mut_obj_add_strcpy(doc, item, "source_handle", edge->source_handle_id);
        yyjson_mut_obj_add_strcpy(doc, item, "target", edge->target_id);
        yyjson_mut_obj_add_strcpy(doc, item, "target_handle", edge->target_handle_id);
        yyjson_mut_arr_append(edges, item);
    }
    yyjson_mut_obj_add_val(doc, root, "edges", edges);
    emit_doc(app, "model.snapshot", true, NULL, doc, root);
    yyjson_mut_doc_free(doc);
}

static bool snapshot_event(const char *name)
{
    static const char *const skipped[] = {"node.move", "node.rename", "edit.begin",
                                          "project.save", "project.close"};
    for (size_t i = 0; i < sizeof(skipped) / sizeof(*skipped); ++i)
        if (!strcmp(name, skipped[i])) return false;
    return true;
}

bool nn_app_event_end(NNAppEvent *event, bool ok, const char *name)
{
    NNApplication *app = event->app;
    if (!event->doc) return ok;
    nn_error_set(event->caller, event->caller_cap, event->local);
    emit_doc(app, name, ok, ok ? NULL : event->local, event->doc, event->data);
    yyjson_mut_doc_free(event->doc);
    event->doc = NULL;
    const bool group_end = !strcmp(name, "edit.end");
    if (!app->activity) return ok;
    if (group_end) {
        if (ok && app->snapshot_pending && app->project) emit_snapshot(app);
        app->snapshot_pending = false;
    } else if (ok && snapshot_event(name) && app->project) {
        if (app->history.group_active) app->snapshot_pending = true;
        else emit_snapshot(app);
    }
    return ok;
}

void nn_app_activity_reset(NNApplication *app)
{
    if (!app) return;
    app->snapshot_pending = false;
    free(app->analysis_signature);
    app->analysis_signature = NULL;
}

void nn_app_event_analysis(NNApplication *app, const NNInferenceReport *report)
{
    if (!app || !app->activity || !report) return;
    yyjson_mut_doc *doc = yyjson_mut_doc_new(NULL);
    yyjson_mut_val *root = doc ? yyjson_mut_obj(doc) : NULL;
    yyjson_mut_val *nodes = root ? yyjson_mut_arr(doc) : NULL;
    if (!nodes) { yyjson_mut_doc_free(doc); return; }
    NNInferenceStatus root_status = nn_inference_root_status(report);
    size_t problems = root_status == NN_INFERENCE_SUCCESS ? 0 : 1, listed = 0;
    for (size_t i = 0; i < nn_inference_count(report); ++i) {
        const NNInferenceResult *result = nn_inference_at(report, i);
        if (result->status == NN_INFERENCE_SUCCESS) continue;
        ++problems;
        yyjson_mut_val *item = listed < NN_ANALYSIS_NODE_LIMIT ? yyjson_mut_obj(doc) : NULL;
        if (!item) continue;
        yyjson_mut_obj_add_strcpy(doc, item, "id", result->node_id);
        yyjson_mut_obj_add_strcpy(doc, item, "status", nn_inference_category(result->status));
        yyjson_mut_obj_add_strcpy(doc, item, "message", result->message ? result->message : "");
        yyjson_mut_arr_append(nodes, item);
        ++listed;
    }
    yyjson_mut_obj_add_uint(doc, root, "problems", problems);
    yyjson_mut_obj_add_strcpy(doc, root, "root_status", nn_inference_category(root_status));
    yyjson_mut_obj_add_val(doc, root, "nodes", nodes);
    yyjson_mut_doc_set_root(doc, root);
    char *text = yyjson_mut_write(doc, 0, NULL);
    yyjson_mut_doc_free(doc);
    if (text && (!app->analysis_signature || strcmp(app->analysis_signature, text))) {
        nn_activity_emit(app->activity, app->activity_source, "analysis.result", true, NULL, text);
        free(app->analysis_signature);
        app->analysis_signature = text;
    } else free(text);
}
