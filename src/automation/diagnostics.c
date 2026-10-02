#include "automation_internal.h"
#include "automation_utils.h"
#include "utils/utils.h"

static bool nullable_string(yyjson_mut_doc *doc, yyjson_mut_val *object,
                             const char *key, const char *text)
{
    return text ? nn_automation_json_string(doc, object, key, text) : yyjson_mut_obj_add_null(doc, object, key);
}

static bool append_tensor(yyjson_mut_doc *doc, yyjson_mut_val *tensors,
                          const char *node_id, const char *handle, const char *type,
                          const char *dtype, const char *const *dimensions, size_t count)
{
    yyjson_mut_val *item = yyjson_mut_obj(doc), *shape = yyjson_mut_arr(doc);
    if (!item || !shape || !nn_automation_json_string(doc, item, "node", node_id) ||
        !nullable_string(doc, item, "handle", handle) ||
        !nullable_string(doc, item, "type", type) ||
        !nn_automation_json_string(doc, item, "dtype", dtype)) return false;
    for (size_t d = 0; d < count; ++d) {
        yyjson_mut_val *dimension = yyjson_mut_strcpy(doc, dimensions[d]);
        if (!dimension || !yyjson_mut_arr_append(shape, dimension)) return false;
    }
    return yyjson_mut_obj_add_val(doc, item, "shape", shape) &&
           yyjson_mut_arr_append(tensors, item);
}

yyjson_mut_val *nn_automation_diagnostics(yyjson_mut_doc *doc, NNApplication *app, NNAutomationQuery *q)
{
    const NNInferenceReport *report = nn_app_analysis(app, q->error, sizeof(q->error));
    if (!report) return NULL;
    yyjson_mut_val *result = yyjson_mut_obj(doc), *problems = yyjson_mut_arr(doc), *tensors = yyjson_mut_arr(doc);
    if (!result || !problems || !tensors) goto oom;
    NNInferenceStatus root_status = nn_inference_root_status(report);
    bool complete = root_status == NN_INFERENCE_SUCCESS;
    if (!complete) {
        yyjson_mut_val *item = yyjson_mut_obj(doc);
        const char *code = root_status == NN_INFERENCE_SEMANTIC_ERROR
            ? "model.semantic" : "model.incomplete";
        if (!item || !yyjson_mut_obj_add_null(doc, item, "node") ||
            !nn_automation_json_string(doc, item, "scope", "") ||
            !yyjson_mut_obj_add_null(doc, item, "package") ||
            !nn_automation_json_string(doc, item, "code", code) ||
            !nn_automation_json_string(doc, item, "category", nn_inference_category(root_status)) ||
            !nn_automation_json_string(doc, item, "severity", nn_inference_severity(root_status)) ||
            !nn_automation_json_string(doc, item, "message", nn_inference_root_message(report)) ||
            !yyjson_mut_obj_add_null(doc, item, "file") ||
            !yyjson_mut_obj_add_uint(doc, item, "line", 0) ||
            !yyjson_mut_obj_add_null(doc, item, "causeNode") ||
            !yyjson_mut_arr_append(problems, item)) goto oom;
    }
    const NNModel *model = nn_app_model(app);
    for (size_t i = 0; i < nn_inference_count(report); ++i) {
        const NNInferenceResult *r = nn_inference_at(report, i);
        const NNNode *node = nn_model_find_node(model, r->node_id);
        if (r->status == NN_INFERENCE_SUCCESS) {
            if (!r->output_count) {
                if (!append_tensor(doc, tensors, r->node_id, NULL, NULL,
                                   r->dtype, r->dimensions, r->dimension_count)) goto oom;
            } else for (size_t h = 0; h < r->output_count; ++h) {
                const NNInferenceTensor *t = &r->outputs[h];
                if (!append_tensor(doc, tensors, r->node_id, t->handle_id, t->type,
                                   t->dtype, t->dimensions, t->dimension_count)) goto oom;
            }
        } else {
            yyjson_mut_val *item = yyjson_mut_obj(doc);
            if (!item || !nn_automation_json_string(doc, item, "node", r->node_id)) goto oom;
            complete = false;
            yyjson_mut_val *package = node ? nn_automation_identity(doc, node->package_id, node->package_version) : yyjson_mut_null(doc);
            if (!package || !nn_automation_json_string(doc, item, "code", r->code) ||
                !nn_automation_json_string(doc, item, "category", nn_inference_category(r->status)) ||
                !nn_automation_json_string(doc, item, "severity", nn_inference_severity(r->status)) ||
                !nn_automation_json_string(doc, item, "scope", node ? node->scope_id : "") ||
                !yyjson_mut_obj_add_val(doc, item, "package", package) ||
                !nullable_string(doc, item, "file", r->source_file) ||
                !yyjson_mut_obj_add_uint(doc, item, "line", r->source_line) ||
                !nn_automation_json_string(doc, item, "message", r->message) ||
                !nullable_string(doc, item, "causeNode", r->cause_node_id) ||
                !yyjson_mut_arr_append(problems, item)) goto oom;
        }
    }
    if (!yyjson_mut_obj_add_bool(doc, result, "available", true) ||
        !yyjson_mut_obj_add_bool(doc, result, "complete", complete) ||
        !yyjson_mut_obj_add_val(doc, result, "problems", problems) ||
        !yyjson_mut_obj_add_val(doc, result, "tensors", tensors)) goto oom;
    return result;
oom:
    nn_error_set(q->error, sizeof(q->error), "out of memory serializing analysis");
    return NULL;
}
