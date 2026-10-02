#include "automation_internal.h"
#include "automation_utils.h"
#include "utils/utils.h"

#include <stdlib.h>
#include <string.h>

static bool valid_json_tree(yyjson_val *value, unsigned depth)
{
    if (depth > 64) return false;
    if (yyjson_is_str(value)) return strlen(yyjson_get_str(value)) == yyjson_get_len(value);
    if (yyjson_is_arr(value)) {
        size_t index, count; yyjson_val *item;
        yyjson_arr_foreach(value, index, count, item)
            if (!valid_json_tree(item, depth + 1)) return false;
    } else if (yyjson_is_obj(value)) {
        size_t index, count; yyjson_val *key, *item;
        yyjson_obj_foreach(value, index, count, key, item) {
            if (strlen(yyjson_get_str(key)) != yyjson_get_len(key) ||
                yyjson_obj_get(value, yyjson_get_str(key)) != item ||
                !valid_json_tree(item, depth + 1)) return false;
        }
    }
    return true;
}

char *nn_automation_dispatch(NNApplication *app, const char *request,
                             NNAutomationUiCallback callback, void *user)
{
    yyjson_doc *input = request && strlen(request) <= NN_AUTOMATION_REQUEST_LIMIT
        ? yyjson_read(request, strlen(request), 0) : NULL;
    yyjson_val *root = input ? yyjson_doc_get_root(input) : NULL;
    const char *op = yyjson_get_str(yyjson_obj_get(root, "operation"));
    NNAutomationQuery query = { .args = yyjson_obj_get(root, "args") };
    yyjson_mut_doc *out = yyjson_mut_doc_new(NULL);
    if (!out) { yyjson_doc_free(input); return NULL; }
    yyjson_mut_val *response = yyjson_mut_obj(out), *result = yyjson_mut_null(out);
    yyjson_doc *ui_result = NULL;
    bool okay = false;
    if (!app || !yyjson_is_obj(root) || !op || !yyjson_is_obj(query.args) || !valid_json_tree(root, 0))
        nn_error_set(query.error, sizeof(query.error), "request requires operation string and args object");
    else if (!strcmp(op, "project.snapshot")) { result = nn_automation_snapshot(out, app); okay = result != NULL; }
    else if (!strcmp(op, "analysis.diagnostics")) { result = nn_automation_diagnostics(out, app, &query); okay = result != NULL; }
    else if (!strncmp(op, "ui.", 3)) {
        if (!strcmp(op, "ui.inspect") || !strcmp(op, "ui.arrange") || !strcmp(op, "ui.scope") || !strcmp(op, "ui.screenshot") || !strcmp(op, "ui.reveal")) {
            if (!callback) nn_error_set(query.error, sizeof(query.error), "UI callback unavailable");
            else {
                char *args = yyjson_val_write(query.args, 0, NULL);
                char *json = args ? callback(user, op, args, query.error, sizeof(query.error)) : NULL;
                free(args);
                if (json) ui_result = yyjson_read(json, strlen(json), 0);
                free(json);
                if (ui_result) { result = yyjson_val_mut_copy(out, yyjson_doc_get_root(ui_result)); okay = result != NULL; }
            }
        } else nn_error_set(query.error, sizeof(query.error), "unknown UI operation");
    } else okay = nn_automation_execute(app, op, &query);
    if (!okay && !query.error[0]) nn_error_set(query.error, sizeof(query.error), "operation failed");
    bool serialized = response && yyjson_mut_obj_add_bool(out, response, "ok", okay);
    if (okay) serialized = serialized && result && yyjson_mut_obj_add_val(out, response, "result", result);
    else serialized = serialized && nn_automation_json_string(out, response, "error", query.error);
    yyjson_mut_doc_set_root(out, response);
    char *json = serialized ? yyjson_mut_write(out, YYJSON_WRITE_NEWLINE_AT_END, NULL) : NULL;
    yyjson_doc_free(ui_result); yyjson_doc_free(input); yyjson_mut_doc_free(out);
    return json ? json : nn_text_copy("{\"ok\":false,\"error\":\"out of memory serializing response\"}\n");
}
