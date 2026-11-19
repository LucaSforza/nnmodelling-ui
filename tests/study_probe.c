/* Read-only analysis probe for the expert usability study.
 *
 * Usage: study_probe [--core DIR] [--set NAME=VALUE]... PROJECT_DIR
 * Prints {complete, problems, tensors, parameters, metavariables,
 * resolved_model} as one JSON object. Overrides exist in memory only; the
 * project is closed with discard and never saved. See
 * docs/knowledge/testing/expert-study.md. */
#define _XOPEN_SOURCE 700
#include "application/application.h"
#include "automation/automation.h"
#include "yyjson.h"

#include <errno.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_OVERRIDES 32
#define MAX_TEXT 4096
#define ERROR_CAP 512

static int fail(const char *message)
{
    yyjson_mut_doc *doc = yyjson_mut_doc_new(NULL);
    yyjson_mut_val *root = doc ? yyjson_mut_obj(doc) : NULL;
    char *text = NULL;
    if (root && yyjson_mut_obj_add_strcpy(doc, root, "error", message)) {
        yyjson_mut_doc_set_root(doc, root);
        text = yyjson_mut_write(doc, 0, NULL);
    }
    puts(text ? text : "{\"error\":\"out of memory\"}");
    free(text);
    yyjson_mut_doc_free(doc);
    return 1;
}

/* Replace the value of one metavariable entry, keeping its declared type. */
static bool set_value(yyjson_mut_doc *doc, yyjson_mut_val *entry, const char *text,
                      char *error)
{
    const char *type = yyjson_mut_get_str(yyjson_mut_obj_get(entry, "type"));
    yyjson_mut_val *value = NULL;
    char *end = NULL;
    if (!type) { snprintf(error, ERROR_CAP, "metavariable has no type"); return false; }
    errno = 0;
    if (!strcmp(type, "integer")) {
        long long number = strtoll(text, &end, 10);
        if (*text && !*end && !errno) value = yyjson_mut_sint(doc, number);
    } else if (!strcmp(type, "number")) {
        double number = strtod(text, &end);
        if (*text && !*end && !errno && isfinite(number)) value = yyjson_mut_real(doc, number);
    } else if (!strcmp(type, "boolean")) {
        if (!strcmp(text, "true") || !strcmp(text, "false"))
            value = yyjson_mut_bool(doc, text[0] == 't');
    } else if (!strcmp(type, "json")) {
        yyjson_doc *parsed = yyjson_read(text, strlen(text), 0);
        value = parsed ? yyjson_val_mut_copy(doc, yyjson_doc_get_root(parsed)) : NULL;
        yyjson_doc_free(parsed);
    } else {
        value = yyjson_mut_strcpy(doc, text);
    }
    if (!value) { snprintf(error, ERROR_CAP, "value '%s' is not a valid %s", text, type); return false; }
    yyjson_mut_obj_remove_str(entry, "value");
    if (!yyjson_mut_obj_add_val(doc, entry, "value", value)) {
        snprintf(error, ERROR_CAP, "out of memory");
        return false;
    }
    return true;
}

/* Apply NAME=VALUE overrides to the in-memory project; never saves. */
static bool apply_overrides(NNApplication *app, char *const *overrides, int count, char *error)
{
    char *current = nn_app_metavariables_json(app, error, ERROR_CAP);
    if (!current) return false;
    yyjson_doc *source = yyjson_read(current, strlen(current), 0);
    yyjson_mut_doc *doc = source ? yyjson_doc_mut_copy(source, NULL) : NULL;
    yyjson_mut_val *array = doc ? yyjson_mut_doc_get_root(doc) : NULL;
    char *updated = NULL;
    bool okay = array != NULL;
    if (!okay) snprintf(error, ERROR_CAP, "cannot read metavariables");
    for (int i = 0; okay && i < count; ++i) {
        char *equals = strchr(overrides[i], '=');
        if (!equals || equals == overrides[i]) {
            snprintf(error, ERROR_CAP, "--set expects NAME=VALUE, got '%s'", overrides[i]);
            okay = false;
            break;
        }
        *equals = '\0';
        yyjson_mut_val *entry = NULL, *item;
        yyjson_mut_arr_iter iter = yyjson_mut_arr_iter_with(array);
        while ((item = yyjson_mut_arr_iter_next(&iter))) {
            const char *name = yyjson_mut_get_str(yyjson_mut_obj_get(item, "name"));
            if (name && !strcmp(name, overrides[i])) entry = item;
        }
        if (!entry) snprintf(error, ERROR_CAP, "unknown metavariable '%s'", overrides[i]);
        okay = entry && set_value(doc, entry, equals + 1, error);
        *equals = '=';
    }
    if (okay && !(updated = yyjson_mut_write(doc, 0, NULL))) {
        snprintf(error, ERROR_CAP, "out of memory");
        okay = false;
    }
    if (okay) okay = nn_app_set_metavariables_json(app, updated, error, ERROR_CAP);
    free(updated);
    yyjson_mut_doc_free(doc);
    yyjson_doc_free(source);
    nn_app_free_text(current);
    return okay;
}

static bool add_json_text(yyjson_mut_doc *doc, yyjson_mut_val *root, const char *key,
                          const char *text)
{
    yyjson_doc *parsed = text ? yyjson_read(text, strlen(text), 0) : NULL;
    yyjson_mut_val *value = parsed ? yyjson_val_mut_copy(doc, yyjson_doc_get_root(parsed))
                                   : yyjson_mut_null(doc);
    bool okay = value && yyjson_mut_obj_add_val(doc, root, key, value);
    yyjson_doc_free(parsed);
    return okay;
}

/* Copy the automation diagnostics result so the UI, IPC and probe agree. */
static bool add_diagnostics(yyjson_mut_doc *doc, yyjson_mut_val *root, NNApplication *app,
                            char *error)
{
    char *response = nn_automation_dispatch(app,
        "{\"operation\":\"analysis.diagnostics\",\"args\":{}}", NULL, NULL);
    yyjson_doc *parsed = response ? yyjson_read(response, strlen(response), 0) : NULL;
    yyjson_val *reply = parsed ? yyjson_doc_get_root(parsed) : NULL;
    yyjson_val *result = yyjson_obj_get(reply, "result");
    bool okay = false;
    if (!result) {
        const char *message = yyjson_get_str(yyjson_obj_get(reply, "error"));
        snprintf(error, ERROR_CAP, "%s", message ? message : "diagnostics failed");
    } else {
        okay = true;
        static const char *const keys[] = { "complete", "problems", "tensors" };
        for (size_t i = 0; okay && i < sizeof(keys) / sizeof(keys[0]); ++i) {
            yyjson_mut_val *copy = yyjson_val_mut_copy(doc, yyjson_obj_get(result, keys[i]));
            okay = copy && yyjson_mut_obj_add_val(doc, root, keys[i], copy);
        }
        if (!okay) snprintf(error, ERROR_CAP, "out of memory");
    }
    yyjson_doc_free(parsed);
    free(response);
    return okay;
}

static bool add_parameters(yyjson_mut_doc *doc, yyjson_mut_val *root, NNApplication *app)
{
    char error[ERROR_CAP] = "";
    char *text = nn_app_parameter_counts_json(app, error, sizeof(error));
    yyjson_doc *parsed = text ? yyjson_read(text, strlen(text), 0) : NULL;
    yyjson_val *counts = parsed ? yyjson_doc_get_root(parsed) : NULL;
    yyjson_mut_val *summary = yyjson_mut_obj(doc);
    bool okay = summary != NULL;
    static const char *const keys[] = { "total", "trainable", "complete" };
    for (size_t i = 0; okay && i < sizeof(keys) / sizeof(keys[0]); ++i) {
        yyjson_val *item = yyjson_obj_get(counts, keys[i]);
        yyjson_mut_val *copy = item ? yyjson_val_mut_copy(doc, item) : yyjson_mut_null(doc);
        okay = copy && yyjson_mut_obj_add_val(doc, summary, keys[i], copy);
    }
    okay = okay && yyjson_mut_obj_add_val(doc, root, "parameters", summary);
    yyjson_doc_free(parsed);
    nn_app_free_text(text);
    return okay;
}

static int probe(const char *core, const char *project, char *const *overrides, int count)
{
    char error[ERROR_CAP] = "", message[ERROR_CAP + 64];
    NNApplication *app = nn_app_new(core);
    if (!app) return fail("cannot create application (out of memory or bad core)");
    int status = 1;
    yyjson_mut_doc *doc = NULL;
    char *output = NULL;
    if (!nn_app_open(app, project, error, sizeof(error))) {
        snprintf(message, sizeof(message), "cannot open project: %s", error);
        status = fail(message);
        goto done;
    }
    if (!apply_overrides(app, overrides, count, error)) { status = fail(error); goto close; }
    doc = yyjson_mut_doc_new(NULL);
    yyjson_mut_val *root = doc ? yyjson_mut_obj(doc) : NULL;
    if (!root) { status = fail("out of memory"); goto close; }
    yyjson_mut_doc_set_root(doc, root);
    if (!add_diagnostics(doc, root, app, error)) { status = fail(error); goto close; }
    char *meta = nn_app_metavariables_json(app, error, sizeof(error));
    char *resolved = nn_app_resolved_model_json(app, error, sizeof(error));
    bool okay = add_parameters(doc, root, app) && add_json_text(doc, root, "metavariables", meta) &&
                add_json_text(doc, root, "resolved_model", resolved) &&
                (resolved || yyjson_mut_obj_add_strcpy(doc, root, "resolved_error", error));
    nn_app_free_text(meta);
    nn_app_free_text(resolved);
    output = okay ? yyjson_mut_write(doc, 0, NULL) : NULL;
    if (!output) { status = fail("out of memory"); goto close; }
    puts(output);
    status = 0;
close:
    /* Discard: overrides stay in memory, nothing is written to the project. */
    if (!nn_app_close(app, true, error, sizeof(error)) && !status) status = fail(error);
done:
    free(output);
    yyjson_mut_doc_free(doc);
    nn_app_free(app);
    return status;
}

int main(int argc, char **argv)
{
    const char *core = "stereotype-packages/core", *project = NULL;
    char *overrides[MAX_OVERRIDES];
    int count = 0;
    for (int i = 1; i < argc; ++i) {
        if (strlen(argv[i]) > MAX_TEXT) return fail("argument too long");
        if (!strcmp(argv[i], "--core") && i + 1 < argc) core = argv[++i];
        else if (!strcmp(argv[i], "--set") && i + 1 < argc) {
            if (count == MAX_OVERRIDES) return fail("too many --set options");
            overrides[count++] = argv[++i];
        } else if (argv[i][0] != '-' && !project) project = argv[i];
        else return fail("usage: study_probe [--core DIR] [--set NAME=VALUE]... PROJECT_DIR");
    }
    if (!project) return fail("usage: study_probe [--core DIR] [--set NAME=VALUE]... PROJECT_DIR");
    return probe(core, project, overrides, count);
}
