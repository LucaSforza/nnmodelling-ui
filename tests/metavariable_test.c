#define _XOPEN_SOURCE 700
#include "application/application.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "yyjson.h"

static char *join(const char *left, const char *right)
{
    size_t length = strlen(left) + strlen(right) + 2;
    char *path = malloc(length);
    assert(path);
    snprintf(path, length, "%s/%s", left, right);
    return path;
}

static char *read_file(const char *path)
{
    FILE *file = fopen(path, "rb");
    assert(file && fseek(file, 0, SEEK_END) == 0);
    long length = ftell(file);
    assert(length >= 0 && fseek(file, 0, SEEK_SET) == 0);
    char *text = calloc((size_t)length + 1, 1);
    assert(text && fread(text, 1, (size_t)length, file) == (size_t)length);
    assert(fclose(file) == 0);
    return text;
}

static void assert_equal_member(const yyjson_val *left, const yyjson_val *right,
                                const char *key)
{
    assert(yyjson_equals(yyjson_obj_get(left, key), yyjson_obj_get(right, key)));
}

static NNInferenceStatus node_status(NNApplication *app, const char *node_id)
{
    char error[512] = "";
    const NNInferenceReport *report = nn_app_analysis(app, error, sizeof(error));
    assert(report);
    for (size_t i = 0; i < nn_inference_count(report); ++i) {
        const NNInferenceResult *result = nn_inference_at(report, i);
        if (result && !strcmp(result->node_id, node_id)) return result->status;
    }
    assert(!"node result missing from inference report");
    return NN_INFERENCE_RUNTIME_FAULT;
}

int main(void)
{
    char temporary[] = "/tmp/nn-metavariable-test-XXXXXX";
    char *root = mkdtemp(temporary);
    assert(root);
    char cwd[4096];
    assert(getcwd(cwd, sizeof(cwd)));
    char *core = join(cwd, "stereotype-packages/core");
    NNApplication *app = nn_app_new(core);
    assert(app);
    char error[512] = "";
    assert(nn_app_create(app, root, "variables", "Variables", false, error, sizeof(error)));
    assert(nn_app_add_node(app, "dense", "core.linear", "0.1.0", "", 0, 0,
                           error, sizeof(error)));
    assert(nn_app_create_dataset(app, "local.variables-data", "1.0.0",
        "{\"name\":\"Variables data\",\"batch\":{\"inputs\":{\"input\":{\"dtype\":\"float32\",\"shape\":[\"B\",8]}},\"targets\":{}}}",
        true, error, sizeof(error)));
    assert(nn_app_add_node(app, "source", "core.input", "0.1.0", "", 0, 0,
                           error, sizeof(error)));
    assert(nn_app_set_parameter_text(app, "source", "binding", "input",
                                     error, sizeof(error)));
    assert(nn_app_connect(app, "source-dense", "source", "out", "dense", "in",
                          error, sizeof(error)));

    const char *variables = "[{\"name\":\"width\",\"type\":\"integer\",\"value\":24},{\"name\":\"scale\",\"type\":\"number\",\"value\":1}]";
    assert(nn_app_set_metavariables_json(app, variables, error, sizeof(error)));
    assert(!nn_app_can_undo(app));
    assert(nn_project_dirty(nn_app_project(app)));
    assert(nn_app_save(app, error, sizeof(error)));
    assert(!nn_project_dirty(nn_app_project(app)));
    assert(nn_app_set_metavariables_json(app, variables, error, sizeof(error)));
    assert(!nn_project_dirty(nn_app_project(app)) && !nn_app_can_undo(app));

    assert(nn_app_add_node(app, "tail", "core.relu", "0.1.0", "", 100, 0,
                           error, sizeof(error)));
    assert(nn_app_can_undo(app) && nn_project_dirty(nn_app_project(app)));
    assert(nn_app_set_metavariables_json(app, variables, error, sizeof(error)));
    assert(nn_app_can_undo(app) && nn_project_dirty(nn_app_project(app)));
    const char *bad_variables = "[{\"name\":\"width\",\"type\":\"integer\",\"value\":\"24\"}]";
    assert(!nn_app_set_metavariables_json(app, bad_variables, error, sizeof(error)));
    assert(nn_app_can_undo(app) && nn_project_dirty(nn_app_project(app)));
    const char *wrapped_variable = "[{\"name\":\"payload\",\"type\":\"json\",\"value\":[{\"$ref\":\"width\"}]}]";
    assert(!nn_app_set_metavariables_json(app, wrapped_variable, error, sizeof(error)));
    assert(nn_app_can_undo(app) && nn_project_dirty(nn_app_project(app)));

    assert(nn_app_set_parameter_text(app, "dense", "in_features", "$width",
                                     error, sizeof(error)));
    assert(nn_app_set_parameter_text(app, "dense", "out_features",
                                     "$(params.in_features * meta.scale * 2)", error, sizeof(error)));
    char *preview = nn_app_resolved_parameter_text(app, "dense", "out_features",
                                                   error, sizeof(error));
    assert(preview && !strcmp(preview, "48"));
    nn_app_free_text(preview);
    char *model_path = join(root, "variables/model.json");
    assert(nn_app_save(app, error, sizeof(error)));
    char *source_json = read_file(model_path);
    char *exported = nn_app_resolved_model_json(app, error, sizeof(error));
    assert(exported && strstr(exported, "\"schemaVersion\": 2") &&
           !strstr(exported, "metavariables") && strstr(exported, "\"in_features\": 24") &&
           strstr(exported, "\"out_features\": 48"));
    yyjson_doc *source_doc = yyjson_read(source_json, strlen(source_json), 0);
    yyjson_doc *export_doc = yyjson_read(exported, strlen(exported), 0);
    assert(source_doc && export_doc);
    const yyjson_val *source_manifest = yyjson_obj_get(yyjson_doc_get_root(source_doc), "manifest");
    const yyjson_val *export_manifest = yyjson_obj_get(yyjson_doc_get_root(export_doc), "manifest");
    assert(yyjson_get_int(yyjson_obj_get(export_manifest, "schemaVersion")) == 2);
    assert(!yyjson_obj_get(export_manifest, "metavariables"));
    assert_equal_member(source_manifest, export_manifest, "id");
    assert_equal_member(source_manifest, export_manifest, "version");
    assert_equal_member(source_manifest, export_manifest, "name");
    assert_equal_member(source_manifest, export_manifest, "description");
    assert_equal_member(source_manifest, export_manifest, "customPackages");
    assert_equal_member(source_manifest, export_manifest, "customDatasets");
    assert_equal_member(source_manifest, export_manifest, "operations");
    const yyjson_val *export_nodes = yyjson_obj_get(yyjson_doc_get_root(export_doc), "nodes");
    const yyjson_val *source_nodes = yyjson_obj_get(yyjson_doc_get_root(source_doc), "nodes");
    assert(yyjson_arr_size(source_nodes) == yyjson_arr_size(export_nodes));
    bool found_dense = false;
    for (size_t i = 0; i < yyjson_arr_size(export_nodes); ++i) {
        const yyjson_val *candidate = yyjson_arr_get(export_nodes, i);
        if (!yyjson_equals_str(yyjson_obj_get(candidate, "id"), "dense")) continue;
        found_dense = true;
        const yyjson_val *data = yyjson_obj_get(candidate, "data");
        const yyjson_val *params = yyjson_obj_get(data, "params");
        assert(yyjson_is_int(yyjson_obj_get(params, "in_features")) &&
               yyjson_get_int(yyjson_obj_get(params, "in_features")) == 24);
        assert(yyjson_is_int(yyjson_obj_get(params, "out_features")) &&
               yyjson_get_int(yyjson_obj_get(params, "out_features")) == 48);
        assert(yyjson_is_bool(yyjson_obj_get(params, "bias")));
        assert(yyjson_is_str(yyjson_obj_get(params, "dtype")));
    }
    assert(found_dense && !nn_project_dirty(nn_app_project(app)));
    yyjson_doc_free(source_doc); yyjson_doc_free(export_doc); free(source_json);
    nn_app_free_text(exported);
    char *source_binding = nn_app_parameter_text(app, "dense", "in_features");
    assert(source_binding && !strcmp(source_binding, "{\"$ref\":\"width\"}"));
    nn_app_free_text(source_binding);

    assert(nn_app_set_parameter_text(app, "dense", "out_features", "$(true)",
                                     error, sizeof(error)));
    assert(!nn_app_resolved_parameter_text(app, "dense", "out_features",
                                           error, sizeof(error)));
    assert(strstr(error, "dense") && strstr(error, "out_features"));
    assert(node_status(app, "dense") == NN_INFERENCE_SEMANTIC_ERROR);
    assert(nn_app_set_parameter_text(app, "dense", "out_features", "$(1.5)",
                                     error, sizeof(error)));
    assert(!nn_app_resolved_parameter_text(app, "dense", "out_features",
                                           error, sizeof(error)));
    assert(strstr(error, "dense") && strstr(error, "out_features"));
    assert(node_status(app, "dense") == NN_INFERENCE_SEMANTIC_ERROR);
    assert(nn_app_set_parameter_text(app, "dense", "in_features", "$(params.out_features)",
                                    error, sizeof(error)));
    assert(nn_app_set_parameter_text(app, "dense", "out_features", "$(params.in_features)",
                                    error, sizeof(error)));
    assert(!nn_app_resolved_parameter_text(app, "dense", "out_features",
                                           error, sizeof(error)));
    assert(strstr(error, "dense") && (strstr(error, "in_features") || strstr(error, "out_features")));
    assert(node_status(app, "dense") == NN_INFERENCE_SEMANTIC_ERROR);
    assert(nn_app_set_parameter_text(app, "dense", "out_features", "$(1 + )",
                                     error, sizeof(error)));
    assert(node_status(app, "dense") == NN_INFERENCE_COMPILATION_ERROR);
    assert(nn_app_set_parameter_text(app, "dense", "in_features", "$width",
                                     error, sizeof(error)));
    assert(nn_app_set_parameter_text(app, "dense", "out_features",
                                     "$((function() while true do end end)())",
                                     error, sizeof(error)));
    assert(!nn_app_resolved_parameter_text(app, "dense", "out_features",
                                           error, sizeof(error)));
    assert(strstr(error, "dense") && strstr(error, "out_features"));
    assert(node_status(app, "dense") == NN_INFERENCE_RUNTIME_FAULT);
    assert(nn_app_set_parameter_text(app, "dense", "out_features", "$(io.open('/etc/passwd'))",
                                     error, sizeof(error)));
    assert(!nn_app_resolved_parameter_text(app, "dense", "out_features",
                                           error, sizeof(error)));
    assert(strstr(error, "dense") && strstr(error, "out_features"));
    assert(nn_app_set_parameter_text(app, "dense", "out_features",
        "$( (function() local t = {x = 1}; for i = 1, 12 do t = {a = t, b = t} end; return t end)() )",
        error, sizeof(error)));
    assert(!nn_app_resolved_parameter_text(app, "dense", "out_features",
                                           error, sizeof(error)));
    assert(strstr(error, "dense") && strstr(error, "out_features"));
    assert(nn_app_set_parameter_text(app, "dense", "out_features",
        "$( (function() local t = 1; for i = 1, 60 do t = {t} end; return t end)() )",
        error, sizeof(error)));
    assert(!nn_app_resolved_parameter_text(app, "dense", "out_features",
                                           error, sizeof(error)));
    assert(strstr(error, "dense") && strstr(error, "out_features"));
    assert(nn_app_set_parameter_text(app, "dense", "out_features",
        "$( (function f(n) return f(n + 1) end)(1) )",
        error, sizeof(error)));
    assert(!nn_app_resolved_parameter_text(app, "dense", "out_features",
                                           error, sizeof(error)));
    assert(strstr(error, "dense") && strstr(error, "out_features"));
    assert(nn_app_set_parameter_text(app, "dense", "out_features", "$width",
                                     error, sizeof(error)));
    preview = nn_app_resolved_parameter_text(app, "dense", "out_features",
                                            error, sizeof(error));
    assert(preview && !strcmp(preview, "24"));
    nn_app_free_text(preview);
    assert(nn_app_set_parameter_text(app, "source", "binding", "$$slot",
                                     error, sizeof(error)));
    char *escaped = nn_app_resolved_parameter_text(app, "source", "binding",
                                                   error, sizeof(error));
    assert(escaped && !strcmp(escaped, "$slot"));
    nn_app_free_text(escaped);
    assert(nn_app_add_node(app, "parallel", "core.horizontal-repeat", "0.1.0", "", 0, 0,
                           error, sizeof(error)));
    assert(nn_app_set_parameter_text(app, "parallel", "join",
        "{\"id\":\"core.concat\",\"version\":\"0.1.0\",\"parameters\":{\"dim\":{\"$ref\":\"width\"}}}",
        error, sizeof(error)));
    char *resolved_join = nn_app_resolved_parameter_text(app, "parallel", "join", error, sizeof(error));
    assert(resolved_join && strstr(resolved_join, "core.concat") && strstr(resolved_join, "\"dim\":24"));
    nn_app_free_text(resolved_join);
    assert(nn_app_set_parameter_text(app, "parallel", "join",
        "{\"id\":\"core.concat\",\"version\":\"0.1.0\",\"parameters\":{\"dim\":{\"$lua\":\"meta.width\"}}}",
        error, sizeof(error)));
    resolved_join = nn_app_resolved_parameter_text(app, "parallel", "join", error, sizeof(error));
    assert(resolved_join && strstr(resolved_join, "\"dim\":24"));
    nn_app_free_text(resolved_join);
    assert(nn_app_set_parameter_text(app, "dense", "out_features", "$missing",
                                     error, sizeof(error)));
    assert(!nn_app_resolved_model_json(app, error, sizeof(error)));
    assert(strstr(error, "missing") && strstr(error, "dense"));
    assert(node_status(app, "dense") == NN_INFERENCE_UNRESOLVED);
    assert(nn_app_set_parameter_text(app, "dense", "out_features", "$(params.in_features)",
                                     error, sizeof(error)));

    assert(nn_app_save(app, error, sizeof(error)));
    char *saved = read_file(model_path);
    assert(strstr(saved, "\"schemaVersion\": 3") && strstr(saved, "\"metavariables\"") &&
           strstr(saved, "\"$ref\"") && strstr(saved, "\"$lua\""));
    free(saved);
    NNApplication *reopened = nn_app_new(core);
    char *project_path = join(root, "variables");
    assert(reopened && nn_app_open(reopened, project_path, error, sizeof(error)));
    char *roundtrip = nn_app_resolved_parameter_text(reopened, "dense", "out_features",
                                                     error, sizeof(error));
    assert(roundtrip && !strcmp(roundtrip, "24"));
    nn_app_free_text(roundtrip);
    nn_app_free(reopened);
    free(project_path);

    nn_app_free(app);
    free(model_path);
    free(core);
    return 0;
}
