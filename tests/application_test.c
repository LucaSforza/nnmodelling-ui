#define _XOPEN_SOURCE 700
#include "../src/application.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static char *join(const char *left, const char *right)
{
    size_t length = strlen(left) + strlen(right) + 2;
    char *path = malloc(length);
    assert(path);
    snprintf(path, length, "%s/%s", left, right);
    return path;
}

static void add(NNApplication *app, const char *id, const char *package,
                const char *scope, char *error)
{
    assert(nn_app_add_node(app, id, package, "0.1.0", scope, 10, 20, error, 512));
}

int main(void)
{
    char temporary[] = "/tmp/nn-application-test-XXXXXX";
    char *root = mkdtemp(temporary);
    assert(root);
    char cwd[4096];
    assert(getcwd(cwd, sizeof(cwd)));
    char *core = join(cwd, "stereotype-packages/core");
    char *error_project = join(root, "broken");
    assert(mkdir(error_project, 0755) == 0);
    char *broken_model = join(error_project, "model.json");
    FILE *broken = fopen(broken_model, "wb");
    assert(broken && fputs("{broken", broken) >= 0 && fclose(broken) == 0);

    NNApplication *app = nn_app_new(core);
    assert(app);
    char error[512] = "";
    assert(!nn_app_save(app, error, sizeof(error)));
    assert(nn_app_create(app, root, "demo", "Demo", false, error, sizeof(error)));
    const NNProject *project = nn_app_project(app);
    assert(project && !nn_project_dirty(project));
    const NNInferenceReport *analysis = nn_app_analysis(app, error, sizeof(error));
    assert(analysis && !nn_inference_count(analysis));
    assert(nn_app_analysis(app, error, sizeof(error)) == analysis);
    assert(!nn_app_add_node(app, "bad-pos", "core.relu", "0.1.0", "",
                            INFINITY, 0, error, sizeof(error)));
    assert(!nn_project_dirty(project));

    /* Every package currently offered by the real core catalog remains creatable. */
    const NNCatalog *catalog = nn_project_catalog(project);
    size_t package_count = nn_catalog_count(catalog);
    for (size_t i = 0; i < package_count; ++i) {
        const NNPackage *package = nn_catalog_at(catalog, i);
        char id[64];
        snprintf(id, sizeof(id), "catalog-%zu", i);
        assert(nn_app_add_node(app, id, package->id, package->version, "", 0, 0,
                               error, sizeof(error)));
        assert(nn_app_remove_node(app, id, error, sizeof(error)));
    }
    add(app, "horizontal", "core.horizontal-repeat", "", error);
    const NNNode *horizontal = nn_model_find_node(nn_app_model(app), "horizontal");
    assert(horizontal && horizontal->parameter_count == 1);
    assert(!nn_app_set_parameter_text(app, "horizontal", "join", "{}",
                                      error, sizeof(error)));
    assert(strstr(error, "object-valued stereotype parameters"));
    assert(nn_app_remove_node(app, "horizontal", error, sizeof(error)));

    add(app, "source", "core.input", "", error);
    const NNInferenceReport *after_add = nn_app_analysis(app, error, sizeof(error));
    assert(after_add && nn_inference_count(after_add) == 1);
    assert(nn_app_move_node(app, "source", 14, 25, error, sizeof(error)));
    assert(nn_app_analysis(app, error, sizeof(error)) == after_add);
    assert(!nn_app_add_node(app, "bad", "missing.package", "0.1.0", "", 0, 0,
                            error, sizeof(error)));
    assert(nn_app_analysis(app, error, sizeof(error)) == after_add);
    add(app, "a", "core.relu", "", error);
    add(app, "b", "core.relu", "", error);
    add(app, "c", "core.relu", "", error);
    add(app, "d", "core.relu", "", error);
    add(app, "join", "core.add", "", error);
    assert(nn_app_port_count(app, "source", true) == 1);
    assert(nn_app_port_count(app, "source", false) == 0);
    assert(nn_app_port_count(app, "join", false) == 2);
    char handle[32];
    assert(nn_app_port_id(app, "join", false, 0, handle, sizeof(handle)) && !strcmp(handle, "in-1"));
    assert(nn_app_port_id(app, "join", false, 1, handle, sizeof(handle)) && !strcmp(handle, "in-2"));
    assert(nn_app_port_id(app, "source", true, 0, handle, sizeof(handle)) && !strcmp(handle, "out"));
    assert(!nn_app_connect(app, "bad-direction", "source", "in", "a", "in", error, sizeof(error)));
    assert(!nn_app_connect(app, "bad-input", "source", "out", "a", "in-1", error, sizeof(error)));
    assert(nn_app_connect(app, "edge-1", "source", "out", "a", "in", error, sizeof(error)));
    assert(!nn_app_connect(app, "occupied", "source", "out", "a", "in", error, sizeof(error)));
    assert(nn_app_connect(app, "join-1", "a", "out", "join", "in-1", error, sizeof(error)));
    assert(!nn_app_connect(app, "join-alias", "b", "out", "join", "in-01", error, sizeof(error)));
    assert(!nn_app_connect(app, "join-occupied", "b", "out", "join", "in-1", error, sizeof(error)));
    assert(nn_app_connect(app, "join-2", "b", "out", "join", "in-2", error, sizeof(error)));
    assert(nn_app_port_count(app, "join", false) == 3);
    assert(nn_app_port_id(app, "join", false, 2, handle, sizeof(handle)) && !strcmp(handle, "in-3"));
    assert(nn_app_connect(app, "cycle-1", "c", "out", "d", "in", error, sizeof(error)));
    assert(!nn_app_connect(app, "cycle-2", "d", "out", "c", "in", error, sizeof(error)));

    add(app, "linear", "core.linear", "", error);
    const NNNode *linear = nn_model_find_node(nn_app_model(app), "linear");
    assert(linear);
    bool bias = false;
    bool has_bias = false;
    for (size_t i = 0; i < linear->parameter_count; ++i) {
        if (!strcmp(linear->parameters[i].key, "in_features"))
            assert(linear->parameters[i].value.type == NN_VALUE_INT && linear->parameters[i].value.as.integer == 1);
        if (!strcmp(linear->parameters[i].key, "bias")) {
            has_bias = true;
            assert(linear->parameters[i].value.type == NN_VALUE_BOOL);
            bias = linear->parameters[i].value.as.boolean;
        }
    }
    assert(has_bias && bias);
    assert(!nn_app_set_parameter_text(app, "linear", "in_features", "0", error, sizeof(error)));
    assert(!nn_app_set_parameter_text(app, "linear", "in_features", "999999999999999999999", error, sizeof(error)));
    assert(!nn_app_set_parameter_text(app, "linear", "unknown", "2", error, sizeof(error)));
    assert(!nn_app_set_parameter_text(app, "linear", "dtype", "int8", error, sizeof(error)));
    assert(nn_app_set_parameter_text(app, "linear", "bias", "false", error, sizeof(error)));
    char *text = nn_app_parameter_text(app, "linear", "dtype");
    assert(text && !strcmp(text, "float32"));
    nn_app_free_text(text);
    assert(nn_app_set_parameter_text(app, "linear", "dtype", "float64", error, sizeof(error)));
    text = nn_app_parameter_text(app, "linear", "dtype");
    assert(text && !strcmp(text, "float64"));
    nn_app_free_text(text);
    char *input_binding = nn_app_parameter_text(app, "source", "binding");
    assert(input_binding && !strcmp(input_binding, ""));
    nn_app_free_text(input_binding);

    add(app, "subflow", "core.repeat", "", error);
    assert(nn_app_node_is_subflow(app, "subflow"));
    assert(!nn_app_add_node(app, "orphan", "core.relu", "0.1.0", "missing", 0, 0,
                            error, sizeof(error)));
    add(app, "child", "core.relu", "subflow", error);
    assert(!nn_app_remove_node(app, "subflow", error, sizeof(error)));
    assert(nn_app_remove_node(app, "child", error, sizeof(error)));
    assert(nn_app_remove_node(app, "subflow", error, sizeof(error)));
    assert(!nn_app_node_is_subflow(app, "source"));

    const char *directory = nn_project_directory(nn_app_project(app));
    char *saved_directory = strdup(directory);
    assert(saved_directory);
    assert(!nn_app_move_node(app, "a", NAN, 0, error, sizeof(error)));
    assert(!nn_app_open(app, error_project, error, sizeof(error)));
    assert(nn_app_project(app) == project && nn_project_dirty(project));
    assert(nn_app_save(app, error, sizeof(error)));
    assert(!nn_project_dirty(project));
    assert(nn_app_close(app, false, error, sizeof(error)));
    assert(nn_app_project(app) == NULL);
    assert(nn_app_open(app, saved_directory, error, sizeof(error)));
    assert(nn_model_find_node(nn_app_model(app), "linear"));
    assert(nn_model_find_node(nn_app_model(app), "join"));
    assert(nn_model_edge_count(nn_app_model(app)) == 4);
    linear = nn_model_find_node(nn_app_model(app), "linear");
    assert(linear);
    for (size_t i = 0; i < linear->parameter_count; ++i)
        if (!strcmp(linear->parameters[i].key, "bias"))
            assert(!linear->parameters[i].value.as.boolean);
    assert(nn_app_close(app, true, error, sizeof(error)));
    nn_app_free(app);

    free(saved_directory);
    free(broken_model);
    free(error_project);
    free(core);
    puts("application boundary: ok");
    return 0;
}
