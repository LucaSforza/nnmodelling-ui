#define _XOPEN_SOURCE 700
#include "application/application.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static char *path_join(const char *left, const char *right)
{
    size_t size = strlen(left) + strlen(right) + 2;
    char *path = malloc(size);
    assert(path);
    snprintf(path, size, "%s/%s", left, right);
    return path;
}

static void add(NNApplication *app, const char *id, const char *package,
                const char *scope, char *error)
{
    assert(nn_app_add_node(app, id, package, "0.1.0", scope, 40, 80,
                           error, 512));
}

static void test_model_copy_and_swap(void)
{
    NNModel *model = nn_model_new();
    assert(model);
    char error[256] = "";
    assert(nn_model_add_node(model, "a", "A", "core.relu", "0.1.0", "",
                             10, 20, error, sizeof(error)));
    const NNValue leaves[] = {
        {.type = NN_VALUE_INT, .as.integer = 7},
        {.type = NN_VALUE_STRING, .as.string = "deep"},
    };
    NNValue array = {.type = NN_VALUE_ARRAY,
                     .as.array = {.items = (NNValue *)leaves, .count = 2}};
    assert(nn_model_set_parameter(model, "a", "nested", &array,
                                  error, sizeof(error)));
    assert(nn_model_set_boundary_handle(model, "a", "out", error, sizeof(error)));
    NNModel *copy = nn_model_copy(model);
    assert(copy && nn_model_equal(model, copy));
    assert(nn_model_move_node(model, "a", 100, 100, error, sizeof(error)));
    assert(!nn_model_equal(model, copy));
    nn_model_swap(model, copy);
    assert(nn_model_find_node(model, "a")->x == 20);
    assert(!nn_model_equal(model, copy));
    nn_model_free(copy);
    nn_model_free(model);
}

static void test_commands_groups_and_revisions(NNApplication *app, const char *root,
                                               char *error)
{
    const NNModel *model = nn_app_model(app);
    assert(model);
    const size_t initial_nodes = nn_model_node_count(model);
    assert(!nn_app_can_undo(app) && !nn_app_can_redo(app));
    const NNProject *project = nn_app_project(app);
    assert(project && !nn_project_dirty(project));

    add(app, "source", "core.input", "", error);
    assert(nn_app_can_undo(app) && nn_project_dirty(project));
    const size_t after_source = nn_model_node_count(nn_app_model(app));
    assert(!nn_app_open(app, root, error, 512));
    assert(nn_app_can_undo(app));
    assert(nn_model_node_count(nn_app_model(app)) == after_source);
    add(app, "layer", "core.relu", "", error);
    assert(nn_app_connect(app, "flow", "source", "out", "layer", "in",
                          error, 512));
    assert(nn_model_edge_count(nn_app_model(app)) == 1);
    assert(nn_app_undo(app, error, 512));
    assert(nn_model_edge_count(nn_app_model(app)) == 0 && nn_app_can_redo(app));
    assert(nn_app_redo(app, error, 512));
    assert(nn_model_edge_count(nn_app_model(app)) == 1);

    /* Removing a node snapshots its incident edges as part of the same edit. */
    assert(nn_app_remove_node(app, "source", error, 512));
    assert(!nn_model_find_node(nn_app_model(app), "source"));
    assert(nn_model_edge_count(nn_app_model(app)) == 0);
    assert(nn_app_undo(app, error, 512));
    assert(nn_model_find_node(nn_app_model(app), "source"));
    assert(nn_model_edge_count(nn_app_model(app)) == 1);
    assert(nn_app_redo(app, error, 512));
    assert(nn_model_edge_count(nn_app_model(app)) == 0);
    assert(nn_app_undo(app, error, 512));

    add(app, "linear", "core.linear", "", error);
    assert(nn_app_rename_node(app, "linear", "Renamed", error, 512));
    assert(!strcmp(nn_model_find_node(nn_app_model(app), "linear")->label, "Renamed"));
    assert(nn_app_undo(app, error, 512));
    assert(strcmp(nn_model_find_node(nn_app_model(app), "linear")->label, "Renamed"));
    assert(nn_app_set_parameter_text(app, "linear", "bias", "false", error, 512));
    assert(nn_app_undo(app, error, 512));
    char *bias = nn_app_parameter_text(app, "linear", "bias");
    assert(bias && !strcmp(bias, "true"));
    nn_app_free_text(bias);

    assert(nn_app_save(app, error, 512));
    assert(!nn_project_dirty(project));
    assert(nn_app_move_node(app, "layer", 200, 120, error, 512));
    assert(nn_project_dirty(project));
    assert(nn_app_undo(app, error, 512));
    assert(!nn_project_dirty(project)); /* The saved revision is restored. */
    assert(nn_app_redo(app, error, 512));
    assert(nn_project_dirty(project));

    /* No-op and failed commands leave the redo branch intact. */
    assert(nn_app_undo(app, error, 512));
    assert(nn_app_can_redo(app));
    assert(nn_app_move_node(app, "layer", 40, 80, error, 512));
    assert(nn_app_can_redo(app));
    assert(!nn_app_remove_node(app, "missing", error, 512));
    assert(nn_app_can_redo(app));
    assert(nn_app_redo(app, error, 512));
    assert(nn_model_find_node(nn_app_model(app), "layer")->x == 200);

    assert(nn_app_begin_edit(app, error, 512));
    add(app, "group-a", "core.relu", "", error);
    add(app, "group-b", "core.relu", "", error);
    assert(!nn_app_can_undo(app));
    assert(nn_app_end_edit(app, true, error, 512));
    assert(nn_app_can_undo(app));
    assert(nn_app_undo(app, error, 512));
    assert(!nn_model_find_node(nn_app_model(app), "group-a"));
    assert(!nn_model_find_node(nn_app_model(app), "group-b"));

    assert(nn_app_begin_edit(app, error, 512));
    add(app, "rolled-back", "core.relu", "", error);
    assert(nn_app_end_edit(app, false, error, 512));
    assert(!nn_model_find_node(nn_app_model(app), "rolled-back"));

    assert(nn_app_begin_edit(app, error, 512));
    add(app, "group-failure", "core.relu", "", error);
    assert(!nn_app_add_node(app, "invalid", "missing.package", "0.1.0", "",
                            0, 0, error, 512));
    assert(!nn_app_end_edit(app, true, error, 512));
    assert(!nn_model_find_node(nn_app_model(app), "group-failure"));

    assert(nn_app_add_node(app, "sub", "core.repeat", "0.1.0", "",
                           300, 160, error, 512));
    const size_t with_subflow = nn_model_node_count(nn_app_model(app));
    assert(with_subflow >= initial_nodes + 5); /* owner, input, boundary terminal */
    assert(nn_app_undo(app, error, 512));
    assert(!nn_model_find_node(nn_app_model(app), "sub"));
    assert(!nn_model_find_node(nn_app_model(app), "sub-input"));
    assert(!nn_model_find_node(nn_app_model(app), "sub-boundary-out"));
    assert(nn_app_redo(app, error, 512));
    assert(nn_model_node_count(nn_app_model(app)) == with_subflow);

    /* The oldest of 101 edits falls out of the bounded undo history. */
    for (int i = 1; i <= 101; ++i)
        assert(nn_app_move_node(app, "layer", 200 + i * 20, 120, error, 512));
    for (int i = 0; i < 100; ++i) assert(nn_app_undo(app, error, 512));
    assert(nn_model_find_node(nn_app_model(app), "layer")->x == 220);
    assert(!nn_app_can_undo(app));

    assert(nn_app_close(app, true, error, 512));
    assert(!nn_app_can_undo(app) && !nn_app_can_redo(app));
}

int main(void)
{
    char cwd[4096];
    assert(getcwd(cwd, sizeof(cwd)));
    char *core = path_join(cwd, "stereotype-packages/core");
    char temporary[] = "/tmp/nn-history-test-XXXXXX";
    char *root = mkdtemp(temporary);
    assert(root);

    test_model_copy_and_swap();
    NNApplication *app = nn_app_new(core);
    assert(app);
    char error[512] = "";
    assert(nn_app_create(app, root, "history", "History", false,
                         error, sizeof(error)));
    test_commands_groups_and_revisions(app, root, error);

    /* Successful resource creation is a saved history barrier. */
    char *project_dir = path_join(root, "history");
    assert(nn_app_open(app, project_dir, error, sizeof(error)));
    free(project_dir);
    assert(nn_app_add_node(app, "before-resource", "core.relu", "0.1.0", "",
                           80, 80, error, sizeof(error)));
    assert(!nn_app_create_dataset(app, "local.bad", "0.1.0", "{}", false,
                                 error, sizeof(error)));
    assert(nn_app_can_undo(app) && nn_project_dirty(nn_app_project(app)));
    const char *dataset = "{\"name\":\"History data\",\"batch\":{"
        "\"inputs\":{\"image\":{\"dtype\":\"float32\",\"shape\":[\"B\",8]}},"
        "\"targets\":{}}}";
    assert(nn_app_create_dataset(app, "local.history", "0.1.0", dataset, false,
                                 error, sizeof(error)));
    assert(!nn_app_can_undo(app) && !nn_app_can_redo(app));
    assert(!nn_project_dirty(nn_app_project(app)));
    nn_app_free(app);
    free(core);
    puts("graph history invariants: ok");
    return 0;
}
