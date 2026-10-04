#define _XOPEN_SOURCE 700

#include "application/application.h"
#include "catalog/catalog.h"
#include "model/model.h"
#include "project/project.h"
#include "visualization/visualization.h"

#include <assert.h>
#include <dirent.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static const NNNode *find_package_in_scope(const NNModel *model, const char *scope,
                                           const char *package)
{
    for (size_t i = 0; i < nn_model_node_count(model); ++i) {
        const NNNode *node = nn_model_node_at(model, i);
        if (node && !strcmp(node->scope_id ? node->scope_id : "", scope) &&
            !strcmp(node->package_id, package)) return node;
    }
    return NULL;
}

static void path(char *buffer, size_t capacity, const char *root, const char *relative)
{
    assert(snprintf(buffer, capacity, "%s/%s", root, relative) < (int)capacity);
}

static void write_text(const char *filename, const char *text)
{
    FILE *file = fopen(filename, "wb");
    assert(file);
    const size_t length = strlen(text);
    assert(fwrite(text, 1, length, file) == length);
    assert(fclose(file) == 0);
}

static void remove_tree(const char *root)
{
    struct stat info;
    if (lstat(root, &info) != 0) return;
    if (!S_ISDIR(info.st_mode)) {
        assert(unlink(root) == 0);
        return;
    }
    DIR *directory = opendir(root);
    assert(directory);
    struct dirent *entry;
    while ((entry = readdir(directory))) {
        if (!strcmp(entry->d_name, ".") || !strcmp(entry->d_name, "..")) continue;
        char child[4096];
        path(child, sizeof(child), root, entry->d_name);
        remove_tree(child);
    }
    assert(closedir(directory) == 0);
    assert(rmdir(root) == 0);
}

static void assert_fit_visible(const NN3DScene *scene, double width, double height)
{
    NN3DCamera camera = {0};
    nn_3d_camera_fit(scene, &camera, width / height);
    NN3DFrame frame = {0};
    assert(nn_3d_frame(scene, &camera, width, height, &frame));
    for (size_t n = 0; n < nn_3d_node_count(scene); ++n) {
        bool visible = false;
        for (size_t i = 0; i < frame.count; ++i) {
            const NN3DPrimitive *item = &frame.items[i];
            if (item->kind != NN_3D_FACE || item->node != n) continue;
            visible = true;
            for (size_t p = 0; p < 4; ++p) {
                assert(item->x[p] >= -1.0 && item->x[p] <= width + 1.0);
                assert(item->y[p] >= -1.0 && item->y[p] <= height + 1.0);
            }
        }
        assert(visible);
    }
    nn_3d_frame_dispose(&frame);
}

static void create_subflow(NNApplication *app, const char *id, const char *name,
                           const char *dependencies, const char *outputs,
                           char *error, size_t capacity)
{
    char definition[1024];
    assert(snprintf(definition, sizeof(definition),
        "{\"name\":\"%s\",\"kind\":\"subflow\","
        "\"view\":{\"color\":\"#4779c4\",\"width\":180,\"height\":100},"
        "\"parameters\":{},%s}", name, outputs) < (int)sizeof(definition));
    const char *rule = "return function(context,parameters,services) "
                       "return {status='success',output=context.inputs[1]} end";
    assert(nn_app_create_stereotype(app, id, "0.1.0", definition, rule,
                                    dependencies, error, capacity));
}

static void install_visualization(const char *project_dir, const char *id,
                                  const char *dependencies, const char *lua)
{
    char relative[256], manifest_path[4096], lua_path[4096], manifest[1024];
    assert(snprintf(relative, sizeof(relative), "packages/%s-0.1.0", id) < (int)sizeof(relative));
    char package_dir[4096];
    path(package_dir, sizeof(package_dir), project_dir, relative);
    assert(snprintf(manifest_path, sizeof(manifest_path), "%s/manifest.json", package_dir) < (int)sizeof(manifest_path));
    assert(snprintf(lua_path, sizeof(lua_path), "%s/visualization.lua", package_dir) < (int)sizeof(lua_path));
    assert(snprintf(manifest, sizeof(manifest),
        "{\"schemaVersion\":1,\"id\":\"%s\",\"version\":\"0.1.0\","
        "\"dependencies\":%s,\"entrypoints\":{"
        "\"definition\":\"definition.json\","
        "\"inference\":{\"language\":\"lua\",\"file\":\"inference.lua\"},"
        "\"visualization\":{\"language\":\"lua\",\"file\":\"visualization.lua\"}}}\n",
        id, dependencies) < (int)sizeof(manifest));
    write_text(manifest_path, manifest);
    write_text(lua_path, lua);
}

static void test_repeat_fixture_and_camera(void)
{
    char error[512] = {0};
    NNProject *project = nn_project_open("examples/tiny-decoder-llm",
        "stereotype-packages/core", error, sizeof(error));
    assert(project && error[0] == '\0');
    const NNModel *model = nn_project_model(project);
    NNModel *before = nn_model_copy(model);
    assert(before);
    const NNNode *repeat = NULL;
    for (size_t i = 0; i < nn_model_node_count(model); ++i) {
        const NNNode *node = nn_model_node_at(model, i);
        if (node && !strcmp(node->package_id, "core.repeat")) { repeat = node; break; }
    }
    assert(repeat);
    const NNNode *child = NULL;
    for (size_t i = 0; i < nn_model_node_count(model); ++i) {
        const NNNode *node = nn_model_node_at(model, i);
        if (node && !strcmp(node->scope_id ? node->scope_id : "", repeat->id) &&
            strcmp(node->package_id, "core.input") && strcmp(node->package_id, "core.output")) {
            child = node;
            break;
        }
    }
    assert(child);
    NN3DScene *scene = nn_3d_build(project, error, sizeof(error));
    if (!scene) fprintf(stderr, "repeat fixture 3D build: %s\n", error);
    assert(scene);
    size_t copies = 0;
    for (size_t i = 0; i < nn_3d_node_count(scene); ++i) {
        const NN3DNode *node = nn_3d_node_at(scene, i);
        assert(node && node->path && node->source_id && node->label && node->package_id);
        if (!strcmp(node->source_id, child->id)) ++copies;
        for (size_t j = 0; j < i; ++j)
            assert(strcmp(node->path, nn_3d_node_at(scene, j)->path));
    }
    assert(copies == 6);
    assert(nn_3d_group_count(scene) >= 6);
    assert_fit_visible(scene, 1280, 720);
    assert_fit_visible(scene, 420, 900);
    bool loss_edge = false;
    for (size_t i = 0; i < nn_3d_edge_count(scene); ++i) {
        const NN3DEdge *edge = nn_3d_edge_at(scene, i);
        if (edge && edge->loss) loss_edge = true;
    }
    assert(loss_edge);

    NN3DCamera camera = {0};
    nn_3d_camera_fit(scene, &camera, 16.0 / 9.0);
    NN3DFrame frame = {0};
    assert(nn_3d_frame(scene, &camera, 1280, 720, &frame) && frame.count);
    bool picked = false;
    for (size_t i = 0; i < frame.count && !picked; ++i) {
        const NN3DPrimitive *primitive = &frame.items[i];
        if (primitive->kind != NN_3D_FACE || primitive->node == SIZE_MAX) continue;
        const double x = (primitive->x[0] + primitive->x[2]) / 2.0;
        const double y = (primitive->y[0] + primitive->y[2]) / 2.0;
        picked = nn_3d_pick(&frame, x, y) == primitive->node;
    }
    assert(picked);
    nn_3d_frame_dispose(&frame);
    const NN3DVec fitted = camera.position;
    nn_3d_camera_home(scene, &camera, 16.0 / 9.0);
    assert(nn_3d_frame(scene, &camera, 1280, 720, &frame) && frame.count);
    nn_3d_frame_dispose(&frame);
    nn_3d_camera_look(&camera, 0.2, -0.1);
    nn_3d_camera_move(&camera, 0.1, 0.0, 0.2);
    assert(camera.position.x != fitted.x || camera.position.y != fitted.y || camera.position.z != fitted.z);
    assert(nn_model_equal(before, nn_project_model(project)));
    assert(!nn_project_dirty(project));
    nn_3d_free(scene);
    nn_model_free(before);
    nn_project_close(project);
}

static void test_empty_scene_camera(void)
{
    char temporary[] = "/tmp/nn-3d-empty-XXXXXX";
    char *parent = mkdtemp(temporary);
    assert(parent);
    NNApplication *app = nn_app_new("stereotype-packages/core");
    assert(app);
    char error[512] = {0};
    assert(nn_app_create(app, parent, "empty", "Empty", false, error, sizeof(error)));
    NN3DScene *scene = nn_3d_build(nn_app_project(app), error, sizeof(error));
    if (!scene) fprintf(stderr, "empty project 3D build: %s\n", error);
    assert(scene);
    NN3DCamera camera = {0};
    nn_3d_camera_fit(scene, &camera, 1.5);
    assert(isfinite(camera.position.x) && isfinite(camera.position.y) && isfinite(camera.position.z));
    assert(isfinite(camera.yaw) && isfinite(camera.pitch));
    NN3DFrame frame = {0};
    assert(nn_3d_frame(scene, &camera, 900, 600, &frame));
    nn_3d_frame_dispose(&frame);
    nn_3d_camera_home(scene, &camera, 1.5);
    assert(isfinite(camera.position.x) && isfinite(camera.position.y) && isfinite(camera.position.z));
    nn_3d_free(scene);
    nn_app_free(app);
    remove_tree(parent);
}

static void test_nested_subflows_and_custom_recipe(void)
{
    char temporary[] = "/tmp/nn-3d-test-XXXXXX";
    char *parent = mkdtemp(temporary);
    assert(parent);
    NNApplication *app = nn_app_new("stereotype-packages/core");
    assert(app);
    char error[512] = {0};
    assert(nn_app_create(app, parent, "composition", "Composition", false, error, sizeof(error)));
    create_subflow(app, "local.inner", "Inner", "{}", "\"outputs\":[{\"id\":\"out\",\"type\":\"output\"}]", error, sizeof(error));
    const char *recipe_dependencies = "{\"core.relu\":\"0.1.0\",\"core.cross-entropy\":\"0.1.0\"}";
    create_subflow(app, "local.recipe", "Recipe", recipe_dependencies,
        "\"outputs\":[{\"id\":\"out\",\"type\":\"output\"},{\"id\":\"loss\",\"type\":\"loss\"}]",
        error, sizeof(error));
    create_subflow(app, "local.invalid", "Invalid recipe", "{}",
        "\"outputs\":[{\"id\":\"out\",\"type\":\"output\"}]", error, sizeof(error));
    create_subflow(app, "local.invalid-hash", "Invalid hashed recipe", "{}",
        "\"outputs\":[{\"id\":\"out\",\"type\":\"output\"}]", error, sizeof(error));
    create_subflow(app, "local.invalid-endpoint", "Invalid endpoint recipe", "{}",
        "\"outputs\":[{\"id\":\"out\",\"type\":\"output\"}]", error, sizeof(error));
    const char *recipe =
        "return function(parameters) return {"
        "nodes={{id='body',body=true,label='Nested body'},"
        "{id='gate',package={id='core.relu',version='0.1.0'},parameters={},label='Recipe ReLU'},"
        "{id='objective',package={id='core.cross-entropy',version='0.1.0'},parameters={},label='Objective'}},"
        "edges={{source='$input',sourceHandle='out',target='body',targetHandle='in'},"
        "{source='body',sourceHandle='out',target='gate',targetHandle='in'},"
        "{source='gate',sourceHandle='out',target='objective',targetHandle='in'}},"
        "outputs={out={node='gate',handle='out'},loss={node='objective',handle='loss'}}} end";
    char project_dir[4096];
    assert(snprintf(project_dir, sizeof(project_dir), "%s", nn_project_directory(nn_app_project(app))) < (int)sizeof(project_dir));
    const char *layer_definition =
        "{\"name\":\"Visualization layer\",\"kind\":\"layer\","
        "\"view\":{\"color\":\"#4779c4\",\"width\":180,\"height\":100},\"parameters\":{}}";
    const char *layer_rule = "return function(context,parameters,services) return {status='success'} end";
    assert(nn_app_create_stereotype(app, "local.layer-visual", "0.1.0", layer_definition,
                                    layer_rule, "{}", error, sizeof(error)));
    install_visualization(project_dir, "local.layer-visual", "{}",
        "return function() return {nodes={},edges={},outputs={}} end");
    NNResourceRef layer_resource = {"local.layer-visual", "0.1.0", "packages/local.layer-visual-0.1.0"};
    NNCatalog *layer_catalog = nn_catalog_load("stereotype-packages/core", project_dir,
                                               &layer_resource, 1, error, sizeof(error));
    assert(!layer_catalog && error[0]);
    char layer_package[4096];
    path(layer_package, sizeof(layer_package), project_dir, layer_resource.path);
    char layer_manifest[4096];
    path(layer_manifest, sizeof(layer_manifest), layer_package, "manifest.json");
    write_text(layer_manifest,
        "{\"schemaVersion\":1,\"id\":\"local.layer-visual\",\"version\":\"0.1.0\","
        "\"dependencies\":{},\"entrypoints\":{\"definition\":\"definition.json\","
        "\"inference\":{\"language\":\"lua\",\"file\":\"inference.lua\"}}}\n");

    install_visualization(project_dir, "local.recipe", recipe_dependencies, recipe);
    install_visualization(project_dir, "local.invalid", "{}",
        "return function() return {nodes={{id='bad',body=true}},edges={},outputs={}} end");
    install_visualization(project_dir, "local.invalid-hash", "{}",
        "return function() return {nodes={bad={id='bad',body=true}},edges={},outputs={}} end");
    install_visualization(project_dir, "local.invalid-endpoint", "{}",
        "return function() return {nodes={{id='gate',package={id='core.relu',version='0.1.0'},parameters={}}},"
        "edges={{source='missing',sourceHandle='out',target='gate',targetHandle='in'}},"
        "outputs={out={node='gate',handle='out'}}} end");
    if (!nn_app_open(app, project_dir, error, sizeof(error))) {
        fprintf(stderr, "open custom visualization test project: %s\n", error);
        abort();
    }

    assert(nn_app_add_node(app, "recipe-owner", "local.recipe", "0.1.0", "", 0, 0, error, sizeof(error)));
    const NNModel *model = nn_app_model(app);
    const NNNode *owner_input = find_package_in_scope(model, "recipe-owner", "core.input");
    const NNNode *owner_output = find_package_in_scope(model, "recipe-owner", "core.output");
    const NNNode *owner_loss = find_package_in_scope(model, "recipe-owner", "core.loss-output");
    assert(owner_input && owner_output && owner_output->boundary_handle_id);
    assert(owner_loss && owner_loss->boundary_handle_id);
    char *owner_input_id = strdup(owner_input->id);
    char *owner_output_id = strdup(owner_output->id);
    assert(owner_input_id && owner_output_id);
    assert(nn_app_add_node(app, "nested-owner", "local.inner", "0.1.0", "recipe-owner", 0, 0,
                           error, sizeof(error)));
    model = nn_app_model(app);
    const NNNode *nested_input = find_package_in_scope(model, "nested-owner", "core.input");
    const NNNode *nested_output = find_package_in_scope(model, "nested-owner", "core.output");
    assert(nested_input && nested_output && nested_output->boundary_handle_id);
    char *nested_input_id = strdup(nested_input->id);
    char *nested_output_id = strdup(nested_output->id);
    assert(nested_input_id && nested_output_id);
    assert(nn_app_add_node(app, "deep-relu", "core.relu", "0.1.0", "nested-owner", 0, 0,
                           error, sizeof(error)));
    assert(nn_app_connect(app, "deep-in", nested_input_id, "out", "deep-relu", "in", error, sizeof(error)));
    assert(nn_app_connect(app, "deep-out", "deep-relu", "out", nested_output_id, "in", error, sizeof(error)));
    assert(nn_app_connect(app, "owner-in", owner_input_id, "out", "nested-owner", "in", error, sizeof(error)));
    assert(nn_app_connect(app, "owner-out", "nested-owner", "out", owner_output_id, "in", error, sizeof(error)));
    free(owner_input_id);
    free(owner_output_id);
    free(nested_input_id);
    free(nested_output_id);
    assert(nn_app_add_node(app, "parallel", "core.horizontal-repeat", "0.1.0", "", 400, 0,
                           error, sizeof(error)));
    nn_project_set_dirty((NNProject *)nn_app_project(app), false);
    NNModel *before = nn_model_copy(nn_app_model(app));
    assert(before);

    NN3DScene *scene = nn_3d_build(nn_app_project(app), error, sizeof(error));
    if (!scene) fprintf(stderr, "custom composition 3D build: %s\n", error);
    assert(scene);
    size_t nested = 0, synthetic_gates = 0, synthetic_joins = 0, synthetic_objectives = 0;
    for (size_t i = 0; i < nn_3d_node_count(scene); ++i) {
        const NN3DNode *node = nn_3d_node_at(scene, i);
        assert(node && node->path);
        if (!strcmp(node->source_id, "deep-relu")) ++nested;
        if (!strcmp(node->source_id, "recipe-owner") && !strcmp(node->package_id, "core.relu"))
            ++synthetic_gates;
        if (!strcmp(node->source_id, "recipe-owner") && !strcmp(node->package_id, "core.cross-entropy"))
            ++synthetic_objectives;
        if (!strcmp(node->source_id, "parallel") && !strcmp(node->package_id, "core.concat"))
            ++synthetic_joins;
        for (size_t j = 0; j < i; ++j)
            assert(strcmp(node->path, nn_3d_node_at(scene, j)->path));
    }
    assert(nested == 1 && synthetic_gates == 1 && synthetic_objectives == 1 && synthetic_joins == 1);
    size_t recipe_groups = 0, nested_groups = 0, repeat_groups = 0;
    for (size_t i = 0; i < nn_3d_group_count(scene); ++i) {
        const NN3DGroup *group = nn_3d_group_at(scene, i);
        assert(group && group->path && group->source_id);
        if (!strcmp(group->source_id, "recipe-owner")) ++recipe_groups;
        if (!strcmp(group->source_id, "nested-owner")) ++nested_groups;
        if (!strcmp(group->source_id, "parallel")) ++repeat_groups;
    }
    assert(recipe_groups == 2 && nested_groups == 2 && repeat_groups == 3);
    assert(nn_model_equal(before, nn_app_model(app)));
    assert(!nn_project_dirty(nn_app_project(app)));
    nn_3d_free(scene);

    const char *invalid_owners[] = {"local.invalid", "local.invalid-hash", "local.invalid-endpoint"};
    for (size_t i = 0; i < sizeof(invalid_owners) / sizeof(invalid_owners[0]); ++i) {
        char owner_id[32];
        assert(snprintf(owner_id, sizeof(owner_id), "bad-owner-%zu", i) < (int)sizeof(owner_id));
        assert(nn_app_add_node(app, owner_id, invalid_owners[i], "0.1.0", "", 800 + 240 * (int)i, 0,
                               error, sizeof(error)));
        nn_project_set_dirty((NNProject *)nn_app_project(app), false);
        NNModel *with_bad = nn_model_copy(nn_app_model(app));
        assert(with_bad);
        scene = nn_3d_build(nn_app_project(app), error, sizeof(error));
        assert(!scene && error[0]);
        assert(nn_model_equal(with_bad, nn_app_model(app)));
        assert(!nn_project_dirty(nn_app_project(app)));
        nn_model_free(with_bad);
        assert(nn_app_undo(app, error, sizeof(error)));
        assert(nn_model_equal(before, nn_app_model(app)));
        nn_project_set_dirty((NNProject *)nn_app_project(app), false);
    }
    nn_model_free(before);
    nn_app_free(app);
    remove_tree(parent);
}

int main(void)
{
    test_empty_scene_camera();
    test_repeat_fixture_and_camera();
    test_nested_subflows_and_custom_recipe();
    puts("Visualization scenes, occurrences, composition, picking and immutability passed");
    return 0;
}
