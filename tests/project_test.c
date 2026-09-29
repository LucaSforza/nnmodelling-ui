#define _XOPEN_SOURCE 700
#define _POSIX_C_SOURCE 200809L
#include "../src/project.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

/* The project boundary is tested independently of package parsing. */
struct NNCatalog { int present; };
NNCatalog *nn_catalog_load(const char *core_root, const char *project_root,
                           const NNResourceRef *custom, size_t custom_count,
                           char *err, size_t cap)
{
    (void)core_root; (void)project_root; (void)custom; (void)custom_count;
    (void)err; (void)cap;
    return calloc(1, sizeof(NNCatalog));
}
void nn_catalog_free(NNCatalog *catalog) { free(catalog); }
size_t nn_catalog_count(const NNCatalog *catalog) { return catalog ? 1 : 0; }
const NNPackage *nn_catalog_at(const NNCatalog *catalog, size_t index)
{
    static const NNPackage package = { .id = "core.layer", .version = "1.0" };
    return catalog && index == 0 ? &package : NULL;
}
const NNPackage *nn_catalog_find(const NNCatalog *catalog, const char *id, const char *version)
{
    (void)id; (void)version;
    return catalog ? nn_catalog_at(catalog, 0) : NULL;
}

static void write_file(const char *path, const char *content)
{
    FILE *file = fopen(path, "wb");
    assert(file);
    size_t length = strlen(content);
    assert(fwrite(content, 1, length, file) == length);
    assert(fclose(file) == 0);
}

static char *path_join(const char *left, const char *right)
{
    size_t size = strlen(left) + strlen(right) + 2;
    char *path = malloc(size);
    assert(path);
    snprintf(path, size, "%s/%s", left, right);
    return path;
}

int main(void)
{
    char temporary[] = "/tmp/nn-project-test-XXXXXX";
    char *root = mkdtemp(temporary);
    assert(root);
    char *core = path_join(root, "core");
    assert(mkdir(core, 0755) == 0);
    char *parent = path_join(root, "created");
    assert(mkdir(parent, 0755) == 0);
    char error[256];

    NNProject *blank = nn_project_create(parent, "blank", "Blank project", false,
                                         core, error, sizeof(error));
    assert(blank && !nn_project_dirty(blank));
    assert(!strcmp(nn_project_id(blank), "blank"));
    assert(nn_model_node_count(nn_project_model(blank)) == 0);
    nn_project_close(blank);

    char *fixture = path_join(root, "fixture");
    assert(mkdir(fixture, 0755) == 0);
    char *datasets = path_join(fixture, "datasets");
    char *dataset = path_join(datasets, "toy");
    assert(mkdir(datasets, 0755) == 0 && mkdir(dataset, 0755) == 0);
    char *dataset_manifest = path_join(dataset, "manifest.json");
    char *dataset_definition = path_join(dataset, "dataset.json");
    write_file(dataset_manifest,
        "{\"schemaVersion\":1,\"id\":\"toy\",\"version\":\"1.0\","
        "\"entrypoints\":{\"definition\":\"dataset.json\"}}");
    write_file(dataset_definition,
        "{\"name\":\"Toy data\",\"batch\":{"
        "\"inputs\":{\"features\":{\"dtype\":\"float32\",\"shape\":[\"B\",4]}},"
        "\"targets\":{\"labels\":{\"dtype\":\"int64\",\"shape\":[\"B\"]}}}}");
    char *model = path_join(fixture, "model.json");
    write_file(model,
        "{\"layoutDirection\":\"vertical\",\"nodes\":["
        "{\"id\":\"n1\",\"type\":\"custom\",\"position\":{\"x\":1,\"y\":2},"
        "\"data\":{\"package\":{\"id\":\"core.layer\",\"version\":\"1.0\"},"
        "\"scope\":\"\",\"name\":\"one\",\"params\":{\"config\":[true,7,\"x\"]}}},"
        "{\"id\":\"n2\",\"type\":\"custom\",\"position\":{\"x\":3,\"y\":4},"
        "\"data\":{\"package\":{\"id\":\"core.layer\",\"version\":\"1.0\"},"
        "\"scope\":\"\",\"name\":\"two\",\"params\":{}}}],"
        "\"edges\":[{\"id\":\"e1\",\"source\":\"n1\",\"sourceHandle\":\"out\","
        "\"target\":\"n2\",\"targetHandle\":\"in\"}],"
        "\"manifest\":{\"schemaVersion\":2,\"id\":\"fixture\",\"version\":\"1.0\","
        "\"name\":\"Fixture\",\"description\":\"kept\",\"customPackages\":[],"
        "\"customDatasets\":[{\"id\":\"toy\",\"version\":\"1.0\",\"path\":\"datasets/toy\"}],"
        "\"activeDataset\":{\"id\":\"toy\",\"version\":\"1.0\"}}}");

    NNProject *project = nn_project_open(fixture, core, error, sizeof(error));
    assert(project);
    assert(nn_project_dataset_count(project) == 1);
    assert(!strcmp(nn_project_active_dataset(project)->name, "Toy data"));
    assert(nn_model_node_count(nn_project_model(project)) == 2);
    assert(nn_model_edge_count(nn_project_model(project)) == 1);
    const NNValue *config = &nn_model_find_node(nn_project_model(project), "n1")->parameters[0].value;
    assert(config->type == NN_VALUE_ARRAY && config->as.array.count == 3);

    NNValue update = { .type = NN_VALUE_STRING, .as.string = "saved" };
    assert(nn_model_set_parameter(nn_project_model(project), "n1", "status", &update,
                                  error, sizeof(error)));
    assert(nn_model_move_node(nn_project_model(project), "n2", 33, 44, error, sizeof(error)));
    nn_project_mark_dirty(project);
    assert(nn_project_dirty(project));
    assert(nn_project_save(project, error, sizeof(error)));
    assert(!nn_project_dirty(project));

    char *invalid = path_join(root, "invalid");
    assert(mkdir(invalid, 0755) == 0);
    char *invalid_model = path_join(invalid, "model.json");
    write_file(invalid_model, "{bad json");
    assert(nn_project_open(invalid, core, error, sizeof(error)) == NULL);
    assert(!strcmp(nn_project_id(project), "fixture"));
    assert(nn_model_node_count(nn_project_model(project)) == 2);

    nn_project_close(project);
    project = nn_project_open(fixture, core, error, sizeof(error));
    assert(project && nn_project_dataset_count(project) == 1);
    assert(nn_project_active_dataset(project));
    assert(nn_model_node_count(nn_project_model(project)) == 2);
    assert(nn_model_edge_count(nn_project_model(project)) == 1);
    const NNNode *saved_node = nn_model_find_node(nn_project_model(project), "n2");
    assert(saved_node && saved_node->x == 33 && saved_node->y == 44);
    config = &nn_model_find_node(nn_project_model(project), "n1")->parameters[1].value;
    assert(config->type == NN_VALUE_STRING && !strcmp(config->as.string, "saved"));
    nn_project_close(project);

    char cwd[4096];
    assert(getcwd(cwd, sizeof(cwd)));
    char *repo_core = path_join(cwd, "stereotype-packages/core");
    NNProject *mnist = nn_project_create(parent, "editable-mnist", "Editable MNIST", true,
                                         repo_core, error, sizeof(error));
    assert(mnist);
    assert(!strcmp(nn_project_id(mnist), "editable-mnist"));
    assert(!strcmp(nn_project_name(mnist), "Editable MNIST"));
    assert(nn_project_dataset_count(mnist) == 1 && nn_project_active_dataset(mnist));
    assert(nn_model_node_count(nn_project_model(mnist)) == 8);
    assert(nn_model_edge_count(nn_project_model(mnist)) == 7);
    nn_project_close(mnist);

    free(repo_core); free(invalid_model); free(invalid); free(model);
    free(dataset_definition); free(dataset_manifest); free(dataset); free(datasets);
    free(fixture); free(parent); free(core);
    puts("project persistence: ok");
    return 0;
}
