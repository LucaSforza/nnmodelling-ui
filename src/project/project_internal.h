#ifndef NN_PROJECT_INTERNAL_H
#define NN_PROJECT_INTERNAL_H

#include "project/project.h"
#include "yyjson.h"

struct NNProject {
    char *directory;
    char *core_root;
    char *id;
    char *version;
    char *name;
    char *description;
    char *active_dataset_id;
    char *active_dataset_version;
    char *layout_direction;
    NNResourceRef *packages;
    size_t package_count;
    NNDataset *datasets;
    size_t dataset_count;
    NNModel *model;
    NNCatalog *catalog;
    bool dirty;
};

const char *nn_project_string_field(yyjson_val *object, const char *key);
bool nn_project_valid_id(const char *id);
bool nn_project_valid_semver(const char *text);
char *nn_project_safe_resource_dir(const char *root, const char *relative,
                                   char *error, size_t capacity);
yyjson_doc *nn_project_read_document(const char *path, char *error, size_t capacity);
bool nn_project_write_all(int fd, const char *data, size_t length);
void nn_project_remove_tree(const char *path);
bool nn_project_load_dataset(NNProject *project, NNDataset *dataset,
                             char *error, size_t capacity);
void nn_project_dataset_dispose(NNDataset *dataset);
bool nn_project_parse_value(yyjson_val *source, NNValue *target);
bool nn_project_edge_topology_valid(const NNModel *model, const NNCatalog *catalog,
                                    const NNEdge *edge, char *error, size_t capacity);
bool nn_project_parse_graph(NNProject *project, yyjson_val *root,
                            char *error, size_t capacity);
bool nn_project_write_project_document(NNProject *project, char **json, size_t *length);

#endif
