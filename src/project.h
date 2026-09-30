#ifndef NN_PROJECT_H
#define NN_PROJECT_H

#include <stdbool.h>
#include <stddef.h>

#include "catalog.h"
#include "model.h"

typedef struct NNProject NNProject;

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    char *name;
    char *dtype;
    NNValue shape;
} NNTensorSlot;

typedef struct {
    char *id;
    char *version;
    char *path;
    char *name;
    NNTensorSlot *inputs;
    size_t input_count;
    NNTensorSlot *targets;
    size_t target_count;
} NNDataset;

/* Returned project owns graph, catalogs, strings and resource records. */
NNProject *nn_project_open(const char *directory, const char *core_root,
                           char *error, size_t error_capacity);
NNProject *nn_project_create(const char *parent, const char *id, const char *name,
                             bool mnist_template, const char *core_root,
                             char *error, size_t error_capacity);
bool nn_project_save(NNProject *project, char *error, size_t error_capacity);
void nn_project_close(NNProject *project);
void nn_project_mark_dirty(NNProject *project);

const char *nn_project_directory(const NNProject *project);
const char *nn_project_id(const NNProject *project);
const char *nn_project_version(const NNProject *project);
const char *nn_project_name(const NNProject *project);
bool nn_project_dirty(const NNProject *project);
NNModel *nn_project_model(NNProject *project);
const NNCatalog *nn_project_catalog(const NNProject *project);
size_t nn_project_dataset_count(const NNProject *project);
const NNDataset *nn_project_dataset_at(const NNProject *project, size_t index);
const NNDataset *nn_project_active_dataset(const NNProject *project);
bool nn_project_create_stereotype(NNProject *project, const char *id, const char *version,
                                  const char *definition_json, const char *inference_lua,
                                  const char *dependencies_json, char *error, size_t capacity);
bool nn_project_create_dataset(NNProject *project, const char *id, const char *version,
                               const char *definition_json, bool select,
                               char *error, size_t capacity);
bool nn_project_select_dataset(NNProject *project, const char *id, const char *version,
                               char *error, size_t capacity);
bool nn_project_create_vae(const char *parent, const char *id, const char *name,
                           const char *core_root, NNProject **result,
                           char *error, size_t capacity);

#ifdef __cplusplus
}
#endif

#endif
