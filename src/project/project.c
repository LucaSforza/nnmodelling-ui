#include "project_internal.h"

#include <stdlib.h>
#include <string.h>

void nn_project_close(NNProject *project)
{
    if (!project) return;
    nn_model_free(project->model);
    nn_catalog_free(project->catalog);
    for (size_t i = 0; project->packages && i < project->package_count; ++i) {
        free((char *)project->packages[i].id);
        free((char *)project->packages[i].version);
        free((char *)project->packages[i].path);
    }
    for (size_t i = 0; project->datasets && i < project->dataset_count; ++i) nn_project_dataset_dispose(&project->datasets[i]);
    free(project->packages); free(project->datasets);
    free(project->directory); free(project->core_root); free(project->id); free(project->version);
    free(project->name); free(project->description); free(project->layout_direction);
    free(project->active_dataset_id); free(project->active_dataset_version);
    free(project);
}

void nn_project_mark_dirty(NNProject *project) { if (project) project->dirty = true; }

const char *nn_project_directory(const NNProject *project) { return project ? project->directory : NULL; }

const char *nn_project_id(const NNProject *project) { return project ? project->id : NULL; }

const char *nn_project_version(const NNProject *project) { return project ? project->version : NULL; }

const char *nn_project_name(const NNProject *project) { return project ? project->name : NULL; }

bool nn_project_dirty(const NNProject *project) { return project && project->dirty; }

NNModel *nn_project_model(NNProject *project) { return project ? project->model : NULL; }

const NNCatalog *nn_project_catalog(const NNProject *project) { return project ? project->catalog : NULL; }

size_t nn_project_dataset_count(const NNProject *project) { return project ? project->dataset_count : 0; }

const NNDataset *nn_project_dataset_at(const NNProject *project, size_t index)
{
    return project && index < project->dataset_count ? &project->datasets[index] : NULL;
}

const NNDataset *nn_project_active_dataset(const NNProject *project)
{
    if (!project || !project->active_dataset_id) return NULL;
    for (size_t i = 0; i < project->dataset_count; ++i)
        if (!strcmp(project->datasets[i].id, project->active_dataset_id) &&
            !strcmp(project->datasets[i].version, project->active_dataset_version))
            return &project->datasets[i];
    return NULL;
}
