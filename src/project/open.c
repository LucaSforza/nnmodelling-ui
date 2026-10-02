#define _XOPEN_SOURCE 700
#define _POSIX_C_SOURCE 200809L
#include "project_internal.h"
#include "utils/utils.h"

#include <stdlib.h>
#include <string.h>

static bool parse_references(yyjson_val *array, NNResourceRef **references,
                             size_t *count, char *error, size_t capacity)
{
    if (!yyjson_is_arr(array) || yyjson_arr_size(array) > 128) {
        nn_errorf(error, capacity, "invalid resource reference list"); return false;
    }
    size_t total = yyjson_arr_size(array);
    *references = calloc(total ? total : 1, sizeof(**references));
    if (!*references) return nn_fail(error, capacity, "out of memory loading resource references");
    *count = total;
    for (size_t i = 0; i < *count; ++i) {
        yyjson_val *entry = yyjson_arr_get(array, i);
        const char *id = nn_project_string_field(entry, "id");
        const char *version = nn_project_string_field(entry, "version");
        const char *path = nn_project_string_field(entry, "path");
        if (!nn_project_valid_id(id) || !nn_project_valid_id(version) || !path) {
            nn_errorf(error, capacity, "invalid resource reference"); return false;
        }
        (*references)[i] = (NNResourceRef){nn_text_copy(id), nn_text_copy(version), nn_text_copy(path)};
        if (!(*references)[i].id || !(*references)[i].version || !(*references)[i].path)
            return nn_fail(error, capacity, "out of memory loading resource reference");
    }
    return true;
}

NNProject *nn_project_open(const char *directory, const char *core_root,
                           char *error, size_t capacity)
{
    if (error && capacity) error[0] = '\0';
    if (!directory || !core_root) { nn_errorf(error, capacity, "project path missing"); return NULL; }
    char *resolved = realpath(directory, NULL);
    if (!resolved) { nn_errorf(error, capacity, "project directory unavailable: %s", directory); return NULL; }
    char *model_path = nn_path_join(resolved, "model.json");
    yyjson_doc *document = model_path ? nn_project_read_document(model_path, error, capacity) : NULL;
    if (!model_path) nn_error_set(error, capacity, "unable to build model path");
    free(model_path);
    if (!document) { free(resolved); return NULL; }
    NNProject *project = calloc(1, sizeof(*project));
    if (!project) { yyjson_doc_free(document); free(resolved); nn_error_set(error, capacity, "out of memory opening project"); return NULL; }
    project->directory = resolved;
    project->core_root = realpath(core_root, NULL);
    yyjson_val *root = yyjson_doc_get_root(document);
    yyjson_val *manifest = yyjson_obj_get(root, "manifest");
    const char *id = nn_project_string_field(manifest, "id");
    const char *version = nn_project_string_field(manifest, "version");
    const char *name = nn_project_string_field(manifest, "name");
    const char *description = nn_project_string_field(manifest, "description");
    bool okay = yyjson_get_int(yyjson_obj_get(manifest, "schemaVersion")) == 2 &&
                nn_project_valid_id(id) && nn_project_valid_id(version) && name && *name;
    if (!okay) nn_error_set(error, capacity, "invalid schema-v2 project manifest");
    if (okay) {
        project->id = nn_text_copy(id);
        project->version = nn_text_copy(version);
        project->name = nn_text_copy(name);
        project->description = nn_text_copy(description ? description : "");
        okay = project->id && project->version && project->name && project->description;
        if (!okay) nn_error_set(error, capacity, "out of memory opening project manifest");
    }
    const char *layout = nn_project_string_field(root, "layoutDirection");
    project->layout_direction = nn_text_copy(layout ? layout : "horizontal");
    if (okay && !project->layout_direction) {
        okay = nn_fail(error, capacity, "out of memory copying project layout");
    }
    if (okay && !project->core_root) {
        okay = nn_fail(error, capacity, "core package directory unavailable");
    }
    if (okay) okay = parse_references(yyjson_obj_get(manifest, "customPackages"),
                                      &project->packages, &project->package_count, error, capacity);
    if (okay) project->catalog = nn_catalog_load(core_root, project->directory,
                                                 project->packages, project->package_count,
                                                 error, capacity);
    if (!project->catalog) okay = false;
    NNResourceRef *dataset_refs = NULL;
    size_t dataset_count = 0;
    if (okay) okay = parse_references(yyjson_obj_get(manifest, "customDatasets"),
                                      &dataset_refs, &dataset_count, error, capacity);
    if (okay) {
        project->datasets = calloc(dataset_count ? dataset_count : 1, sizeof(*project->datasets));
        okay = project->datasets != NULL;
        if (okay) project->dataset_count = dataset_count;
        else nn_error_set(error, capacity, "out of memory loading datasets");
    }
    for (size_t i = 0; okay && i < dataset_count; ++i) {
        NNDataset *dataset = &project->datasets[i];
        dataset->id = nn_text_copy(dataset_refs[i].id);
        dataset->version = nn_text_copy(dataset_refs[i].version);
        dataset->path = nn_text_copy(dataset_refs[i].path);
        okay = dataset->id && dataset->version && dataset->path &&
               nn_project_load_dataset(project, dataset, error, capacity);
        for (size_t j = 0; okay && j < i; ++j)
            if (!strcmp(dataset->id, project->datasets[j].id) &&
                !strcmp(dataset->version, project->datasets[j].version)) {
                nn_errorf(error, capacity, "duplicate dataset identity"); okay = false;
            }
    }
    for (size_t i = 0; i < dataset_count; ++i) {
        free((char *)dataset_refs[i].id);
        free((char *)dataset_refs[i].version);
        free((char *)dataset_refs[i].path);
    }
    free(dataset_refs);
    yyjson_val *active = yyjson_obj_get(manifest, "activeDataset");
    if (okay && active) {
        const char *selected_id = nn_project_string_field(active, "id");
        const char *selected_version = nn_project_string_field(active, "version");
        bool found = false;
        for (size_t i = 0; selected_id && selected_version && i < project->dataset_count; ++i)
            if (!strcmp(selected_id, project->datasets[i].id) &&
                !strcmp(selected_version, project->datasets[i].version)) found = true;
        if (!found) okay = nn_fail(error, capacity, "active dataset is undeclared");
        else {
            project->active_dataset_id = nn_text_copy(selected_id);
            project->active_dataset_version = nn_text_copy(selected_version);
            okay = project->active_dataset_id && project->active_dataset_version;
            if (!okay) nn_error_set(error, capacity, "out of memory selecting project dataset");
        }
    }
    if (okay) okay = nn_project_parse_graph(project, root, error, capacity);
    yyjson_doc_free(document);
    if (!okay) {
        if (error && capacity && !error[0]) nn_error_set(error, capacity, "unable to load project resources");
        nn_project_close(project); return NULL;
    }
    return project;
}
