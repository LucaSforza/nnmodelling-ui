#define _XOPEN_SOURCE 700
#include "project_internal.h"
#include "utils/utils.h"

#include <stdlib.h>
#include <string.h>

static bool parse_slots(yyjson_val *object, NNTensorSlot **slots, size_t *count,
                        char *error, size_t capacity)
{
    if (!yyjson_is_obj(object)) { nn_errorf(error, capacity, "dataset slots must be an object"); return false; }
    size_t total = yyjson_obj_size(object);
    if (total > 64) { nn_errorf(error, capacity, "too many dataset slots"); return false; }
    *slots = calloc(total ? total : 1, sizeof(**slots));
    if (!*slots) return nn_fail(error, capacity, "out of memory loading dataset slots");
    *count = total;
    size_t object_index, object_max;
    yyjson_val *key, *value;
    size_t index = 0;
    yyjson_obj_foreach(object, object_index, object_max, key, value) {
        NNTensorSlot *slot = &(*slots)[index++];
        const char *dtype = nn_project_string_field(value, "dtype");
        yyjson_val *shape = yyjson_obj_get(value, "shape");
        const char *allowed[] = {"float16","bfloat16","float32","float64","int8","int16","int32","int64","uint8","bool"};
        bool dtype_ok = false;
        for (size_t d = 0; dtype && d < sizeof(allowed) / sizeof(allowed[0]); ++d)
            if (!strcmp(dtype, allowed[d])) dtype_ok = true;
        if (!yyjson_get_str(key) || !*yyjson_get_str(key) || !dtype_ok ||
            !yyjson_is_arr(shape) || !yyjson_arr_size(shape) || yyjson_arr_size(shape) > 64 ||
            !nn_project_parse_value(shape, &slot->shape)) {
            nn_errorf(error, capacity, "invalid dataset tensor slot");
            return false;
        }
        slot->name = nn_text_copy(yyjson_get_str(key));
        slot->dtype = nn_text_copy(dtype);
        if (!slot->name || !slot->dtype) return nn_fail(error, capacity, "out of memory loading dataset slot");
        for (size_t previous = 0; previous + 1 < index; ++previous)
            if (!strcmp((*slots)[previous].name, slot->name)) {
                nn_errorf(error, capacity, "duplicate dataset slot name: %s", slot->name);
                return false;
            }
        for (size_t i = 0; i < slot->shape.as.array.count; ++i) {
            NNValue *dimension = &slot->shape.as.array.items[i];
            if (!((dimension->type == NN_VALUE_INT && dimension->as.integer > 0) ||
                  (dimension->type == NN_VALUE_STRING &&
                   !strcmp(dimension->as.string, "B")))) {
                nn_errorf(error, capacity, "invalid dataset tensor dimension");
                return false;
            }
        }
    }
    return true;
}

bool nn_project_load_dataset(NNProject *project, NNDataset *dataset,
                         char *error, size_t capacity)
{
    char *directory = nn_project_safe_resource_dir(project->directory, dataset->path, error, capacity);
    if (!directory) return false;
    char *manifest_path = nn_path_join(directory, "manifest.json");
    yyjson_doc *manifest = manifest_path ? nn_project_read_document(manifest_path, error, capacity) : NULL;
    if (!manifest_path) nn_error_set(error, capacity, "unable to build dataset manifest path");
    free(manifest_path);
    if (!manifest) { free(directory); return false; }
    yyjson_val *root = yyjson_doc_get_root(manifest);
    yyjson_val *entrypoints = yyjson_obj_get(root, "entrypoints");
    const char *definition = nn_project_string_field(entrypoints, "definition");
    const char *id = nn_project_string_field(root, "id"), *version = nn_project_string_field(root, "version");
    bool okay = yyjson_get_int(yyjson_obj_get(root, "schemaVersion")) == 1 &&
                id && version && !strcmp(id, dataset->id) &&
                !strcmp(version, dataset->version) && definition &&
                nn_project_valid_id(definition) && !strchr(definition, '/');
    if (!okay) nn_errorf(error, capacity, "dataset identity or definition invalid: %s", dataset->path);
    char *definition_path = okay ? nn_path_join(directory, definition) : NULL;
    yyjson_doc *definition_doc = definition_path ? nn_project_read_document(definition_path, error, capacity) : NULL;
    if (okay && !definition_path) nn_error_set(error, capacity, "unable to build dataset definition path");
    free(definition_path);
    free(directory);
    yyjson_doc_free(manifest);
    if (!definition_doc) return false;
    root = yyjson_doc_get_root(definition_doc);
    const char *name = nn_project_string_field(root, "name");
    yyjson_val *batch = yyjson_obj_get(root, "batch");
    okay = name && *name && batch && (dataset->name = nn_text_copy(name)) &&
           parse_slots(yyjson_obj_get(batch, "inputs"), &dataset->inputs,
                        &dataset->input_count, error, capacity) &&
           parse_slots(yyjson_obj_get(batch, "targets"), &dataset->targets,
                        &dataset->target_count, error, capacity);
    if (okay && !dataset->input_count) { nn_errorf(error, capacity, "dataset requires at least one input slot"); okay = false; }
    for (size_t i = 0; okay && i < dataset->input_count; ++i)
        for (size_t j = 0; j < dataset->target_count; ++j)
            if (!strcmp(dataset->inputs[i].name, dataset->targets[j].name)) {
                nn_errorf(error, capacity, "duplicate dataset slot name: %s", dataset->inputs[i].name);
                okay = false;
                break;
            }
    if (!okay && error && capacity && !error[0]) nn_errorf(error, capacity, "invalid dataset definition");
    yyjson_doc_free(definition_doc);
    return okay;
}

void nn_project_dataset_dispose(NNDataset *dataset)
{
    if (!dataset) return;
    free(dataset->id); free(dataset->version); free(dataset->path); free(dataset->name);
    for (size_t i = 0; dataset->inputs && i < dataset->input_count; ++i) {
        free(dataset->inputs[i].name); free(dataset->inputs[i].dtype);
        nn_value_dispose(&dataset->inputs[i].shape);
    }
    for (size_t i = 0; dataset->targets && i < dataset->target_count; ++i) {
        free(dataset->targets[i].name); free(dataset->targets[i].dtype);
        nn_value_dispose(&dataset->targets[i].shape);
    }
    free(dataset->inputs); free(dataset->targets);
}

bool nn_project_select_dataset(NNProject *project, const char *id, const char *version,
                               char *error, size_t capacity)
{
    if (error && capacity) error[0] = '\0';
    if (!project || !id || !version) { nn_errorf(error, capacity, "invalid dataset identity"); return false; }
    const NNDataset *selected = NULL;
    for (size_t i = 0; i < project->dataset_count; ++i)
        if (!strcmp(project->datasets[i].id, id) && !strcmp(project->datasets[i].version, version)) selected = &project->datasets[i];
    if (!selected) { nn_errorf(error, capacity, "dataset identity is not declared"); return false; }
    char *new_id = nn_text_copy(id), *new_version = nn_text_copy(version);
    if (!new_id || !new_version) { free(new_id); free(new_version); nn_errorf(error, capacity, "out of memory"); return false; }
    char *old_id = project->active_dataset_id, *old_version = project->active_dataset_version;
    project->active_dataset_id = new_id;
    project->active_dataset_version = new_version;
    project->dirty = true;
    free(old_id); free(old_version);
    if (error && capacity) error[0] = '\0';
    return true;
}
