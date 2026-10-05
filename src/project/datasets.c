#define _XOPEN_SOURCE 700
#include "project_internal.h"
#include "utils/utils.h"

#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdint.h>
#include <sys/stat.h>
#include <unistd.h>

#define NN_DATASET_DEFINITION_LIMIT (16u * 1024u * 1024u)

static bool parse_slots(yyjson_val *object, NNTensorSlot **slots, size_t *count,
                        char *error, size_t capacity);

static char *dataset_definition_path(const NNProject *project, const NNDataset *dataset,
                                    char *error, size_t capacity)
{
    char *directory = nn_project_safe_resource_dir(project->directory, dataset->path, error, capacity);
    char *manifest_path = directory ? nn_path_join(directory, "manifest.json") : NULL;
    yyjson_doc *manifest = manifest_path ? nn_project_read_document(manifest_path, error, capacity) : NULL;
    yyjson_val *root = manifest ? yyjson_doc_get_root(manifest) : NULL;
    yyjson_val *entrypoints = root ? yyjson_obj_get(root, "entrypoints") : NULL;
    const char *definition = nn_project_string_field(entrypoints, "definition");
    bool valid = manifest && yyjson_get_int(yyjson_obj_get(root, "schemaVersion")) == 1 &&
        !strcmp(nn_project_string_field(root, "id") ? nn_project_string_field(root, "id") : "", dataset->id) &&
        !strcmp(nn_project_string_field(root, "version") ? nn_project_string_field(root, "version") : "", dataset->version) &&
        definition && nn_project_valid_id(definition) && !strchr(definition, '/');
    char *path = valid && directory ? nn_path_join(directory, definition) : NULL;
    if (!valid) nn_errorf(error, capacity, "dataset identity or definition invalid: %s", dataset->path);
    else if (!path) nn_errorf(error, capacity, "unable to build dataset definition path");
    if (manifest) yyjson_doc_free(manifest);
    free(manifest_path); free(directory);
    return path;
}

static bool parse_dataset_definition(NNDataset *dataset, yyjson_doc *definition_doc,
                                     char *error, size_t capacity)
{
    yyjson_val *root = yyjson_doc_get_root(definition_doc);
    const char *name = nn_project_string_field(root, "name");
    yyjson_val *batch = yyjson_obj_get(root, "batch");
    bool okay = yyjson_is_obj(root) && name && *name && yyjson_is_obj(batch) &&
        (dataset->name = nn_text_copy(name)) &&
        parse_slots(yyjson_obj_get(batch, "inputs"), &dataset->inputs,
                   &dataset->input_count, error, capacity) &&
        parse_slots(yyjson_obj_get(batch, "targets"), &dataset->targets,
                   &dataset->target_count, error, capacity);
    if (okay && !dataset->input_count) {
        nn_errorf(error, capacity, "dataset requires at least one input slot");
        okay = false;
    }
    for (size_t i = 0; okay && i < dataset->input_count; ++i)
        for (size_t j = 0; j < dataset->target_count; ++j)
            if (!strcmp(dataset->inputs[i].name, dataset->targets[j].name)) {
                nn_errorf(error, capacity, "duplicate dataset slot name: %s", dataset->inputs[i].name);
                okay = false;
                break;
            }
    if (!okay && error && capacity && !error[0])
        nn_errorf(error, capacity, "invalid dataset definition");
    return okay;
}

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
    char *definition_path = dataset_definition_path(project, dataset, error, capacity);
    yyjson_doc *definition_doc = definition_path ? nn_project_read_document(definition_path, error, capacity) : NULL;
    free(definition_path);
    if (!definition_doc) return false;
    bool okay = parse_dataset_definition(dataset, definition_doc, error, capacity);
    yyjson_doc_free(definition_doc);
    return okay;
}

static bool read_definition_bytes(const char *path, char **contents, size_t *length,
                                  char *error, size_t capacity)
{
    struct stat st;
    if (lstat(path, &st) || !S_ISREG(st.st_mode) || st.st_size < 0 ||
        (uintmax_t)st.st_size > NN_DATASET_DEFINITION_LIMIT) {
        nn_errorf(error, capacity, "dataset definition is not a regular file or exceeds the size limit");
        return false;
    }
    int fd = open(path, O_RDONLY | O_NOFOLLOW);
    if (fd < 0) { nn_errorf(error, capacity, "cannot open dataset definition: %s", strerror(errno)); return false; }
    size_t size = (size_t)st.st_size;
    char *text = malloc(size + 1);
    if (!text) { close(fd); nn_errorf(error, capacity, "out of memory reading dataset definition"); return false; }
    size_t offset = 0;
    while (offset < size) {
        ssize_t count = read(fd, text + offset, size - offset);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) break;
        offset += (size_t)count;
    }
    bool okay = offset == size && close(fd) == 0;
    if (!okay) {
        if (offset != size) close(fd);
        free(text);
        nn_errorf(error, capacity, "cannot read complete dataset definition");
        return false;
    }
    text[size] = '\0'; *contents = text; *length = size;
    return true;
}

static bool write_sibling(const char *path, const char *contents, size_t length,
                          mode_t mode, char **created_path, char *error, size_t capacity)
{
    static const char suffix[] = ".nn-update-XXXXXX";
    size_t path_length = strlen(path), suffix_length = strlen(suffix);
    char *sibling = malloc(path_length + suffix_length + 1);
    if (!sibling) { nn_errorf(error, capacity, "out of memory staging dataset definition"); return false; }
    memcpy(sibling, path, path_length);
    memcpy(sibling + path_length, suffix, suffix_length + 1);
    int fd = mkstemp(sibling);
    if (fd < 0) { free(sibling); nn_errorf(error, capacity, "cannot stage dataset definition: %s", strerror(errno)); return false; }
    bool okay = fchmod(fd, mode & 0777) == 0 && nn_project_write_all(fd, contents, length) && fsync(fd) == 0;
    if (close(fd)) okay = false;
    if (!okay) {
        unlink(sibling); free(sibling);
        nn_errorf(error, capacity, "cannot flush staged dataset definition");
        return false;
    }
    *created_path = sibling;
    return true;
}

static bool replace_text(yyjson_mut_doc *doc, yyjson_mut_val *object, const char *key,
                         yyjson_mut_val *value)
{
    if (yyjson_mut_obj_get(object, key)) yyjson_mut_obj_remove_key(object, key);
    return yyjson_mut_obj_add_val(doc, object, key, value);
}

static yyjson_mut_val *merge_slot_map(yyjson_mut_doc *doc, yyjson_val *old_slots,
                                      yyjson_val *new_slots)
{
    yyjson_mut_val *merged = yyjson_mut_obj(doc);
    if (!merged || !yyjson_is_obj(new_slots)) return NULL;
    size_t index, max; yyjson_val *key, *value;
    yyjson_obj_foreach(new_slots, index, max, key, value) {
        const char *name = yyjson_get_str(key);
        yyjson_val *old_slot = old_slots ? yyjson_obj_get(old_slots, name) : NULL;
        yyjson_mut_val *slot = old_slot && yyjson_is_obj(old_slot)
            ? yyjson_val_mut_copy(doc, old_slot) : yyjson_mut_obj(doc);
        yyjson_val *dtype = yyjson_obj_get(value, "dtype");
        yyjson_val *shape = yyjson_obj_get(value, "shape");
        yyjson_mut_val *dtype_copy = dtype ? yyjson_val_mut_copy(doc, dtype) : NULL;
        yyjson_mut_val *shape_copy = shape ? yyjson_val_mut_copy(doc, shape) : NULL;
        if (!name || !slot || !dtype_copy || !shape_copy ||
            !replace_text(doc, slot, "dtype", dtype_copy) ||
            !replace_text(doc, slot, "shape", shape_copy) ||
            !yyjson_mut_obj_add_val(doc, merged, name, slot)) return NULL;
    }
    return merged;
}

static yyjson_mut_val *merge_dataset_definition(yyjson_mut_doc *doc,
                                               yyjson_val *original,
                                               yyjson_val *submitted)
{
    yyjson_val *old_batch = yyjson_obj_get(original, "batch");
    yyjson_val *new_batch = yyjson_obj_get(submitted, "batch");
    yyjson_mut_val *root = yyjson_mut_obj(doc), *batch = yyjson_mut_obj(doc);
    if (!root || !batch || !yyjson_is_obj(original) || !yyjson_is_obj(submitted) ||
        !yyjson_is_obj(old_batch) || !yyjson_is_obj(new_batch)) return NULL;
    size_t index, max; yyjson_val *key, *value;
    yyjson_obj_foreach(original, index, max, key, value) {
        const char *name = yyjson_get_str(key);
        if (!strcmp(name, "name") || !strcmp(name, "description") || !strcmp(name, "batch")) continue;
        yyjson_mut_val *copy = yyjson_val_mut_copy(doc, value);
        if (!copy || !yyjson_mut_obj_add_val(doc, root, name, copy)) return NULL;
    }
    const char *editable[] = {"name", "description"};
    for (size_t i = 0; i < 2; ++i) {
        yyjson_val *field = yyjson_obj_get(submitted, editable[i]);
        if (!field) field = yyjson_obj_get(original, editable[i]);
        if (!field) continue;
        yyjson_mut_val *copy = yyjson_val_mut_copy(doc, field);
        if (!copy || !yyjson_mut_obj_add_val(doc, root, editable[i], copy)) return NULL;
    }
    yyjson_obj_foreach(old_batch, index, max, key, value) {
        const char *name = yyjson_get_str(key);
        if (!strcmp(name, "inputs") || !strcmp(name, "targets")) continue;
        yyjson_mut_val *copy = yyjson_val_mut_copy(doc, value);
        if (!copy || !yyjson_mut_obj_add_val(doc, batch, name, copy)) return NULL;
    }
    const char *maps[] = {"inputs", "targets"};
    for (size_t i = 0; i < 2; ++i) {
        yyjson_mut_val *slots = merge_slot_map(doc, yyjson_obj_get(old_batch, maps[i]),
                                               yyjson_obj_get(new_batch, maps[i]));
        if (!slots || !yyjson_mut_obj_add_val(doc, batch, maps[i], slots)) return NULL;
    }
    if (!yyjson_mut_obj_add_val(doc, root, "batch", batch)) return NULL;
    return root;
}

char *nn_project_dataset_definition(const NNProject *project, const char *id,
                                   const char *version, char *error, size_t capacity)
{
    if (error && capacity) error[0] = '\0';
    if (!project || !id || !version) { nn_errorf(error, capacity, "invalid dataset identity"); return NULL; }
    const NNDataset *dataset = NULL;
    for (size_t i = 0; i < project->dataset_count; ++i)
        if (!strcmp(project->datasets[i].id, id) && !strcmp(project->datasets[i].version, version)) dataset = &project->datasets[i];
    if (!dataset) { nn_errorf(error, capacity, "dataset identity is not declared"); return NULL; }
    char *path = dataset_definition_path(project, dataset, error, capacity);
    char *bytes = NULL; size_t length = 0;
    bool okay = path && read_definition_bytes(path, &bytes, &length, error, capacity);
    yyjson_doc *definition = okay ? yyjson_read(bytes, length, 0) : NULL;
    if (okay && !definition) { nn_errorf(error, capacity, "dataset definition is invalid JSON"); okay = false; }
    char *result = NULL;
    if (okay) {
        size_t result_length = 0;
        result = yyjson_val_write_opts(yyjson_doc_get_root(definition), YYJSON_WRITE_PRETTY_TWO_SPACES,
                                       NULL, &result_length, NULL);
        if (result) {
            char *grown = realloc(result, result_length + 2);
            if (!grown) { free(result); result = NULL; }
            else { grown[result_length++] = '\n'; grown[result_length] = '\0'; result = grown; }
        }
        if (!result) nn_errorf(error, capacity, "out of memory serializing dataset definition");
    }
    if (definition) yyjson_doc_free(definition);
    free(bytes); free(path);
    return result;
}

bool nn_project_update_dataset(NNProject *project, const char *id, const char *version,
                               const char *definition_json, char *error, size_t capacity)
{
    if (error && capacity) error[0] = '\0';
    if (!project || !id || !version || !definition_json || strlen(definition_json) > NN_DATASET_DEFINITION_LIMIT) {
        nn_errorf(error, capacity, "invalid dataset update payload"); return false;
    }
    size_t index = project->dataset_count;
    for (size_t i = 0; i < project->dataset_count; ++i)
        if (!strcmp(project->datasets[i].id, id) && !strcmp(project->datasets[i].version, version)) index = i;
    if (index == project->dataset_count) { nn_errorf(error, capacity, "dataset identity is not declared"); return false; }
    const NNDataset *current = &project->datasets[index];
    char *path = dataset_definition_path(project, current, error, capacity);
    char *original_bytes = NULL; size_t original_length = 0;
    bool okay = path && read_definition_bytes(path, &original_bytes, &original_length, error, capacity);
    struct stat original_stat;
    if (okay && lstat(path, &original_stat)) { nn_errorf(error, capacity, "cannot inspect dataset definition: %s", strerror(errno)); okay = false; }
    yyjson_doc *original = okay ? yyjson_read(original_bytes, original_length, 0) : NULL;
    yyjson_doc *submitted = yyjson_read(definition_json, strlen(definition_json), 0);
    yyjson_mut_doc *merged_doc = yyjson_mut_doc_new(NULL);
    yyjson_mut_val *merged_root = original && submitted && merged_doc
        ? merge_dataset_definition(merged_doc, yyjson_doc_get_root(original), yyjson_doc_get_root(submitted)) : NULL;
    if (okay && (!original || !submitted || !merged_root)) {
        nn_errorf(error, capacity, "dataset definition and update must be JSON objects"); okay = false;
    }
    NNDataset candidate = { .id = nn_text_copy(current->id), .version = nn_text_copy(current->version),
                            .path = nn_text_copy(current->path) };
    if (okay && (!candidate.id || !candidate.version || !candidate.path)) {
        nn_errorf(error, capacity, "out of memory staging dataset metadata"); okay = false;
    }
    if (okay) {
        yyjson_mut_doc_set_root(merged_doc, merged_root);
    }
    /* Validate the merged DOM through the same parser used by project loading. */
    if (okay) {
        size_t merged_length = 0;
        char *merged_json = yyjson_mut_write_opts(merged_doc, YYJSON_WRITE_PRETTY_TWO_SPACES,
                                                  NULL, &merged_length, NULL);
        yyjson_doc *candidate_doc = merged_json ? yyjson_read(merged_json, merged_length, 0) : NULL;
        if (!candidate_doc) { nn_errorf(error, capacity, "cannot parse merged dataset definition"); okay = false; }
        else if (!parse_dataset_definition(&candidate, candidate_doc, error, capacity)) okay = false;
        if (candidate_doc) yyjson_doc_free(candidate_doc);
        if (!merged_json) nn_errorf(error, capacity, "cannot serialize merged dataset definition");
        if (merged_json) {
            char *grown = realloc(merged_json, merged_length + 2);
            if (!grown) { free(merged_json); merged_json = NULL; nn_errorf(error, capacity, "out of memory staging merged dataset definition"); okay = false; }
            else { grown[merged_length++] = '\n'; grown[merged_length] = '\0'; merged_json = grown; }
        }
        if (okay && merged_json && merged_length > NN_DATASET_DEFINITION_LIMIT) {
            nn_errorf(error, capacity, "dataset JSON exceeds resource file limit"); okay = false;
        }
        if (okay && merged_json) {
            char *temporary = NULL, *backup = NULL;
            okay = write_sibling(path, merged_json, merged_length, original_stat.st_mode,
                                 &temporary, error, capacity) &&
                   write_sibling(path, original_bytes, original_length, original_stat.st_mode,
                                 &backup, error, capacity);
            bool replaced = false, swapped = false;
            bool backup_moved = false, keep_backup = false;
            bool old_dirty = project->dirty;
            NNDataset old_dataset = project->datasets[index];
            if (okay) {
                if (rename(temporary, path)) { nn_errorf(error, capacity, "cannot replace dataset definition: %s", strerror(errno)); okay = false; }
                else replaced = true;
            }
            if (okay) {
                project->datasets[index] = candidate; memset(&candidate, 0, sizeof(candidate));
                swapped = true;
                project->dirty = true;
                okay = nn_project_save(project, error, capacity);
            }
            if (okay) {
                nn_project_dataset_dispose(&old_dataset);
                if (backup) unlink(backup);
            } else {
                if (replaced) {
                    if (backup && rename(backup, path) == 0) backup_moved = true;
                    else {
                        keep_backup = backup != NULL;
                        nn_errorf(error, capacity, "dataset update failed; original definition retained for recovery at %s", backup ? backup : path);
                    }
                }
                if (swapped) {
                    nn_project_dataset_dispose(&project->datasets[index]);
                    project->datasets[index] = old_dataset;
                    memset(&old_dataset, 0, sizeof(old_dataset));
                }
                project->dirty = old_dirty;
                if (swapped) nn_project_dataset_dispose(&old_dataset);
            }
            if (temporary) { unlink(temporary); free(temporary); }
            if (backup && !backup_moved && !keep_backup) unlink(backup);
            if (backup) free(backup);
        }
        free(merged_json);
    }
    nn_project_dataset_dispose(&candidate);
    if (merged_doc) yyjson_mut_doc_free(merged_doc);
    if (submitted) yyjson_doc_free(submitted);
    if (original) yyjson_doc_free(original);
    free(original_bytes); free(path);
    if (!okay && error && capacity && !error[0]) nn_errorf(error, capacity, "dataset update failed");
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
