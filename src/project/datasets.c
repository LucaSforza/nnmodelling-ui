#define _XOPEN_SOURCE 700
#include "project_internal.h"
#include "utils/utils.h"

#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <dirent.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdint.h>
#include <sys/stat.h>
#include <unistd.h>

#define NN_DATASET_DEFINITION_LIMIT (16u * 1024u * 1024u)

static bool parse_slots(yyjson_val *object, NNTensorSlot **slots, size_t *count,
                        char *error, size_t capacity);

static bool dataset_entrypoint_path(const char *directory, const char *relative,
                                   char **path_out)
{
    *path_out = NULL;
    if (!relative || !*relative || relative[0] == '/' || strlen(relative) > 512) return false;
    char *current = nn_text_copy(directory);
    if (!current) return false;
    const char *part = relative;
    for (const char *end = relative;; ++end) {
        if (*end != '/' && *end != '\0') continue;
        size_t length = (size_t)(end - part);
        if (!length || (length == 1 && part[0] == '.') ||
            (length == 2 && part[0] == '.' && part[1] == '.') ||
            (*end == '/' && end[1] == '\0')) { free(current); return false; }
        for (size_t i = 0; i < length; ++i) {
            unsigned char c = (unsigned char)part[i];
            if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                  (c >= '0' && c <= '9') || c == '-' || c == '_' || c == '.')) {
                free(current); return false;
            }
        }
        char *segment = malloc(length + 1);
        if (!segment) { free(current); return false; }
        memcpy(segment, part, length); segment[length] = '\0';
        char *next = nn_path_join(current, segment);
        free(segment); free(current); current = next;
        if (!current) return false;
        struct stat st;
        if (lstat(current, &st) ||
            (*end == '/' ? !S_ISDIR(st.st_mode) : !S_ISREG(st.st_mode))) {
            free(current); return false;
        }
        if (*end == '\0') break;
        part = end + 1;
    }
    *path_out = current;
    return true;
}

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
        definition;
    char *path = NULL;
    if (valid && (!directory || !dataset_entrypoint_path(directory, definition, &path))) valid = false;
    if (!valid) nn_errorf(error, capacity, "dataset identity or definition path invalid: %s", dataset->path);
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

bool nn_project_bind_dataset_input(NNProject *project, const NNDataset *dataset,
                                   char *error, size_t capacity)
{
    if (!project || !dataset) return nn_fail(error, capacity, "invalid dataset binding request");
    if (dataset->input_count != 1) return true;
    const NNNode *input = NULL;
    size_t count = 0;
    for (size_t i = 0; i < nn_model_node_count(project->model); ++i) {
        const NNNode *node = nn_model_node_at(project->model, i);
        const NNPackage *package = nn_catalog_find(project->catalog, node->package_id, node->package_version);
        if (!node->scope_id[0] && package && package->kind && !strcmp(package->kind, "input")) {
            input = node;
            ++count;
        }
    }
    if (count > 1) return nn_fail(error, capacity, "cannot bind a single-input dataset with multiple root Input nodes");
    if (!input) return true;
    const NNValue *binding = NULL;
    for (size_t i = 0; i < input->parameter_count; ++i)
        if (!strcmp(input->parameters[i].key, "binding")) binding = &input->parameters[i].value;
    if (binding && binding->type == NN_VALUE_STRING && binding->as.string &&
        !strcmp(binding->as.string, dataset->inputs[0].name)) return true;
    NNValue value = {.type = NN_VALUE_STRING, .as.string = dataset->inputs[0].name};
    return nn_model_set_parameter(project->model, input->id, "binding", &value, error, capacity);
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

#define NN_DATASET_IMPORT_FILE_LIMIT 100000u
#define NN_DATASET_IMPORT_BYTE_LIMIT ((uintmax_t)8 * 1024u * 1024u * 1024u)

typedef struct { size_t files; uintmax_t bytes; } NNImportBudget;

static bool path_contains(const char *parent, const char *child)
{
    size_t length = strlen(parent);
    if (strncmp(parent, child, length)) return false;
    return child[length] == '\0' || parent[length - 1] == '/' || child[length] == '/';
}

static bool copy_import_entry(const char *source, const char *destination,
                              NNImportBudget *budget, char *error, size_t capacity)
{
    struct stat st;
    if (lstat(source, &st) || S_ISLNK(st.st_mode))
        return nn_fail(error, capacity, "dataset source contains a missing or symbolic link entry");
    if (S_ISDIR(st.st_mode)) {
        if (mkdir(destination, 0755))
            return nn_fail(error, capacity, "cannot create staged dataset directory");
        DIR *directory = opendir(source);
        if (!directory) return nn_fail(error, capacity, "cannot read dataset source directory");
        bool okay = true;
        struct dirent *entry;
        while (okay && (entry = readdir(directory))) {
            if (!strcmp(entry->d_name, ".") || !strcmp(entry->d_name, "..")) continue;
            char *from = nn_path_join(source, entry->d_name);
            char *to = nn_path_join(destination, entry->d_name);
            okay = from && to && copy_import_entry(from, to, budget, error, capacity);
            free(from); free(to);
        }
        if (closedir(directory)) okay = false;
        if (!okay && error && capacity && !error[0]) nn_errorf(error, capacity, "cannot finish copying dataset directory");
        return okay;
    }
    if (!S_ISREG(st.st_mode) || st.st_size < 0 || (uintmax_t)st.st_size > NN_DATASET_IMPORT_BYTE_LIMIT ||
        ++budget->files > NN_DATASET_IMPORT_FILE_LIMIT ||
        budget->bytes > NN_DATASET_IMPORT_BYTE_LIMIT - (uintmax_t)st.st_size)
        return nn_fail(error, capacity, "dataset source contains an unsupported or oversized entry");
    int in = open(source, O_RDONLY | O_NOFOLLOW);
    if (in < 0) return nn_fail(error, capacity, "cannot open dataset source file");
    struct stat opened;
    if (fstat(in, &opened) || !S_ISREG(opened.st_mode) || opened.st_dev != st.st_dev || opened.st_ino != st.st_ino) {
        close(in);
        return nn_fail(error, capacity, "dataset source changed while importing");
    }
    int out = open(destination, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW, st.st_mode & 0777);
    if (out < 0) { close(in); return nn_fail(error, capacity, "cannot create staged dataset file"); }
    char buffer[65536];
    bool okay = true;
    for (;;) {
        ssize_t n = read(in, buffer, sizeof(buffer));
        if (n < 0 && errno == EINTR) continue;
        if (n < 0) { okay = false; break; }
        if (!n) break;
        if ((uintmax_t)n > NN_DATASET_IMPORT_BYTE_LIMIT - budget->bytes) { okay = false; break; }
        if (!nn_project_write_all(out, buffer, (size_t)n)) { okay = false; break; }
        budget->bytes += (uintmax_t)n;
    }
    if (fsync(out)) okay = false;
    if (close(out)) okay = false;
    if (close(in)) okay = false;
    if (!okay) return nn_fail(error, capacity, "cannot copy complete dataset file");
    return true;
}

static bool copy_import_contents(const char *source, const char *destination,
                                 NNImportBudget *budget, char *error, size_t capacity)
{
    DIR *directory = opendir(source);
    if (!directory) return nn_fail(error, capacity, "cannot read dataset source directory");
    bool okay = true;
    struct dirent *entry;
    while (okay && (entry = readdir(directory))) {
        if (!strcmp(entry->d_name, ".") || !strcmp(entry->d_name, "..")) continue;
        char *from = nn_path_join(source, entry->d_name);
        char *to = nn_path_join(destination, entry->d_name);
        okay = from && to && copy_import_entry(from, to, budget, error, capacity);
        free(from); free(to);
    }
    if (closedir(directory)) okay = false;
    if (!okay && error && capacity && !error[0]) nn_errorf(error, capacity, "cannot finish copying dataset directory");
    return okay;
}

static bool import_relative_path(const char *text, bool leaf_only)
{
    if (!text || !*text || text[0] == '/' || strlen(text) > 512) return false;
    const char *part = text;
    for (const char *end = text;; ++end) {
        if (*end != '/' && *end != '\0') continue;
        size_t n = (size_t)(end - part);
        if (!n || (n == 1 && part[0] == '.') || (n == 2 && part[0] == '.' && part[1] == '.') ||
            (leaf_only && *end == '/')) return false;
        for (size_t i = 0; i < n; ++i) {
            unsigned char c = (unsigned char)part[i];
            if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                  (c >= '0' && c <= '9') || c == '-' || c == '_' || c == '.')) return false;
        }
        if (!*end) return true;
        part = end + 1;
    }
}

static bool write_import_text(const char *path, const char *text)
{
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC | O_NOFOLLOW, 0644);
    if (fd < 0) return false;
    size_t length = strlen(text);
    bool okay = nn_project_write_all(fd, text, length) && fsync(fd) == 0;
    if (close(fd)) okay = false;
    return okay;
}

static char *pretty_import_json(yyjson_val *root)
{
    size_t length = 0;
    char *text = yyjson_val_write_opts(root, YYJSON_WRITE_PRETTY_TWO_SPACES, NULL, &length, NULL);
    if (!text) return NULL;
    char *grown = realloc(text, length + 2);
    if (!grown) { free(text); return NULL; }
    grown[length++] = '\n'; grown[length] = '\0';
    return grown;
}

static char *pretty_mut_import_json(yyjson_mut_val *root)
{
    size_t length = 0;
    char *text = yyjson_mut_val_write_opts(root, YYJSON_WRITE_PRETTY_TWO_SPACES, NULL, &length, NULL);
    if (!text) return NULL;
    char *grown = realloc(text, length + 2);
    if (!grown) { free(text); return NULL; }
    grown[length++] = '\n'; grown[length] = '\0';
    return grown;
}

static bool mkdir_import_parents(const char *root, const char *relative)
{
    char *current = nn_text_copy(root);
    if (!current) return false;
    const char *part = relative;
    for (const char *end = relative; *end; ++end) {
        if (*end != '/') continue;
        size_t n = (size_t)(end - part);
        char *segment = malloc(n + 1);
        if (!segment) { free(current); return false; }
        memcpy(segment, part, n); segment[n] = '\0';
        char *next = nn_path_join(current, segment);
        free(segment); free(current); current = next;
        if (!current) return false;
        if (mkdir(current, 0755) && errno != EEXIST) { free(current); return false; }
        struct stat st;
        if (lstat(current, &st) || !S_ISDIR(st.st_mode)) { free(current); return false; }
        part = end + 1;
    }
    free(current);
    return true;
}

static const char *dataset_import_scaffold(void)
{
    return "from collections.abc import Iterator\nfrom typing import Generic, TypeVar\n\n"
           "from torch import Tensor\nfrom nnmodelling_runtime import Batch, DatasetAdapter\n\n"
           "InputT = TypeVar(\"InputT\")\nOutputT = TypeVar(\"OutputT\")\n\n\n"
           "class Dataset(DatasetAdapter[InputT, OutputT], Generic[InputT, OutputT]):\n"
           "    def tokenize(self, value: InputT) -> Tensor | dict[str, Tensor]:\n"
           "        raise NotImplementedError(\"Implement dataset input tokenization.\")\n\n"
           "    def untokenize(self, tensor: Tensor) -> OutputT:\n"
           "        raise NotImplementedError(\"Implement prediction decoding.\")\n\n"
           "    def load(self, split: str, batch_size: int) -> Iterator[Batch]:\n"
           "        raise NotImplementedError(\"Load and batch the requested dataset split.\")\n";
}

bool nn_project_import_dataset(NNProject *project, const char *source_dir,
                               const char *manifest_json, const char *definition_json,
                               bool select, char *error, size_t capacity)
{
    if (error && capacity) error[0] = '\0';
    if (!project || !source_dir || !manifest_json || !definition_json ||
        strlen(manifest_json) > 4u * 1024u * 1024u || strlen(definition_json) > NN_DATASET_DEFINITION_LIMIT)
        return nn_fail(error, capacity, "invalid dataset import payload");
    struct stat selected_source;
    if (lstat(source_dir, &selected_source) || S_ISLNK(selected_source.st_mode) || !S_ISDIR(selected_source.st_mode))
        return nn_fail(error, capacity, "dataset source must be a non-symbolic local directory");
    char *source = realpath(source_dir, NULL);
    struct stat source_stat;
    if (!source || lstat(source, &source_stat) || !S_ISDIR(source_stat.st_mode)) {
        free(source);
        return nn_fail(error, capacity, "dataset source must be a local directory");
    }
    if (path_contains(project->directory, source) || path_contains(source, project->directory)) {
        free(source);
        return nn_fail(error, capacity, "dataset source cannot overlap the destination project directory");
    }
    yyjson_doc *manifest = yyjson_read(manifest_json, strlen(manifest_json), 0);
    yyjson_doc *definition = yyjson_read(definition_json, strlen(definition_json), 0);
    yyjson_mut_doc *mutable_manifest = manifest ? yyjson_doc_mut_copy(manifest, NULL) : NULL;
    yyjson_val *manifest_root = manifest ? yyjson_doc_get_root(manifest) : NULL;
    const char *id = nn_project_string_field(manifest_root, "id");
    const char *version = nn_project_string_field(manifest_root, "version");
    yyjson_val *entrypoints = yyjson_obj_get(manifest_root, "entrypoints");
    const char *definition_file = nn_project_string_field(entrypoints, "definition");
    yyjson_val *adapter = yyjson_obj_get(entrypoints, "python");
    const char *adapter_file = nn_project_string_field(adapter, "file");
    bool valid = manifest && definition && mutable_manifest && yyjson_is_obj(manifest_root) &&
        yyjson_get_int(yyjson_obj_get(manifest_root, "schemaVersion")) == 1 &&
        nn_project_valid_id(id) && nn_project_valid_semver(version) &&
        import_relative_path(definition_file, false) &&
        (!adapter || (yyjson_is_obj(adapter) && nn_project_string_field(adapter, "language") &&
                      !strcmp(nn_project_string_field(adapter, "language"), "python") &&
                      import_relative_path(adapter_file, false)));
    if (!valid) nn_error_set(error, capacity, "dataset manifest must contain safe schema-1 identity and entrypoints");
    yyjson_val *root = definition ? yyjson_doc_get_root(definition) : NULL;
    NNDataset candidate = {0};
    if (valid) {
        candidate.id = nn_text_copy(id); candidate.version = nn_text_copy(version);
        if (!candidate.id || !candidate.version || !parse_dataset_definition(&candidate, definition, error, capacity)) valid = false;
    }
    for (size_t i = 0; valid && i < project->dataset_count; ++i)
        if (!strcmp(project->datasets[i].id, id) && !strcmp(project->datasets[i].version, version)) {
            nn_error_set(error, capacity, "duplicate dataset identity"); valid = false;
        }
    if (valid && project->dataset_count >= 128) { nn_error_set(error, capacity, "dataset catalog limit reached"); valid = false; }
    if (valid && !adapter) {
        yyjson_mut_val *mutable_root = yyjson_mut_doc_get_root(mutable_manifest);
        yyjson_mut_val *mutable_entries = yyjson_mut_obj_get(mutable_root, "entrypoints");
        yyjson_mut_val *python = yyjson_mut_obj(mutable_manifest);
        if (!python || !yyjson_mut_obj_add_strcpy(mutable_manifest, python, "language", "python") ||
            !yyjson_mut_obj_add_strcpy(mutable_manifest, python, "file", "dataset.py") ||
            !yyjson_mut_obj_add_val(mutable_manifest, mutable_entries, "python", python)) valid = false;
        adapter_file = "dataset.py";
    }
    if (valid && (!strcmp(definition_file, "manifest.json") || !strcmp(definition_file, "pyproject.toml") ||
                  !strcmp(definition_file, adapter_file) || !strcmp(adapter_file, "manifest.json") ||
                  !strcmp(adapter_file, "pyproject.toml"))) {
        nn_error_set(error, capacity, "dataset entrypoints collide with reserved metadata paths"); valid = false;
    }
    char name[256] = {0};
    if (valid && snprintf(name, sizeof(name), "%s-%s", id, version) >= (int)sizeof(name)) valid = false;
    char *relative = valid ? nn_path_join("datasets", name) : NULL;
    char *category = relative ? nn_path_join(project->directory, "datasets") : NULL;
    char *destination = relative ? nn_path_join(project->directory, relative) : NULL;
    struct stat st;
    bool category_created = false;
    if (valid && category && lstat(category, &st)) {
        if (errno != ENOENT || mkdir(category, 0755)) valid = false;
        else category_created = true;
    } else if (valid && (!category || !S_ISDIR(st.st_mode))) valid = false;
    if (valid) {
        if (!destination) valid = false;
        else if (!lstat(destination, &st)) {
            nn_error_set(error, capacity, "dataset destination already exists"); valid = false;
        } else if (errno != ENOENT) valid = false;
    }
    char *stage_template = category ? nn_path_join(category, ".nn-import-XXXXXX") : NULL;
    char *stage = valid && stage_template ? mkdtemp(stage_template) : NULL;
    if (valid && !stage) { nn_errorf(error, capacity, "cannot create dataset staging directory: %s", strerror(errno)); valid = false; }
    NNImportBudget budget = {0};
    if (valid && !copy_import_contents(source, stage, &budget, error, capacity)) valid = false;
    char *definition_path = valid ? nn_path_join(stage, definition_file) : NULL;
    char *manifest_path = valid ? nn_path_join(stage, "manifest.json") : NULL;
    char *adapter_path = valid ? nn_path_join(stage, adapter_file) : NULL;
    char *pretty_definition = valid ? pretty_import_json(root) : NULL;
    char *pretty_manifest = valid ? pretty_mut_import_json(yyjson_mut_doc_get_root(mutable_manifest)) : NULL;
    if (valid && (!definition_path || !manifest_path || !adapter_path || !pretty_definition || !pretty_manifest ||
        !mkdir_import_parents(stage, definition_file) ||
        !write_import_text(definition_path, pretty_definition) || !write_import_text(manifest_path, pretty_manifest))) {
        nn_error_set(error, capacity, "cannot stage dataset metadata"); valid = false;
    }
    bool adapter_missing = false;
    if (valid && lstat(adapter_path, &st)) {
        if (errno != ENOENT) valid = false;
        else adapter_missing = true;
    } else if (valid && !S_ISREG(st.st_mode)) valid = false;
    if (valid && adapter_missing) {
        if (!mkdir_import_parents(stage, adapter_file) || !write_import_text(adapter_path, dataset_import_scaffold())) {
            nn_error_set(error, capacity, "cannot create dataset adapter scaffold"); valid = false;
        }
    }
    char *pyproject_path = valid ? nn_path_join(stage, "pyproject.toml") : NULL;
    char pyproject[512];
    int pyproject_len = snprintf(pyproject, sizeof(pyproject),
        "[project]\nname = \"nnmodelling-%s\"\nversion = \"%s\"\ndependencies = [\"nnmodelling-runtime>=0.1.0\"]\n", id, version);
    if (valid) {
        if (!pyproject_path) valid = false;
        else if (!lstat(pyproject_path, &st)) valid = S_ISREG(st.st_mode);
        else if (errno == ENOENT)
            valid = pyproject_len >= 0 && (size_t)pyproject_len < sizeof(pyproject) &&
                write_import_text(pyproject_path, pyproject);
        else valid = false;
    }
    NNDataset *grown = valid ? malloc((project->dataset_count + 1) * sizeof(*grown)) : NULL;
    if (valid && !grown) valid = false;
    NNModel *model_backup = valid && select ? nn_model_copy(project->model) : NULL;
    if (valid && select && !model_backup) { nn_error_set(error, capacity, "out of memory staging dataset activation"); valid = false; }
    if (valid && rename(stage, destination)) { nn_errorf(error, capacity, "cannot publish staged dataset: %s", strerror(errno)); valid = false; }
    bool published = valid;
    if (valid) {
        NNDataset *old = project->datasets; size_t old_count = project->dataset_count;
        char *old_id = project->active_dataset_id, *old_version = project->active_dataset_version;
        bool old_dirty = project->dirty;
        if (old_count) memcpy(grown, old, old_count * sizeof(*grown));
        candidate.path = nn_text_copy(relative);
        if (!candidate.path) valid = false;
        if (valid) {
            project->datasets = grown; project->datasets[project->dataset_count++] = candidate;
            memset(&candidate, 0, sizeof(candidate));
            if (select) {
                project->active_dataset_id = nn_text_copy(id);
                project->active_dataset_version = nn_text_copy(version);
                valid = project->active_dataset_id && project->active_dataset_version;
            }
            if (valid && select)
                valid = nn_project_bind_dataset_input(project, &project->datasets[old_count], error, capacity);
            project->dirty = true;
            if (valid) valid = nn_project_save(project, error, capacity);
        }
        if (valid) {
            free(old); if (select) { free(old_id); free(old_version); }
            grown = NULL;
        } else {
            if (project->datasets == grown) {
                if (select) { free(project->active_dataset_id); free(project->active_dataset_version); }
                project->active_dataset_id = old_id; project->active_dataset_version = old_version;
                if (project->dataset_count > old_count) nn_project_dataset_dispose(&project->datasets[old_count]);
                project->datasets = old; project->dataset_count = old_count;
            }
            project->dirty = old_dirty;
        }
    }
    if (!valid && model_backup) nn_model_swap(project->model, model_backup);
    nn_model_free(model_backup);
    if (stage && !published) {
        nn_project_remove_tree(stage);
        if (!lstat(stage, &st)) nn_errorf(error, capacity, "dataset rollback could not remove staging directory: %s", stage);
    }
    if (published && !valid) {
        nn_project_remove_tree(destination);
        if (!lstat(destination, &st)) nn_errorf(error, capacity, "dataset rollback could not remove resource directory: %s", destination);
    }
    if (!valid && category_created && rmdir(category) && errno != ENOENT)
        nn_errorf(error, capacity, "dataset rollback could not remove directory: %s", category);
    if (!valid && error && capacity && !error[0]) nn_error_set(error, capacity, "dataset import failed");
    free(grown); nn_project_dataset_dispose(&candidate);
    if (manifest) yyjson_doc_free(manifest);
    if (definition) yyjson_doc_free(definition);
    if (mutable_manifest) yyjson_mut_doc_free(mutable_manifest);
    free(source); free(relative); free(category); free(destination); free(stage_template);
    free(definition_path); free(manifest_path); free(adapter_path); free(pretty_definition);
    free(pretty_manifest); free(pyproject_path);
    return valid;
}
