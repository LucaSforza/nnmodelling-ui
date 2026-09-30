#define _XOPEN_SOURCE 700
#define _POSIX_C_SOURCE 200809L
#include "project.h"

#include "yyjson.h"

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <dirent.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

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

static void problem(char *error, size_t capacity, const char *format, ...)
{
    if (!error || !capacity) return;
    va_list args;
    va_start(args, format);
    vsnprintf(error, capacity, format, args);
    va_end(args);
}

static char *copy_string(const char *text)
{
    if (!text) return NULL;
    size_t size = strlen(text) + 1;
    char *copy = malloc(size);
    if (copy) memcpy(copy, text, size);
    return copy;
}

static char *join_path(const char *left, const char *right)
{
    size_t a = strlen(left), b = strlen(right);
    if (a > SIZE_MAX - b - 2) return NULL;
    char *path = malloc(a + b + 2);
    if (path) snprintf(path, a + b + 2, "%s/%s", left, right);
    return path;
}

static const char *string_field(yyjson_val *object, const char *key)
{
    return yyjson_get_str(yyjson_obj_get(object, key));
}

static bool valid_id(const char *id)
{
    if (!id || !*id || strlen(id) > 96 || !strcmp(id, ".") || !strcmp(id, "..")) return false;
    for (const unsigned char *c = (const unsigned char *)id; *c; ++c)
        if (!((*c >= 'a' && *c <= 'z') || (*c >= 'A' && *c <= 'Z') ||
              (*c >= '0' && *c <= '9') || *c == '-' || *c == '_' || *c == '.'))
            return false;
    return true;
}

static bool valid_semver(const char *text)
{
    if (!text || !*text) return false;
    for (unsigned part = 0; part < 3; ++part) {
        unsigned long value = 0;
        if (*text < '0' || *text > '9' || (*text == '0' && text[1] >= '0' && text[1] <= '9')) return false;
        while (*text >= '0' && *text <= '9') {
            unsigned digit = (unsigned)(*text++ - '0');
            if (value > (ULONG_MAX - digit) / 10) return false;
            value = value * 10 + digit;
        }
        if (part < 2) { if (*text++ != '.') return false; }
        else if (*text) return false;
    }
    return true;
}

/* Existing resource directory only: reject traversal and every symlink hop. */
static char *safe_resource_dir(const char *root, const char *relative,
                               char *error, size_t capacity)
{
    if (!relative || !*relative || relative[0] == '/' || strlen(relative) > 512) {
        problem(error, capacity, "invalid resource path");
        return NULL;
    }
    char *path = copy_string(root);
    if (!path) return NULL;
    const char *part = relative;
    for (const char *end = relative;; ++end) {
        if (*end != '/' && *end != '\0') continue;
        size_t length = (size_t)(end - part);
        if (!length || (length == 1 && part[0] == '.') ||
            (length == 2 && part[0] == '.' && part[1] == '.')) {
            problem(error, capacity, "resource path traversal rejected: %s", relative);
            free(path);
            return NULL;
        }
        for (size_t i = 0; i < length; ++i) {
            unsigned char c = (unsigned char)part[i];
            if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                  (c >= '0' && c <= '9') || c == '-' || c == '_' || c == '.')) {
                problem(error, capacity, "invalid resource path: %s", relative);
                free(path);
                return NULL;
            }
        }
        char *segment = malloc(length + 1);
        if (!segment) { free(path); return NULL; }
        memcpy(segment, part, length);
        segment[length] = '\0';
        char *next = join_path(path, segment);
        free(segment);
        free(path);
        path = next;
        if (!path) return NULL;
        struct stat st;
        if (lstat(path, &st) || !S_ISDIR(st.st_mode)) {
            problem(error, capacity, "resource directory missing or symlinked: %s", relative);
            free(path);
            return NULL;
        }
        if (*end == '\0') break;
        part = end + 1;
    }
    return path;
}

static yyjson_doc *read_document(const char *path, char *error, size_t capacity)
{
    struct stat st;
    if (lstat(path, &st) || !S_ISREG(st.st_mode) || st.st_size > 16 * 1024 * 1024) {
        problem(error, capacity, "missing, symlinked or oversized JSON: %s", path);
        return NULL;
    }
    yyjson_read_err parse_error;
    yyjson_doc *document = yyjson_read_file(path, 0, NULL, &parse_error);
    if (!document) problem(error, capacity, "JSON parse error in %s: %s", path, parse_error.msg);
    return document;
}

static bool parse_value(yyjson_val *source, NNValue *target)
{
    memset(target, 0, sizeof(*target));
    if (yyjson_is_bool(source)) {
        target->type = NN_VALUE_BOOL;
        target->as.boolean = yyjson_get_bool(source);
    } else if (yyjson_is_int(source)) {
        target->type = NN_VALUE_INT;
        target->as.integer = yyjson_get_sint(source);
    } else if (yyjson_is_real(source)) {
        target->type = NN_VALUE_REAL;
        target->as.real = yyjson_get_real(source);
    } else if (yyjson_is_str(source)) {
        target->type = NN_VALUE_STRING;
        target->as.string = copy_string(yyjson_get_str(source));
        if (!target->as.string) return false;
    } else if (yyjson_is_arr(source)) {
        target->type = NN_VALUE_ARRAY;
        target->as.array.count = yyjson_arr_size(source);
        if (target->as.array.count > 1024) return false;
        target->as.array.items = calloc(target->as.array.count ? target->as.array.count : 1,
                                        sizeof(NNValue));
        if (!target->as.array.items) return false;
        for (size_t i = 0; i < target->as.array.count; ++i)
            if (!parse_value(yyjson_arr_get(source, i), &target->as.array.items[i])) {
                nn_value_dispose(target);
                return false;
            }
    } else return false;
    return true;
}

static bool parse_slots(yyjson_val *object, NNTensorSlot **slots, size_t *count,
                        char *error, size_t capacity)
{
    if (!yyjson_is_obj(object)) { problem(error, capacity, "dataset slots must be an object"); return false; }
    size_t total = yyjson_obj_size(object);
    if (total > 64) { problem(error, capacity, "too many dataset slots"); return false; }
    *slots = calloc(total ? total : 1, sizeof(**slots));
    if (!*slots) return false;
    *count = total;
    size_t object_index, object_max;
    yyjson_val *key, *value;
    size_t index = 0;
    yyjson_obj_foreach(object, object_index, object_max, key, value) {
        NNTensorSlot *slot = &(*slots)[index++];
        const char *dtype = string_field(value, "dtype");
        yyjson_val *shape = yyjson_obj_get(value, "shape");
        const char *allowed[] = {"float16","bfloat16","float32","float64","int8","int16","int32","int64","uint8","bool"};
        bool dtype_ok = false;
        for (size_t d = 0; dtype && d < sizeof(allowed) / sizeof(allowed[0]); ++d)
            if (!strcmp(dtype, allowed[d])) dtype_ok = true;
        if (!yyjson_get_str(key) || !*yyjson_get_str(key) || !dtype_ok ||
            !yyjson_is_arr(shape) || !yyjson_arr_size(shape) || yyjson_arr_size(shape) > 64 ||
            !parse_value(shape, &slot->shape)) {
            problem(error, capacity, "invalid dataset tensor slot");
            return false;
        }
        slot->name = copy_string(yyjson_get_str(key));
        slot->dtype = copy_string(dtype);
        if (!slot->name || !slot->dtype) return false;
        for (size_t previous = 0; previous + 1 < index; ++previous)
            if (!strcmp((*slots)[previous].name, slot->name)) {
                problem(error, capacity, "duplicate dataset slot name: %s", slot->name);
                return false;
            }
        for (size_t i = 0; i < slot->shape.as.array.count; ++i) {
            NNValue *dimension = &slot->shape.as.array.items[i];
            if (!((dimension->type == NN_VALUE_INT && dimension->as.integer > 0) ||
                  (dimension->type == NN_VALUE_STRING &&
                   !strcmp(dimension->as.string, "B")))) {
                problem(error, capacity, "invalid dataset tensor dimension");
                return false;
            }
        }
    }
    return true;
}

static bool load_dataset(NNProject *project, NNDataset *dataset,
                         char *error, size_t capacity)
{
    char *directory = safe_resource_dir(project->directory, dataset->path, error, capacity);
    if (!directory) return false;
    char *manifest_path = join_path(directory, "manifest.json");
    yyjson_doc *manifest = manifest_path ? read_document(manifest_path, error, capacity) : NULL;
    free(manifest_path);
    if (!manifest) { free(directory); return false; }
    yyjson_val *root = yyjson_doc_get_root(manifest);
    yyjson_val *entrypoints = yyjson_obj_get(root, "entrypoints");
    const char *definition = string_field(entrypoints, "definition");
    const char *id = string_field(root, "id"), *version = string_field(root, "version");
    bool okay = yyjson_get_int(yyjson_obj_get(root, "schemaVersion")) == 1 &&
                id && version && !strcmp(id, dataset->id) &&
                !strcmp(version, dataset->version) && definition &&
                valid_id(definition) && !strchr(definition, '/');
    if (!okay) problem(error, capacity, "dataset identity or definition invalid: %s", dataset->path);
    char *definition_path = okay ? join_path(directory, definition) : NULL;
    yyjson_doc *definition_doc = definition_path ? read_document(definition_path, error, capacity) : NULL;
    free(definition_path);
    free(directory);
    yyjson_doc_free(manifest);
    if (!definition_doc) return false;
    root = yyjson_doc_get_root(definition_doc);
    const char *name = string_field(root, "name");
    yyjson_val *batch = yyjson_obj_get(root, "batch");
    okay = name && *name && batch && (dataset->name = copy_string(name)) &&
           parse_slots(yyjson_obj_get(batch, "inputs"), &dataset->inputs,
                        &dataset->input_count, error, capacity) &&
           parse_slots(yyjson_obj_get(batch, "targets"), &dataset->targets,
                        &dataset->target_count, error, capacity);
    if (okay && !dataset->input_count) { problem(error, capacity, "dataset requires at least one input slot"); okay = false; }
    for (size_t i = 0; okay && i < dataset->input_count; ++i)
        for (size_t j = 0; j < dataset->target_count; ++j)
            if (!strcmp(dataset->inputs[i].name, dataset->targets[j].name)) {
                problem(error, capacity, "duplicate dataset slot name: %s", dataset->inputs[i].name);
                okay = false;
                break;
            }
    if (!okay && error && !error[0]) problem(error, capacity, "invalid dataset definition");
    yyjson_doc_free(definition_doc);
    return okay;
}

static bool parse_references(yyjson_val *array, NNResourceRef **references,
                             size_t *count, char *error, size_t capacity)
{
    if (!yyjson_is_arr(array) || yyjson_arr_size(array) > 128) {
        problem(error, capacity, "invalid resource reference list"); return false;
    }
    *count = yyjson_arr_size(array);
    *references = calloc(*count ? *count : 1, sizeof(**references));
    if (!*references) return false;
    for (size_t i = 0; i < *count; ++i) {
        yyjson_val *entry = yyjson_arr_get(array, i);
        const char *id = string_field(entry, "id");
        const char *version = string_field(entry, "version");
        const char *path = string_field(entry, "path");
        if (!valid_id(id) || !valid_id(version) || !path) {
            problem(error, capacity, "invalid resource reference"); return false;
        }
        (*references)[i] = (NNResourceRef){copy_string(id), copy_string(version), copy_string(path)};
        if (!(*references)[i].id || !(*references)[i].version || !(*references)[i].path) return false;
    }
    return true;
}

static bool parse_graph(NNProject *project, yyjson_val *root,
                        char *error, size_t capacity)
{
    yyjson_val *nodes = yyjson_obj_get(root, "nodes"), *edges = yyjson_obj_get(root, "edges");
    if (!yyjson_is_arr(nodes) || !yyjson_is_arr(edges) ||
        yyjson_arr_size(nodes) > 10000 || yyjson_arr_size(edges) > 20000) {
        problem(error, capacity, "invalid graph arrays"); return false;
    }
    project->model = nn_model_new();
    if (!project->model) return false;
    for (size_t i = 0; i < yyjson_arr_size(nodes); ++i) {
        yyjson_val *node = yyjson_arr_get(nodes, i);
        yyjson_val *position = yyjson_obj_get(node, "position");
        yyjson_val *data = yyjson_obj_get(node, "data");
        yyjson_val *package = yyjson_obj_get(data, "package");
        const char *id = string_field(node, "id");
        const char *name = string_field(data, "name");
        const char *package_id = string_field(package, "id");
        const char *version = string_field(package, "version");
        const char *scope = string_field(data, "scope");
        yyjson_val *x = yyjson_obj_get(position, "x"), *y = yyjson_obj_get(position, "y");
        if (!id || !package_id || !version || !yyjson_is_num(x) || !yyjson_is_num(y) ||
            !nn_catalog_find(project->catalog, package_id, version) ||
            !nn_model_add_node(project->model, id, name ? name : package_id,
                               package_id, version, scope ? scope : "",
                               yyjson_get_num(x), yyjson_get_num(y), error, capacity)) {
            if (error && !error[0]) problem(error, capacity, "invalid node or undeclared package");
            return false;
        }
        yyjson_val *parameters = yyjson_obj_get(data, "params");
        if (!yyjson_is_obj(parameters)) { problem(error, capacity, "node params missing"); return false; }
        size_t parameter_index, parameter_max;
        yyjson_val *key, *value;
        yyjson_obj_foreach(parameters, parameter_index, parameter_max, key, value) {
            NNValue parsed;
            if (!parse_value(value, &parsed)) {
                problem(error, capacity, "unsupported parameter value"); return false;
            }
            bool set = nn_model_set_parameter(project->model, id, yyjson_get_str(key),
                                              &parsed, error, capacity);
            nn_value_dispose(&parsed);
            if (!set) return false;
        }
    }
    for (size_t i = 0; i < yyjson_arr_size(edges); ++i) {
        yyjson_val *edge = yyjson_arr_get(edges, i);
        const char *id = string_field(edge, "id");
        const char *source = string_field(edge, "source");
        const char *target = string_field(edge, "target");
        const char *source_handle = string_field(edge, "sourceHandle");
        const char *target_handle = string_field(edge, "targetHandle");
        if (!id || !source || !target || !source_handle || !target_handle ||
            !nn_model_connect(project->model, id, source, source_handle,
                              target, target_handle, error, capacity)) return false;
    }
    return true;
}

NNProject *nn_project_open(const char *directory, const char *core_root,
                           char *error, size_t capacity)
{
    if (error && capacity) error[0] = '\0';
    if (!directory || !core_root) { problem(error, capacity, "project path missing"); return NULL; }
    char *resolved = realpath(directory, NULL);
    if (!resolved) { problem(error, capacity, "project directory unavailable: %s", directory); return NULL; }
    char *model_path = join_path(resolved, "model.json");
    yyjson_doc *document = model_path ? read_document(model_path, error, capacity) : NULL;
    free(model_path);
    if (!document) { free(resolved); return NULL; }
    NNProject *project = calloc(1, sizeof(*project));
    if (!project) { yyjson_doc_free(document); free(resolved); return NULL; }
    project->directory = resolved;
    project->core_root = realpath(core_root, NULL);
    yyjson_val *root = yyjson_doc_get_root(document);
    yyjson_val *manifest = yyjson_obj_get(root, "manifest");
    const char *id = string_field(manifest, "id");
    const char *version = string_field(manifest, "version");
    const char *name = string_field(manifest, "name");
    const char *description = string_field(manifest, "description");
    bool okay = yyjson_get_int(yyjson_obj_get(manifest, "schemaVersion")) == 2 &&
                valid_id(id) && valid_id(version) && name && *name &&
                (project->id = copy_string(id)) &&
                (project->version = copy_string(version)) &&
                (project->name = copy_string(name)) &&
                (project->description = copy_string(description ? description : ""));
    const char *layout = string_field(root, "layoutDirection");
    project->layout_direction = copy_string(layout ? layout : "horizontal");
    if (!project->layout_direction || !project->core_root) okay = false;
    if (!okay) problem(error, capacity, "invalid schema-v2 project manifest");
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
        project->dataset_count = dataset_count;
        project->datasets = calloc(dataset_count ? dataset_count : 1, sizeof(*project->datasets));
        okay = project->datasets != NULL;
    }
    for (size_t i = 0; okay && i < dataset_count; ++i) {
        NNDataset *dataset = &project->datasets[i];
        dataset->id = copy_string(dataset_refs[i].id);
        dataset->version = copy_string(dataset_refs[i].version);
        dataset->path = copy_string(dataset_refs[i].path);
        okay = dataset->id && dataset->version && dataset->path &&
               load_dataset(project, dataset, error, capacity);
        for (size_t j = 0; okay && j < i; ++j)
            if (!strcmp(dataset->id, project->datasets[j].id) &&
                !strcmp(dataset->version, project->datasets[j].version)) {
                problem(error, capacity, "duplicate dataset identity"); okay = false;
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
        const char *selected_id = string_field(active, "id");
        const char *selected_version = string_field(active, "version");
        bool found = false;
        for (size_t i = 0; selected_id && selected_version && i < project->dataset_count; ++i)
            if (!strcmp(selected_id, project->datasets[i].id) &&
                !strcmp(selected_version, project->datasets[i].version)) found = true;
        okay = found && (project->active_dataset_id = copy_string(selected_id)) &&
               (project->active_dataset_version = copy_string(selected_version));
        if (!okay) problem(error, capacity, "active dataset is undeclared");
    }
    if (okay) okay = parse_graph(project, root, error, capacity);
    yyjson_doc_free(document);
    if (!okay) { nn_project_close(project); return NULL; }
    return project;
}

static bool add_string(yyjson_mut_doc *doc, yyjson_mut_val *object,
                       const char *key, const char *value)
{
    return value && yyjson_mut_obj_add_str(doc, object, key, value);
}

static yyjson_mut_val *write_value(yyjson_mut_doc *doc, const NNValue *value)
{
    switch (value->type) {
    case NN_VALUE_BOOL: return yyjson_mut_bool(doc, value->as.boolean);
    case NN_VALUE_INT: return yyjson_mut_sint(doc, value->as.integer);
    case NN_VALUE_REAL: return isfinite(value->as.real) ? yyjson_mut_real(doc, value->as.real) : NULL;
    case NN_VALUE_STRING: return value->as.string ? yyjson_mut_strcpy(doc, value->as.string) : NULL;
    case NN_VALUE_ARRAY: {
        if (value->as.array.count && !value->as.array.items) return NULL;
        yyjson_mut_val *array = yyjson_mut_arr(doc);
        if (!array) return NULL;
        for (size_t i = 0; i < value->as.array.count; ++i) {
            yyjson_mut_val *item = write_value(doc, &value->as.array.items[i]);
            if (!item || !yyjson_mut_arr_append(array, item)) return NULL;
        }
        return array;
    }
    default: return NULL;
    }
}

static bool add_value(yyjson_mut_doc *doc, yyjson_mut_val *object,
                      const char *key, const NNValue *value)
{
    yyjson_mut_val *serialized = write_value(doc, value);
    return serialized && yyjson_mut_obj_add_val(doc, object, key, serialized);
}

static bool write_project_document(NNProject *project, char **json, size_t *length)
{
    yyjson_mut_doc *doc = yyjson_mut_doc_new(NULL);
    if (!doc) return false;
    yyjson_mut_val *root = yyjson_mut_obj(doc);
    yyjson_mut_val *nodes = yyjson_mut_arr(doc), *edges = yyjson_mut_arr(doc);
    yyjson_mut_val *manifest = yyjson_mut_obj(doc);
    bool okay = root && nodes && edges && manifest &&
        yyjson_mut_obj_add_val(doc, root, "nodes", nodes) &&
        yyjson_mut_obj_add_val(doc, root, "edges", edges) &&
        add_string(doc, root, "layoutDirection", project->layout_direction ? project->layout_direction : "horizontal") &&
        yyjson_mut_obj_add_val(doc, root, "manifest", manifest) &&
        yyjson_mut_obj_add_int(doc, manifest, "schemaVersion", 2) &&
        add_string(doc, manifest, "id", project->id) &&
        add_string(doc, manifest, "version", project->version) &&
        add_string(doc, manifest, "name", project->name) &&
        add_string(doc, manifest, "description", project->description ? project->description : "");
    yyjson_mut_val *packages = okay ? yyjson_mut_arr(doc) : NULL;
    yyjson_mut_val *datasets = okay ? yyjson_mut_arr(doc) : NULL;
    okay = okay && packages && datasets &&
        yyjson_mut_obj_add_val(doc, manifest, "customPackages", packages) &&
        yyjson_mut_obj_add_val(doc, manifest, "customDatasets", datasets);
    for (size_t i = 0; okay && i < project->package_count; ++i) {
        const NNResourceRef *ref = &project->packages[i];
        yyjson_mut_val *item = yyjson_mut_obj(doc);
        okay = item && add_string(doc, item, "id", ref->id) &&
            add_string(doc, item, "version", ref->version) && add_string(doc, item, "path", ref->path) &&
            yyjson_mut_arr_append(packages, item);
    }
    for (size_t i = 0; okay && i < project->dataset_count; ++i) {
        const NNDataset *ref = &project->datasets[i];
        yyjson_mut_val *item = yyjson_mut_obj(doc);
        okay = item && add_string(doc, item, "id", ref->id) &&
            add_string(doc, item, "version", ref->version) && add_string(doc, item, "path", ref->path) &&
            yyjson_mut_arr_append(datasets, item);
    }
    if (okay && project->active_dataset_id) {
        yyjson_mut_val *active = yyjson_mut_obj(doc);
        okay = active && add_string(doc, active, "id", project->active_dataset_id) &&
            add_string(doc, active, "version", project->active_dataset_version) &&
            yyjson_mut_obj_add_val(doc, manifest, "activeDataset", active);
    }
    for (size_t i = 0; okay && i < nn_model_node_count(project->model); ++i) {
        const NNNode *node = nn_model_node_at(project->model, i);
        yyjson_mut_val *item = yyjson_mut_obj(doc), *position = yyjson_mut_obj(doc);
        yyjson_mut_val *data = yyjson_mut_obj(doc), *package = yyjson_mut_obj(doc);
        yyjson_mut_val *params = yyjson_mut_obj(doc);
        okay = item && position && data && package && params &&
            add_string(doc, item, "id", node->id) && add_string(doc, item, "type", "custom") &&
            yyjson_mut_obj_add_val(doc, item, "position", position) &&
            yyjson_mut_obj_add_double(doc, position, "x", node->x) &&
            yyjson_mut_obj_add_double(doc, position, "y", node->y) &&
            yyjson_mut_obj_add_val(doc, item, "data", data) &&
            add_string(doc, data, "name", node->label) &&
            add_string(doc, data, "scope", node->scope_id) &&
            yyjson_mut_obj_add_val(doc, data, "package", package) &&
            add_string(doc, package, "id", node->package_id) &&
            add_string(doc, package, "version", node->package_version) &&
            yyjson_mut_obj_add_val(doc, data, "params", params);
        for (size_t p = 0; okay && p < node->parameter_count; ++p)
            okay = add_value(doc, params, node->parameters[p].key, &node->parameters[p].value);
        if (okay) okay = yyjson_mut_arr_append(nodes, item);
    }
    for (size_t i = 0; okay && i < nn_model_edge_count(project->model); ++i) {
        const NNEdge *edge = nn_model_edge_at(project->model, i);
        yyjson_mut_val *item = yyjson_mut_obj(doc);
        okay = item && add_string(doc, item, "id", edge->id) &&
            add_string(doc, item, "source", edge->source_id) &&
            add_string(doc, item, "sourceHandle", edge->source_handle_id) &&
            add_string(doc, item, "target", edge->target_id) &&
            add_string(doc, item, "targetHandle", edge->target_handle_id) &&
            yyjson_mut_arr_append(edges, item);
    }
    if (okay) {
        yyjson_mut_doc_set_root(doc, root);
        *json = yyjson_mut_write_opts(doc, YYJSON_WRITE_PRETTY_TWO_SPACES, NULL, length, NULL);
        okay = *json != NULL;
        if (okay) {
            char *terminated = realloc(*json, *length + 2);
            if (!terminated) { free(*json); *json = NULL; okay = false; }
            else { terminated[(*length)++] = '\n'; terminated[*length] = '\0'; *json = terminated; }
        }
    }
    yyjson_mut_doc_free(doc);
    return okay;
}

static bool write_all(int fd, const char *data, size_t length)
{
    while (length) {
        ssize_t written = write(fd, data, length);
        if (written < 0 && errno == EINTR) continue;
        if (written <= 0) return false;
        data += written;
        length -= (size_t)written;
    }
    return true;
}

bool nn_project_save(NNProject *project, char *error, size_t capacity)
{
    if (error && capacity) error[0] = '\0';
    if (!project || !project->model) {
        problem(error, capacity, "no active project to save");
        return false;
    }
    char *json = NULL;
    size_t length = 0;
    if (!write_project_document(project, &json, &length)) {
        problem(error, capacity, "unable to serialize project");
        free(json);
        return false;
    }
    char *destination = join_path(project->directory, "model.json");
    char *temporary = join_path(project->directory, ".model.json.tmp.XXXXXX");
    if (!destination || !temporary) {
        problem(error, capacity, "out of memory");
        free(destination); free(temporary); free(json);
        return false;
    }
    int fd = mkstemp(temporary);
    bool okay = fd >= 0;
    if (!okay) problem(error, capacity, "cannot create project temporary file: %s", strerror(errno));
    if (okay && !write_all(fd, json, length)) {
        okay = false; problem(error, capacity, "cannot write project: %s", strerror(errno));
    }
    if (okay && fsync(fd)) {
        okay = false; problem(error, capacity, "cannot flush project: %s", strerror(errno));
    }
    if (fd >= 0 && close(fd) && okay) {
        okay = false; problem(error, capacity, "cannot close project temporary file: %s", strerror(errno));
    }
    if (okay && rename(temporary, destination)) {
        okay = false; problem(error, capacity, "cannot replace project file: %s", strerror(errno));
    }
    if (!okay) unlink(temporary);
    else project->dirty = false;
    free(destination); free(temporary); free(json);
    return okay;
}

static void dataset_dispose(NNDataset *dataset)
{
    if (!dataset) return;
    free(dataset->id); free(dataset->version); free(dataset->path); free(dataset->name);
    for (size_t i = 0; i < dataset->input_count; ++i) {
        free(dataset->inputs[i].name); free(dataset->inputs[i].dtype);
        nn_value_dispose(&dataset->inputs[i].shape);
    }
    for (size_t i = 0; i < dataset->target_count; ++i) {
        free(dataset->targets[i].name); free(dataset->targets[i].dtype);
        nn_value_dispose(&dataset->targets[i].shape);
    }
    free(dataset->inputs); free(dataset->targets);
}

void nn_project_close(NNProject *project)
{
    if (!project) return;
    nn_model_free(project->model);
    nn_catalog_free(project->catalog);
    for (size_t i = 0; i < project->package_count; ++i) {
        free((char *)project->packages[i].id);
        free((char *)project->packages[i].version);
        free((char *)project->packages[i].path);
    }
    for (size_t i = 0; i < project->dataset_count; ++i) dataset_dispose(&project->datasets[i]);
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

bool nn_project_select_dataset(NNProject *project, const char *id, const char *version,
                               char *error, size_t capacity)
{
    if (error && capacity) error[0] = '\0';
    if (!project || !id || !version) { problem(error, capacity, "invalid dataset identity"); return false; }
    const NNDataset *selected = NULL;
    for (size_t i = 0; i < project->dataset_count; ++i)
        if (!strcmp(project->datasets[i].id, id) && !strcmp(project->datasets[i].version, version)) selected = &project->datasets[i];
    if (!selected) { problem(error, capacity, "dataset identity is not declared"); return false; }
    char *new_id = copy_string(id), *new_version = copy_string(version);
    if (!new_id || !new_version) { free(new_id); free(new_version); problem(error, capacity, "out of memory"); return false; }
    char *old_id = project->active_dataset_id, *old_version = project->active_dataset_version;
    project->active_dataset_id = new_id;
    project->active_dataset_version = new_version;
    project->dirty = true;
    free(old_id); free(old_version);
    if (error && capacity) error[0] = '\0';
    return true;
}

static bool copy_file(const char *source, const char *destination)
{
    struct stat st;
    if (lstat(source, &st) || !S_ISREG(st.st_mode)) return false;
    int input = open(source, O_RDONLY | O_NOFOLLOW);
    if (input < 0) return false;
    int output = open(destination, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW, st.st_mode & 0777);
    bool okay = output >= 0;
    char buffer[16384];
    while (okay) {
        ssize_t count = read(input, buffer, sizeof(buffer));
        if (count < 0 && errno == EINTR) continue;
        if (count < 0) { okay = false; break; }
        if (!count) break;
        okay = write_all(output, buffer, (size_t)count);
    }
    if (output >= 0 && close(output)) okay = false;
    close(input);
    return okay;
}

static bool copy_tree(const char *source, const char *destination);

static bool copy_tree_contents(const char *source, const char *destination)
{
    struct stat st;
    if (lstat(source, &st) || !S_ISDIR(st.st_mode)) return false;
    DIR *dir = opendir(source);
    if (!dir) return false;
    bool okay = true;
    struct dirent *entry;
    while (okay && (entry = readdir(dir))) {
        if (!strcmp(entry->d_name, ".") || !strcmp(entry->d_name, "..")) continue;
        char *from = join_path(source, entry->d_name), *to = join_path(destination, entry->d_name);
        if (!from || !to || lstat(from, &st)) okay = false;
        else if (S_ISDIR(st.st_mode)) okay = copy_tree(from, to);
        else if (S_ISREG(st.st_mode)) okay = copy_file(from, to);
        else okay = false;
        free(from); free(to);
    }
    closedir(dir);
    return okay;
}

static bool copy_tree(const char *source, const char *destination)
{
    struct stat st;
    if (lstat(source, &st) || !S_ISDIR(st.st_mode) ||
        mkdir(destination, st.st_mode & 0777)) return false;
    return copy_tree_contents(source, destination);
}

static void remove_tree(const char *path)
{
    struct stat st;
    if (!path || lstat(path, &st)) return;
    if (S_ISDIR(st.st_mode)) {
        DIR *dir = opendir(path);
        if (dir) {
            struct dirent *entry;
            while ((entry = readdir(dir))) {
                if (strcmp(entry->d_name, ".") && strcmp(entry->d_name, "..")) {
                    char *child = join_path(path, entry->d_name);
                    if (child) { remove_tree(child); free(child); }
                }
            }
            closedir(dir);
        }
        rmdir(path);
    } else unlink(path);
}

static char *repo_root_from_core(const char *core_root)
{
    char *path = realpath(core_root, NULL);
    if (!path) return NULL;
    for (int i = 0; i < 2; ++i) {
        char *slash = strrchr(path, '/');
        if (!slash || slash == path) { free(path); return NULL; }
        *slash = '\0';
    }
    return path;
}

NNProject *nn_project_create(const char *parent, const char *id, const char *name,
                             bool mnist_template, const char *core_root,
                             char *error, size_t capacity)
{
    if (error && capacity) error[0] = '\0';
    if (!parent || !valid_id(id) || !name || !*name || !core_root) {
        problem(error, capacity, "invalid project identity or parent"); return NULL;
    }
    struct stat parent_stat;
    if (stat(parent, &parent_stat) || !S_ISDIR(parent_stat.st_mode)) {
        problem(error, capacity, "project parent directory unavailable"); return NULL;
    }
    char *directory = join_path(parent, id);
    if (!directory) { problem(error, capacity, "out of memory"); return NULL; }
    if (mkdir(directory, 0755)) {
        problem(error, capacity, "cannot create project directory: %s", strerror(errno));
        free(directory); return NULL;
    }
    bool okay = true;
    if (mnist_template) {
        char *repo = repo_root_from_core(core_root);
        char *template_root = repo ? join_path(repo, "examples/mnist-mlp") : NULL;
        char *source_model = template_root ? join_path(template_root, "model.json") : NULL;
        char *destination_model = join_path(directory, "model.json");
        char *source_dataset = template_root ? join_path(template_root, "datasets/mnist") : NULL;
        char *destination_dataset = join_path(directory, "datasets/mnist");
        char *destination_datasets = join_path(directory, "datasets");
        okay = source_model && destination_model && source_dataset && destination_dataset &&
            destination_datasets && copy_file(source_model, destination_model) &&
            mkdir(destination_datasets, 0755) == 0 && copy_tree(source_dataset, destination_dataset);
        free(repo); free(template_root); free(source_model); free(destination_model);
        free(source_dataset); free(destination_dataset); free(destination_datasets);
    } else {
        NNProject seed = { .directory = directory, .id = (char *)id, .version = "0.1.0",
            .name = (char *)name, .description = "", .layout_direction = "horizontal",
            .model = nn_model_new() };
        okay = seed.model != NULL;
        char *json = NULL; size_t length = 0;
        if (okay) okay = write_project_document(&seed, &json, &length);
        char *path = join_path(directory, "model.json");
        int fd = path && okay ? open(path, O_WRONLY | O_CREAT | O_EXCL, 0644) : -1;
        if (fd < 0) okay = false;
        if (okay) okay = write_all(fd, json, length) && fsync(fd) == 0;
        if (fd >= 0 && close(fd)) okay = false;
        free(path); free(json); nn_model_free(seed.model);
    }
    if (!okay) {
        problem(error, capacity, "cannot create project from requested template");
        remove_tree(directory); free(directory); return NULL;
    }
    NNProject *project = nn_project_open(directory, core_root, error, capacity);
    free(directory);
    if (!project) {
        /* Creation is one transaction: a bad template or package scope leaves no child. */
        char *failed = join_path(parent, id);
        if (failed) { remove_tree(failed); free(failed); }
        return NULL;
    }
    if (strcmp(project->id, id) || strcmp(project->name, name)) {
        char *new_id = copy_string(id), *new_name = copy_string(name);
        if (!new_id || !new_name) {
            free(new_id); free(new_name); nn_project_close(project);
            char *failed = join_path(parent, id);
            if (failed) { remove_tree(failed); free(failed); }
            problem(error, capacity, "out of memory"); return NULL;
        }
        free(project->id); free(project->name);
        project->id = new_id; project->name = new_name;
        project->dirty = true;
        if (!nn_project_save(project, error, capacity)) {
            nn_project_close(project);
            char *failed = join_path(parent, id);
            if (failed) { remove_tree(failed); free(failed); }
            return NULL;
        }
    }
    return project;
}

static bool write_new_text(const char *path, const char *text)
{
    int fd = open(path, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW, 0644);
    if (fd < 0) return false;
    size_t length = strlen(text);
    bool okay = write_all(fd, text, length) && fsync(fd) == 0;
    if (close(fd)) okay = false;
    return okay;
}

static char *resource_manifest(const char *id, const char *version,
                               const char *definition, const char *lua,
                               const char *dependencies)
{
    yyjson_doc *deps_doc = dependencies ? yyjson_read(dependencies, strlen(dependencies), 0) : NULL;
    yyjson_mut_doc *doc = yyjson_mut_doc_new(NULL);
    if (!doc || !deps_doc || !yyjson_is_obj(yyjson_doc_get_root(deps_doc))) {
        if (doc) yyjson_mut_doc_free(doc);
        if (deps_doc) yyjson_doc_free(deps_doc);
        return NULL;
    }
    yyjson_mut_val *root = yyjson_mut_obj(doc), *entry = yyjson_mut_obj(doc), *inf = yyjson_mut_obj(doc);
    yyjson_mut_val *deps = yyjson_val_mut_copy(doc, yyjson_doc_get_root(deps_doc));
    bool okay = root && entry && inf && deps &&
        yyjson_mut_obj_add_int(doc, root, "schemaVersion", 1) &&
        yyjson_mut_obj_add_strcpy(doc, root, "id", id) &&
        yyjson_mut_obj_add_strcpy(doc, root, "version", version) &&
        yyjson_mut_obj_add_val(doc, root, "dependencies", deps) &&
        yyjson_mut_obj_add_val(doc, root, "entrypoints", entry) &&
        yyjson_mut_obj_add_strcpy(doc, entry, "definition", definition) &&
        yyjson_mut_obj_add_val(doc, entry, "inference", inf) &&
        yyjson_mut_obj_add_strcpy(doc, inf, "language", "lua") &&
        yyjson_mut_obj_add_strcpy(doc, inf, "file", lua);
    char *json = NULL;
    size_t length = 0;
    if (okay) { yyjson_mut_doc_set_root(doc, root); json = yyjson_mut_write_opts(doc, YYJSON_WRITE_PRETTY_TWO_SPACES, NULL, &length, NULL); }
    if (json) { char *grown = realloc(json, length + 2); if (!grown) { free(json); json = NULL; } else { grown[length++] = '\n'; grown[length] = 0; json = grown; } }
    yyjson_mut_doc_free(doc); yyjson_doc_free(deps_doc);
    return json;
}

static char *dataset_manifest(const char *id, const char *version)
{
    yyjson_mut_doc *doc = yyjson_mut_doc_new(NULL);
    if (!doc) return NULL;
    yyjson_mut_val *root = yyjson_mut_obj(doc), *entry = yyjson_mut_obj(doc);
    bool okay = root && entry && yyjson_mut_obj_add_int(doc, root, "schemaVersion", 1) &&
        yyjson_mut_obj_add_strcpy(doc, root, "id", id) &&
        yyjson_mut_obj_add_strcpy(doc, root, "version", version) &&
        yyjson_mut_obj_add_val(doc, root, "entrypoints", entry) &&
        yyjson_mut_obj_add_strcpy(doc, entry, "definition", "dataset.json");
    char *json = NULL; size_t length = 0;
    if (okay) { yyjson_mut_doc_set_root(doc, root); json = yyjson_mut_write_opts(doc, YYJSON_WRITE_PRETTY_TWO_SPACES, NULL, &length, NULL); }
    if (json) { char *grown = realloc(json, length + 2); if (!grown) { free(json); json = NULL; } else { grown[length++] = '\n'; grown[length] = 0; json = grown; } }
    yyjson_mut_doc_free(doc); return json;
}

static char *pretty_json(const char *source)
{
    if (!source) return NULL;
    yyjson_doc *doc = yyjson_read(source, strlen(source), 0);
    if (!doc) return NULL;
    size_t length = 0;
    char *text = yyjson_val_write_opts(yyjson_doc_get_root(doc), YYJSON_WRITE_PRETTY_TWO_SPACES,
                                       NULL, &length, NULL);
    yyjson_doc_free(doc);
    if (!text) return NULL;
    char *grown = realloc(text, length + 2);
    if (!grown) { free(text); return NULL; }
    grown[length++] = '\n'; grown[length] = '\0';
    return grown;
}

static bool make_resource_directory(const char *root, const char *category,
                                    const char *name, char **relative, char **absolute,
                                    bool *category_created)
{
    *relative = NULL; *absolute = NULL;
    *category_created = false;
    char *base = join_path(root, category);
    if (!base) return false;
    struct stat st;
    if (lstat(base, &st)) {
        if (errno != ENOENT || mkdir(base, 0755)) { free(base); return false; }
        *category_created = true;
    } else if (!S_ISDIR(st.st_mode)) { free(base); return false; }
    size_t n = strlen(name) + 1;
    char *rel = malloc(strlen(category) + n + 1);
    if (rel) snprintf(rel, strlen(category) + n + 1, "%s/%s", category, name);
    char *dir = rel ? join_path(root, rel) : NULL;
    free(base);
    if (!rel || !dir) {
        free(rel); free(dir);
        if (*category_created) { char *created = join_path(root, category); if (created) { rmdir(created); free(created); } }
        return false;
    }
    if (mkdir(dir, 0755)) {
        free(rel); free(dir);
        if (*category_created) { char *created = join_path(root, category); if (created) { rmdir(created); free(created); } }
        return false;
    }
    *relative = rel; *absolute = dir; return true;
}

bool nn_project_create_stereotype(NNProject *p, const char *id, const char *version,
                                  const char *definition_json, const char *lua,
                                  const char *dependencies_json, char *error, size_t cap)
{
    if (error && cap) error[0] = '\0';
    if (!p || !valid_id(id) || !version || !definition_json || !lua || strlen(definition_json) > 4u * 1024u * 1024u ||
        (dependencies_json && strlen(dependencies_json) > 4u * 1024u * 1024u) || strlen(lua) > 1024 * 1024 ||
        !valid_semver(version)) { problem(error, cap, "invalid stereotype identity or payload"); return false; }
    if (p->package_count >= 128) { problem(error, cap, "package catalog limit reached"); return false; }
    for (size_t i = 0; i < p->package_count; ++i) if (!strcmp(p->packages[i].id, id) && !strcmp(p->packages[i].version, version)) { problem(error, cap, "duplicate package identity"); return false; }
    for (size_t i = 0; i < nn_catalog_count(p->catalog); ++i) { const NNPackage *x = nn_catalog_at(p->catalog, i); if (!strcmp(x->id, id) && !strcmp(x->version, version)) { problem(error, cap, "core package identity is immutable"); return false; } }
    char name[256]; if (snprintf(name, sizeof(name), "%s-%s", id, version) >= (int)sizeof(name)) { problem(error, cap, "resource path too long"); return false; }
    char *rel = NULL, *dir = NULL, *definition = pretty_json(definition_json); bool category_created = false;
    char *manifest = resource_manifest(id, version, "definition.json", "inference.lua", dependencies_json ? dependencies_json : "{}");
    if ((definition && strlen(definition) > 4u * 1024u * 1024u) ||
        (manifest && strlen(manifest) > 4u * 1024u * 1024u)) {
        free(manifest); free(definition); problem(error, cap, "package JSON exceeds catalog file limit"); return false;
    }
    if (!manifest || !definition || !make_resource_directory(p->directory, "packages", name, &rel, &dir, &category_created)) { free(manifest); free(definition); free(rel); free(dir); problem(error, cap, "cannot create package resource directory"); return false; }
    char *mp = join_path(dir, "manifest.json"), *dp = join_path(dir, "definition.json"), *lp = join_path(dir, "inference.lua");
    bool okay = mp && dp && lp && write_new_text(mp, manifest) && write_new_text(dp, definition) && write_new_text(lp, lua);
    free(manifest); free(definition); free(mp); free(dp); free(lp);
    NNResourceRef *refs = NULL; NNCatalog *candidate = NULL;
    if (okay) {
        refs = calloc(p->package_count + 1, sizeof(*refs)); okay = refs != NULL;
        for (size_t i = 0; okay && i < p->package_count; ++i) {
            refs[i] = (NNResourceRef){copy_string(p->packages[i].id), copy_string(p->packages[i].version), copy_string(p->packages[i].path)};
            okay = refs[i].id && refs[i].version && refs[i].path;
        }
        if (okay) refs[p->package_count] = (NNResourceRef){copy_string(id), copy_string(version), copy_string(rel)};
        if (okay) okay = refs[p->package_count].id && refs[p->package_count].version && refs[p->package_count].path;
        if (okay) candidate = nn_catalog_load(p->core_root, p->directory, refs, p->package_count + 1, error, cap);
        okay = okay && candidate != NULL;
    }
    for (size_t i = 0; okay && i < nn_model_node_count(p->model); ++i) {
        const NNNode *node = nn_model_node_at(p->model, i);
        if (!nn_catalog_find(candidate, node->package_id, node->package_version)) { problem(error, cap, "candidate catalog invalidates graph package"); okay = false; }
    }
    if (okay) {
        NNResourceRef *old_refs = p->packages; size_t old_count = p->package_count;
        NNCatalog *old_catalog = p->catalog; bool old_dirty = p->dirty;
        p->packages = refs; p->package_count++; p->catalog = candidate;
        p->dirty = true;
        okay = nn_project_save(p, error, cap);
        if (okay) { for (size_t i = 0; i < old_count; ++i) { free((char *)old_refs[i].id); free((char *)old_refs[i].version); free((char *)old_refs[i].path); } free(old_refs); nn_catalog_free(old_catalog); refs = NULL; candidate = NULL; }
        else { p->packages = old_refs; p->package_count--; p->catalog = old_catalog; p->dirty = old_dirty; }
    }
    if (refs) { for (size_t i = 0; i <= p->package_count; ++i) { free((char *)refs[i].id); free((char *)refs[i].version); free((char *)refs[i].path); } free(refs); }
    nn_catalog_free(candidate); if (!okay) { remove_tree(dir); if (category_created) { char *base = join_path(p->directory, "packages"); if (base) { rmdir(base); free(base); } } }
    free(rel); free(dir);
    if (!okay && error && cap && !error[0]) problem(error, cap, "stereotype creation failed");
    return okay;
}

bool nn_project_create_dataset(NNProject *p, const char *id, const char *version,
                               const char *definition_json, bool select,
                               char *error, size_t cap)
{
    if (error && cap) error[0] = '\0';
    if (!p || !valid_id(id) || !valid_semver(version) || !definition_json) { problem(error, cap, "invalid dataset identity or payload"); return false; }
    if (p->dataset_count >= 128) { problem(error, cap, "dataset catalog limit reached"); return false; }
    for (size_t i = 0; i < p->dataset_count; ++i) if (!strcmp(p->datasets[i].id, id) && !strcmp(p->datasets[i].version, version)) { problem(error, cap, "duplicate dataset identity"); return false; }
    char name[256]; if (snprintf(name, sizeof(name), "%s-%s", id, version) >= (int)sizeof(name)) { problem(error, cap, "resource path too long"); return false; }
    if (strlen(definition_json) > 16u * 1024u * 1024u) { problem(error, cap, "dataset JSON exceeds resource file limit"); return false; }
    char *rel = NULL, *dir = NULL, *definition = pretty_json(definition_json), *manifest = dataset_manifest(id, version); bool category_created = false;
    if (definition && strlen(definition) > 16u * 1024u * 1024u) {
        free(manifest); free(definition); problem(error, cap, "dataset JSON exceeds resource file limit"); return false;
    }
    if (!manifest || !definition || !make_resource_directory(p->directory, "datasets", name, &rel, &dir, &category_created)) { free(manifest); free(definition); free(rel); free(dir); problem(error, cap, "cannot create dataset resource directory"); return false; }
    char *mp = join_path(dir, "manifest.json"), *dp = join_path(dir, "dataset.json");
    bool okay = mp && dp && write_new_text(mp, manifest) && write_new_text(dp, definition);
    free(manifest); free(definition); free(mp); free(dp);
    NNDataset candidate = { .id = copy_string(id), .version = copy_string(version), .path = copy_string(rel) };
    if (okay) okay = candidate.id && candidate.version && candidate.path && load_dataset(p, &candidate, error, cap);
    NNDataset *grown = okay ? malloc((p->dataset_count + 1) * sizeof(*grown)) : NULL;
    if (okay && !grown) okay = false;
    if (okay) {
        NNDataset *old = p->datasets; size_t old_count = p->dataset_count;
        if (old_count) memcpy(grown, old, old_count * sizeof(*grown));
        char *old_id = p->active_dataset_id, *old_version = p->active_dataset_version; bool old_dirty = p->dirty;
        p->datasets = grown; p->datasets[p->dataset_count++] = candidate; memset(&candidate, 0, sizeof(candidate));
        if (select) { p->active_dataset_id = copy_string(id); p->active_dataset_version = copy_string(version); if (!p->active_dataset_id || !p->active_dataset_version) okay = false; }
        p->dirty = true;
        if (okay) okay = nn_project_save(p, error, cap);
        if (okay) { free(old); if (select) { free(old_id); free(old_version); } }
        else {
            if (select) { free(p->active_dataset_id); free(p->active_dataset_version); p->active_dataset_id = old_id; p->active_dataset_version = old_version; }
            p->dataset_count = old_count; p->datasets = old; p->dirty = old_dirty;
            dataset_dispose(&grown[old_count]);
            free(grown);
        }
    }
    dataset_dispose(&candidate);
    if (!okay) { remove_tree(dir); if (category_created) { char *base = join_path(p->directory, "datasets"); if (base) { rmdir(base); free(base); } } }
    free(rel); free(dir);
    if (!okay && error && cap && !error[0]) problem(error, cap, "dataset creation failed");
    return okay;
}

bool nn_project_create_vae(const char *parent, const char *id, const char *name,
                           const char *core_root, NNProject **result,
                           char *error, size_t cap)
{
    if (!result) { problem(error, cap, "invalid project result"); return false; }
    *result = NULL;
    if (!parent || !valid_id(id) || !name || !*name || !core_root) { problem(error, cap, "invalid project identity"); return false; }
    struct stat st; if (lstat(parent, &st) || !S_ISDIR(st.st_mode)) { problem(error, cap, "project parent unavailable or symlinked"); return false; }
    char *root = repo_root_from_core(core_root), *template = root ? join_path(root, "examples/mnist-vae") : NULL;
    char *destination = join_path(parent, id);
    bool destination_owned = template && destination && mkdir(destination, 0755) == 0;
    bool okay = destination_owned && copy_tree_contents(template, destination);
    free(root); free(template);
    if (!okay) { if (destination_owned) remove_tree(destination); free(destination); problem(error, cap, "cannot copy MNIST VAE template"); return false; }
    NNProject *project = nn_project_open(destination, core_root, error, cap);
    free(destination);
    if (!project) { char *failed = join_path(parent, id); if (failed) { remove_tree(failed); free(failed); } return false; }
    char *new_name = copy_string(name), *new_id = copy_string(id);
    if (!new_name || !new_id) { free(new_name); free(new_id); nn_project_close(project); char *failed = join_path(parent, id); if (failed) { remove_tree(failed); free(failed); } problem(error, cap, "out of memory"); return false; }
    free(project->name); free(project->id); project->name = new_name; project->id = new_id; project->dirty = true;
    if (!nn_project_save(project, error, cap)) { nn_project_close(project); char *failed = join_path(parent, id); if (failed) { remove_tree(failed); free(failed); } return false; }
    *result = project; return true;
}
