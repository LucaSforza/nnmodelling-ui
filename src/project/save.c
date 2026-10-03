#define _POSIX_C_SOURCE 200809L
#include "project_internal.h"
#include "utils/utils.h"

#include <errno.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

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

bool nn_project_write_project_document(NNProject *project, char **json, size_t *length)
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
            yyjson_mut_obj_add_int(doc, position, "x", node->x) &&
            yyjson_mut_obj_add_int(doc, position, "y", node->y) &&
            yyjson_mut_obj_add_val(doc, item, "data", data) &&
            add_string(doc, data, "name", node->label) &&
            add_string(doc, data, "scope", node->scope_id) &&
            yyjson_mut_obj_add_val(doc, data, "package", package) &&
            add_string(doc, package, "id", node->package_id) &&
            add_string(doc, package, "version", node->package_version) &&
            yyjson_mut_obj_add_val(doc, data, "params", params);
        if (okay && node->boundary_handle_id)
            okay = add_string(doc, data, "boundaryHandle", node->boundary_handle_id);
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

bool nn_project_save(NNProject *project, char *error, size_t capacity)
{
    if (error && capacity) error[0] = '\0';
    if (!project || !project->model) {
        nn_errorf(error, capacity, "no active project to save");
        return false;
    }
    char *json = NULL;
    size_t length = 0;
    if (!nn_project_write_project_document(project, &json, &length)) {
        nn_errorf(error, capacity, "unable to serialize project");
        free(json);
        return false;
    }
    char *destination = nn_path_join(project->directory, "model.json");
    char *temporary = nn_path_join(project->directory, ".model.json.tmp.XXXXXX");
    if (!destination || !temporary) {
        nn_errorf(error, capacity, "out of memory");
        free(destination); free(temporary); free(json);
        return false;
    }
    int fd = mkstemp(temporary);
    bool okay = fd >= 0;
    if (!okay) nn_errorf(error, capacity, "cannot create project temporary file: %s", strerror(errno));
    if (okay && !nn_project_write_all(fd, json, length)) {
        okay = false; nn_errorf(error, capacity, "cannot write project: %s", strerror(errno));
    }
    if (okay && fsync(fd)) {
        okay = false; nn_errorf(error, capacity, "cannot flush project: %s", strerror(errno));
    }
    if (fd >= 0 && close(fd) && okay) {
        okay = false; nn_errorf(error, capacity, "cannot close project temporary file: %s", strerror(errno));
    }
    if (okay && rename(temporary, destination)) {
        okay = false; nn_errorf(error, capacity, "cannot replace project file: %s", strerror(errno));
    }
    if (!okay) unlink(temporary);
    else project->dirty = false;
    free(destination); free(temporary); free(json);
    return okay;
}
