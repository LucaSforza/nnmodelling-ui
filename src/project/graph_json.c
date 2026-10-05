#include "project_internal.h"
#include "model/value_json.h"
#include "utils/utils.h"

#include <stdlib.h>
#include <string.h>

bool nn_project_parse_value(yyjson_val *source, NNValue *target)
{
    return nn_value_from_json(source, target);
}

bool nn_project_edge_topology_valid(const NNModel *model, const NNCatalog *catalog,
                                const NNEdge *edge, char *error, size_t capacity)
{
    const NNNode *source = nn_model_find_node(model, edge->source_id);
    const NNNode *target = nn_model_find_node(model, edge->target_id);
    const NNPackage *source_package = source
        ? nn_catalog_find(catalog, source->package_id, source->package_version) : NULL;
    const NNPackage *target_package = target
        ? nn_catalog_find(catalog, target->package_id, target->package_version) : NULL;
    const char *type = NULL;
    if (source_package)
        for (size_t i = 0; i < source_package->output_count; ++i)
            if (!strcmp(source_package->outputs[i].id, edge->source_handle_id))
                type = source_package->outputs[i].type;
    if (!source || !target || !source_package || !target_package || !type) {
        nn_error_set(error, capacity, "edge references an invalid output handle or package");
        return false;
    }
    bool input_valid;
    if (target_package->kind && !strcmp(target_package->kind, "input"))
        input_valid = false;
    else if (target_package->kind && !strcmp(target_package->kind, "join")) {
        size_t order;
        input_valid = nn_join_handle_order(edge->target_handle_id, &order);
    } else input_valid = !strcmp(edge->target_handle_id, "in");
    if (!input_valid) {
        nn_error_set(error, capacity, "edge references an invalid input handle");
        return false;
    }
    if (target_package->kind && !strcmp(target_package->kind, "output") &&
        strcmp(type, "output")) {
        nn_error_set(error, capacity, "output type is incompatible with output terminal");
        return false;
    }
    if (target_package->kind && !strcmp(target_package->kind, "loss-output") &&
        strcmp(type, "loss")) {
        nn_error_set(error, capacity, "output type is incompatible with loss terminal");
        return false;
    }
    return true;
}

bool nn_project_parse_graph(NNProject *project, yyjson_val *root,
                        char *error, size_t capacity)
{
    yyjson_val *nodes = yyjson_obj_get(root, "nodes"), *edges = yyjson_obj_get(root, "edges");
    if (!yyjson_is_arr(nodes) || !yyjson_is_arr(edges) ||
        yyjson_arr_size(nodes) > 10000 || yyjson_arr_size(edges) > 20000) {
        nn_errorf(error, capacity, "invalid graph arrays"); return false;
    }
    project->model = nn_model_new();
    if (!project->model) return nn_fail(error, capacity, "out of memory creating model");
    for (size_t i = 0; i < yyjson_arr_size(nodes); ++i) {
        yyjson_val *node = yyjson_arr_get(nodes, i);
        yyjson_val *position = yyjson_obj_get(node, "position");
        yyjson_val *data = yyjson_obj_get(node, "data");
        yyjson_val *package = yyjson_obj_get(data, "package");
        const char *id = nn_project_string_field(node, "id");
        const char *name = nn_project_string_field(data, "name");
        const char *package_id = nn_project_string_field(package, "id");
        const char *version = nn_project_string_field(package, "version");
        const char *scope = nn_project_string_field(data, "scope");
        yyjson_val *x = yyjson_obj_get(position, "x"), *y = yyjson_obj_get(position, "y");
        if (!id || !package_id || !version || !yyjson_is_num(x) || !yyjson_is_num(y) ||
            !nn_catalog_find(project->catalog, package_id, version) ||
            !nn_model_add_node(project->model, id, name ? name : package_id,
                               package_id, version, scope ? scope : "",
                               yyjson_get_num(x), yyjson_get_num(y), error, capacity)) {
            if (error && capacity && !error[0]) nn_errorf(error, capacity, "invalid node or undeclared package");
            return false;
        }
        yyjson_val *boundary = yyjson_obj_get(data, "boundaryHandle");
        if (boundary) {
            const char *handle = yyjson_get_str(boundary);
            if (!handle || strlen(handle) != yyjson_get_len(boundary) || !*handle ||
                !nn_model_set_boundary_handle(project->model, id, handle,
                                              error, capacity)) {
                if (error && capacity && !error[0])
                    nn_errorf(error, capacity, "invalid boundary handle mapping");
                return false;
            }
        }
        yyjson_val *parameters = yyjson_obj_get(data, "params");
        if (!yyjson_is_obj(parameters)) { nn_errorf(error, capacity, "node params missing"); return false; }
        size_t parameter_index, parameter_max;
        yyjson_val *key, *value;
        yyjson_obj_foreach(parameters, parameter_index, parameter_max, key, value) {
            NNValue parsed;
            if (!nn_project_parse_value(value, &parsed)) {
                nn_errorf(error, capacity, "unsupported parameter value"); return false;
            }
            bool set = nn_model_set_parameter(project->model, id, yyjson_get_str(key),
                                              &parsed, error, capacity);
            nn_value_dispose(&parsed);
            if (!set) return false;
        }
    }
    for (size_t i = 0; i < yyjson_arr_size(edges); ++i) {
        yyjson_val *edge = yyjson_arr_get(edges, i);
        const char *id = nn_project_string_field(edge, "id");
        const char *source = nn_project_string_field(edge, "source");
        const char *target = nn_project_string_field(edge, "target");
        const char *source_handle = nn_project_string_field(edge, "sourceHandle");
        const char *target_handle = nn_project_string_field(edge, "targetHandle");
        if (!id || !source || !target || !source_handle || !target_handle ||
            !nn_model_connect(project->model, id, source, source_handle,
                              target, target_handle, error, capacity)) return false;
        NNEdge candidate = { .id = (char *)id, .source_id = (char *)source,
            .source_handle_id = (char *)source_handle, .target_id = (char *)target,
            .target_handle_id = (char *)target_handle };
        if (!nn_project_edge_topology_valid(project->model, project->catalog, &candidate,
                                 error, capacity)) return false;
    }
    return true;
}
