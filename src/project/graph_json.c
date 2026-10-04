#include "project_internal.h"
#include "catalog/catalog.h"
#include "utils/utils.h"

#include <stdlib.h>
#include <string.h>
#include <limits.h>

static bool validate_stereotype_references(NNProject *project, const NNNode *node,
                                           char *error, size_t capacity)
{
    const NNPackage *package = nn_catalog_find(project->catalog, node->package_id,
                                                node->package_version);
    if (!package) return false;
    for (size_t i = 0; i < node->parameter_count; ++i) {
        const NNParameterDef *definition = nn_catalog_package_parameter(
            package, node->parameters[i].key);
        if (!definition || strcmp(definition->type, "stereotype")) continue;
        NNPackage one = { .parameters = definition, .parameter_count = 1 };
        NNParameter *effective = NULL;
        size_t count = 0;
        bool okay = nn_catalog_parameters(project->catalog, &one,
            &node->parameters[i], 1, &effective, &count, error, capacity);
        nn_catalog_parameters_free(effective, count);
        if (!okay) return false;
    }
    return true;
}

static bool parse_value(yyjson_val *source, NNValue *target, unsigned depth)
{
    if (depth > 64) return false;
    memset(target, 0, sizeof(*target));
    if (yyjson_is_bool(source)) {
        target->type = NN_VALUE_BOOL;
        target->as.boolean = yyjson_get_bool(source);
    } else if (yyjson_is_int(source)) {
        if (yyjson_is_uint(source) && yyjson_get_uint(source) > LLONG_MAX) return false;
        target->type = NN_VALUE_INT;
        target->as.integer = yyjson_get_sint(source);
    } else if (yyjson_is_real(source)) {
        target->type = NN_VALUE_REAL;
        target->as.real = yyjson_get_real(source);
    } else if (yyjson_is_str(source)) {
        if (strlen(yyjson_get_str(source)) != yyjson_get_len(source)) return false;
        target->type = NN_VALUE_STRING;
        target->as.string = nn_text_copy(yyjson_get_str(source));
        if (!target->as.string) return false;
    } else if (yyjson_is_arr(source)) {
        target->type = NN_VALUE_ARRAY;
        size_t count = yyjson_arr_size(source);
        if (count > 1024) return false;
        target->as.array.items = calloc(count ? count : 1,
                                        sizeof(NNValue));
        if (!target->as.array.items) return false;
        target->as.array.count = count;
        for (size_t i = 0; i < target->as.array.count; ++i)
            if (!parse_value(yyjson_arr_get(source, i), &target->as.array.items[i], depth + 1)) {
                nn_value_dispose(target);
                return false;
            }
    } else if (yyjson_is_obj(source)) {
        size_t count = yyjson_obj_size(source);
        if (count > 1024) return false;
        target->type = NN_VALUE_OBJECT;
        target->as.object.items = calloc(count ? count : 1, sizeof(NNParameter));
        if (!target->as.object.items) return false;
        yyjson_obj_iter iter = yyjson_obj_iter_with(source);
        yyjson_val *key;
        while ((key = yyjson_obj_iter_next(&iter))) {
            size_t index = target->as.object.count;
            const char *text = yyjson_get_str(key);
            if (!text || !text[0] || strlen(text) != yyjson_get_len(key)) {
                nn_value_dispose(target); return false;
            }
            for (size_t i = 0; i < index; ++i)
                if (!strcmp(target->as.object.items[i].key, text)) {
                    nn_value_dispose(target);
                    return false;
                }
            target->as.object.items[index].key = nn_text_copy(text);
            if (!target->as.object.items[index].key ||
                !parse_value(yyjson_obj_iter_get_val(key),
                             &target->as.object.items[index].value, depth + 1)) {
                target->as.object.count = index + 1;
                nn_value_dispose(target);
                return false;
            }
            target->as.object.count = index + 1;
        }
    } else return false;
    return true;
}

bool nn_project_parse_value(yyjson_val *source, NNValue *target)
{
    if (!source || !target) return false;
    return parse_value(source, target, 0);
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
    const NNOutputDef *output = nn_catalog_package_output(source_package,
                                                          edge->source_handle_id);
    if (!source || !target || !source_package || !target_package || !output ||
        !output->type) {
        nn_error_set(error, capacity, "edge references an invalid output handle or package");
        return false;
    }
    if (!nn_catalog_package_input_handle_valid(target_package,
                                               edge->target_handle_id)) {
        nn_error_set(error, capacity, "edge references an invalid input handle");
        return false;
    }
    if (nn_catalog_package_is_kind(target_package, "output") &&
        strcmp(output->type, "output")) {
        nn_error_set(error, capacity, "output type is incompatible with output terminal");
        return false;
    }
    if (nn_catalog_package_is_kind(target_package, "loss-output") &&
        strcmp(output->type, "loss")) {
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
        if (!validate_stereotype_references(project,
                nn_model_find_node(project->model, id), error, capacity))
            return false;
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
