#include "automation_internal.h"
#include "automation_utils.h"

static yyjson_mut_val *snapshot_value(yyjson_mut_doc *doc, const NNValue *value)
{
    switch (value->type) {
    case NN_VALUE_BOOL: return yyjson_mut_bool(doc, value->as.boolean);
    case NN_VALUE_INT: return yyjson_mut_sint(doc, value->as.integer);
    case NN_VALUE_REAL: return yyjson_mut_real(doc, value->as.real);
    case NN_VALUE_STRING: return yyjson_mut_strcpy(doc, value->as.string);
    case NN_VALUE_ARRAY: {
        yyjson_mut_val *array = yyjson_mut_arr(doc);
        if (!array) return NULL;
        for (size_t i = 0; i < value->as.array.count; ++i) {
            yyjson_mut_val *item = snapshot_value(doc, &value->as.array.items[i]);
            if (!item || !yyjson_mut_arr_append(array, item)) return NULL;
        }
        return array;
    }
    case NN_VALUE_OBJECT: {
        yyjson_mut_val *object = yyjson_mut_obj(doc);
        if (!object) return NULL;
        for (size_t i = 0; i < value->as.object.count; ++i) {
            const NNParameter *entry = &value->as.object.items[i];
            yyjson_mut_val *item = snapshot_value(doc, &entry->value);
            if (!item || !yyjson_mut_obj_add_val(doc, object, entry->key, item)) return NULL;
        }
        return object;
    }
    default: return yyjson_mut_null(doc);
    }
}

static yyjson_mut_val *slots_json(yyjson_mut_doc *doc, const NNTensorSlot *slots, size_t count)
{
    yyjson_mut_val *array = yyjson_mut_arr(doc);
    if (!array) return NULL;
    for (size_t i = 0; i < count; ++i) {
        yyjson_mut_val *slot = yyjson_mut_obj(doc);
        yyjson_mut_val *shape = snapshot_value(doc, &slots[i].shape);
        if (!slot || !shape || !nn_automation_json_string(doc, slot, "name", slots[i].name) ||
            !nn_automation_json_string(doc, slot, "dtype", slots[i].dtype) ||
            !yyjson_mut_obj_add_val(doc, slot, "shape", shape) ||
            !yyjson_mut_arr_append(array, slot)) return NULL;
    }
    return array;
}

yyjson_mut_val *nn_automation_snapshot(yyjson_mut_doc *doc, NNApplication *app)
{
    const NNProject *project = nn_app_project(app);
    if (!project) return yyjson_mut_null(doc);
    yyjson_mut_val *result = yyjson_mut_obj(doc);
    if (!result || !nn_automation_json_string(doc, result, "id", nn_project_id(project)) ||
        !nn_automation_json_string(doc, result, "version", nn_project_version(project)) ||
        !nn_automation_json_string(doc, result, "name", nn_project_name(project)) ||
        !yyjson_mut_obj_add_bool(doc, result, "dirty", nn_project_dirty(project))) return NULL;
    const NNDataset *active = nn_project_active_dataset(project);
    yyjson_mut_val *active_value = active ? nn_automation_identity(doc, active->id, active->version) : yyjson_mut_null(doc);
    if (!active_value || !yyjson_mut_obj_add_val(doc, result, "activeDataset", active_value)) return NULL;
    yyjson_mut_val *packages = yyjson_mut_arr(doc), *datasets = yyjson_mut_arr(doc);
    if (!packages || !datasets) return NULL;
    const NNCatalog *catalog = nn_project_catalog(project);
    for (size_t i = 0; i < nn_catalog_count(catalog); ++i) {
        const NNPackage *p = nn_catalog_at(catalog, i);
        yyjson_mut_val *item = nn_automation_identity(doc, p->id, p->version);
        yyjson_mut_val *outputs = yyjson_mut_arr(doc);
        if (!item || !outputs || !nn_automation_json_string(doc, item, "name", p->name) ||
            !yyjson_mut_obj_add_val(doc, item, "outputs", outputs)) return NULL;
        for (size_t h = 0; h < p->output_count; ++h) {
            yyjson_mut_val *handle = yyjson_mut_obj(doc);
            if (!handle || !nn_automation_json_string(doc, handle, "id", p->outputs[h].id) ||
                !nn_automation_json_string(doc, handle, "type", p->outputs[h].type) ||
                !yyjson_mut_arr_append(outputs, handle)) return NULL;
        }
        if (!nn_automation_json_string(doc, item, "kind", p->kind) ||
            !yyjson_mut_arr_append(packages, item)) return NULL;
    }
    for (size_t i = 0; i < nn_project_dataset_count(project); ++i) {
        const NNDataset *d = nn_project_dataset_at(project, i);
        yyjson_mut_val *item = nn_automation_identity(doc, d->id, d->version);
        yyjson_mut_val *inputs = slots_json(doc, d->inputs, d->input_count);
        yyjson_mut_val *targets = slots_json(doc, d->targets, d->target_count);
        if (!item || !inputs || !targets || !nn_automation_json_string(doc, item, "name", d->name) ||
            !yyjson_mut_obj_add_val(doc, item, "inputs", inputs) ||
            !yyjson_mut_obj_add_val(doc, item, "targets", targets) ||
            !yyjson_mut_arr_append(datasets, item)) return NULL;
    }
    if (!yyjson_mut_obj_add_val(doc, result, "packages", packages) ||
        !yyjson_mut_obj_add_val(doc, result, "datasets", datasets)) return NULL;
    yyjson_mut_val *nodes = yyjson_mut_arr(doc), *edges = yyjson_mut_arr(doc);
    if (!nodes || !edges) return NULL;
    const NNModel *model = nn_app_model(app);
    for (size_t i = 0; i < nn_model_node_count(model); ++i) {
        const NNNode *n = nn_model_node_at(model, i);
        yyjson_mut_val *item = yyjson_mut_obj(doc), *params = yyjson_mut_obj(doc);
        yyjson_mut_val *package = nn_automation_identity(doc, n->package_id, n->package_version);
        if (!item || !params || !package || !nn_automation_json_string(doc, item, "id", n->id) ||
            !nn_automation_json_string(doc, item, "name", n->label) || !nn_automation_json_string(doc, item, "scope", n->scope_id) ||
            !yyjson_mut_obj_add_val(doc, item, "package", package) ||
            !yyjson_mut_obj_add_int(doc, item, "x", n->x) ||
            !yyjson_mut_obj_add_int(doc, item, "y", n->y)) return NULL;
        if (n->boundary_handle_id) {
            if (!nn_automation_json_string(doc, item, "boundaryHandle", n->boundary_handle_id)) return NULL;
        } else if (!yyjson_mut_obj_add_null(doc, item, "boundaryHandle")) return NULL;
        for (size_t j = 0; j < n->parameter_count; ++j) {
            yyjson_mut_val *key = yyjson_mut_strcpy(doc, n->parameters[j].key);
            yyjson_mut_val *value = snapshot_value(doc, &n->parameters[j].value);
            if (!key || !value || !yyjson_mut_obj_add(params, key, value)) return NULL;
        }
        if (!yyjson_mut_obj_add_val(doc, item, "parameters", params) || !yyjson_mut_arr_append(nodes, item)) return NULL;
    }
    for (size_t i = 0; i < nn_model_edge_count(model); ++i) {
        const NNEdge *e = nn_model_edge_at(model, i);
        yyjson_mut_val *item = yyjson_mut_obj(doc);
        if (!item || !nn_automation_json_string(doc, item, "id", e->id) || !nn_automation_json_string(doc, item, "source", e->source_id) ||
            !nn_automation_json_string(doc, item, "sourceHandle", e->source_handle_id) || !nn_automation_json_string(doc, item, "target", e->target_id) ||
            !nn_automation_json_string(doc, item, "targetHandle", e->target_handle_id) || !nn_automation_json_string(doc, item, "scope", e->scope_id) ||
            !yyjson_mut_arr_append(edges, item)) return NULL;
    }
    if (!yyjson_mut_obj_add_val(doc, result, "nodes", nodes) || !yyjson_mut_obj_add_val(doc, result, "edges", edges)) return NULL;
    return result;
}
