#include "application_internal.h"

#include <stdlib.h>
#include <string.h>
#include "utils/utils.h"

static yyjson_mut_val *write_value(yyjson_mut_doc *doc, const NNValue *value)
{
    switch (value->type) {
    case NN_VALUE_BOOL: return yyjson_mut_bool(doc, value->as.boolean);
    case NN_VALUE_INT: return yyjson_mut_sint(doc, value->as.integer);
    case NN_VALUE_REAL: return yyjson_mut_real(doc, value->as.real);
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
    case NN_VALUE_OBJECT: {
        yyjson_mut_val *object = yyjson_mut_obj(doc);
        if (!object) return NULL;
        for (size_t i = 0; i < value->as.object.count; ++i) {
            const NNParameter *entry = &value->as.object.items[i];
            yyjson_mut_val *key = yyjson_mut_strcpy(doc, entry->key);
            yyjson_mut_val *item = write_value(doc, &entry->value);
            if (!key || !item || !yyjson_mut_obj_add(object, key, item)) return NULL;
        }
        return object;
    }
    default: return NULL;
    }
}

static bool replace_member(yyjson_mut_val *object, const char *key,
                           yyjson_mut_val *value)
{
    yyjson_mut_obj_iter iterator = yyjson_mut_obj_iter_with(object);
    yyjson_mut_val *member = NULL;
    yyjson_mut_val *current = NULL;
    while ((current = yyjson_mut_obj_iter_next(&iterator))) {
        const char *text = yyjson_mut_get_str(current);
        if (text && !strcmp(text, key)) { member = current; break; }
    }
    return member && yyjson_mut_obj_replace(object, member, value);
}

char *nn_app_resolved_model_json(NNApplication *app, char *error, size_t capacity)
{
    if (error && capacity) error[0] = '\0';
    NNProject *project = app ? app->project : NULL;
    if (!project) { nn_fail(error, capacity, "no active project"); return NULL; }
    char *source_text = nn_project_model_json(project, error, capacity);
    if (!source_text) return NULL;
    yyjson_doc *source = yyjson_read(source_text, strlen(source_text), 0);
    free(source_text);
    yyjson_mut_doc *doc = source ? yyjson_doc_mut_copy(source, NULL) : NULL;
    if (source) yyjson_doc_free(source);
    yyjson_mut_val *root = doc ? yyjson_mut_doc_get_root(doc) : NULL;
    yyjson_mut_val *manifest = root ? yyjson_mut_obj_get(root, "manifest") : NULL;
    bool okay = doc && root && manifest;
    if (okay) {
        yyjson_mut_obj_remove_key(manifest, "metavariables");
        okay = replace_member(manifest, "schemaVersion", yyjson_mut_int(doc, 2));
    }
    yyjson_mut_val *nodes = root ? yyjson_mut_obj_get(root, "nodes") : NULL;
    const NNModel *model = nn_project_model(project);
    if (!nodes || yyjson_mut_arr_size(nodes) != nn_model_node_count(model)) okay = false;
    for (size_t i = 0; okay && i < nn_model_node_count(model); ++i) {
        const NNNode *node = nn_model_node_at(model, i);
        yyjson_mut_val *item = yyjson_mut_arr_get(nodes, i);
        yyjson_mut_val *data = item ? yyjson_mut_obj_get(item, "data") : NULL;
        NNParameter *parameters = NULL;
        size_t count = 0;
        if (!nn_inference_resolve_parameters(project, node->id, &parameters, &count,
                                               error, capacity)) { okay = false; break; }
        yyjson_mut_val *values = yyjson_mut_obj(doc);
        okay = data && values;
        for (size_t p = 0; okay && p < count; ++p) {
            yyjson_mut_val *key = yyjson_mut_strcpy(doc, parameters[p].key);
            yyjson_mut_val *value = write_value(doc, &parameters[p].value);
            okay = key && value && yyjson_mut_obj_add(values, key, value);
        }
        if (okay) {
            okay = replace_member(data, "params", values);
        }
        nn_catalog_parameters_free(parameters, count);
        if (!okay && error && capacity && !error[0])
            nn_errorf(error, capacity, "unable to serialize resolved parameters for node '%s'", node->id);
    }
    char *text = okay ? yyjson_mut_write(doc, YYJSON_WRITE_PRETTY_TWO_SPACES, NULL) : NULL;
    yyjson_mut_doc_free(doc);
    if (!text && error && capacity && !error[0]) nn_fail(error, capacity, "unable to create resolved execution model from source document");
    return text;
}

char *nn_app_resolved_parameter_text(NNApplication *app, const char *node,
                                     const char *key, char *error, size_t capacity)
{
    if (error && capacity) error[0] = '\0';
    if (!app || !app->project || !node || !key) { nn_fail(error, capacity, "invalid resolved parameter request"); return NULL; }
    NNParameter *parameters = NULL;
    size_t count = 0;
    if (!nn_inference_resolve_parameters(app->project, node, &parameters, &count, error, capacity)) return NULL;
    const NNValue *value = NULL;
    for (size_t i = 0; i < count; ++i)
        if (!strcmp(parameters[i].key, key)) { value = &parameters[i].value; break; }
    if (!value) { nn_catalog_parameters_free(parameters, count); nn_fail(error, capacity, "unknown parameter"); return NULL; }
    char *text = NULL;
    if (value->type == NN_VALUE_STRING) text = nn_text_copy(value->as.string);
    else {
        yyjson_mut_doc *doc = yyjson_mut_doc_new(NULL);
        yyjson_mut_val *root = doc ? write_value(doc, value) : NULL;
        if (doc && root) {
            yyjson_mut_doc_set_root(doc, root);
            text = yyjson_mut_write(doc, 0, NULL);
        }
        yyjson_mut_doc_free(doc);
    }
    nn_catalog_parameters_free(parameters, count);
    if (!text) nn_fail(error, capacity, "unable to format resolved parameter");
    return text;
}
