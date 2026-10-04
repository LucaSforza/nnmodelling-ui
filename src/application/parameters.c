#include "application_internal.h"
#include "application/application.h"
#include "utils/utils.h"
#include "inference/inference.h"
#include "yyjson.h"

#include <math.h>
#include <stdint.h>
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static bool parameter_definition(const NNPackage *package, const char *key,
                                 const NNParameterDef **result)
{
    if (!package || !key) return false;
    for (size_t i = 0; i < package->parameter_count; ++i) {
        if (!strcmp(package->parameters[i].key, key)) {
            if (result) *result = &package->parameters[i];
            return true;
        }
    }
    return false;
}

static bool valid_json_value(const NNValue *value, unsigned depth)
{
    if (!value || depth > 64) return false;
    switch (value->type) {
    case NN_VALUE_BOOL:
    case NN_VALUE_INT: return true;
    case NN_VALUE_REAL: return isfinite(value->as.real);
    case NN_VALUE_STRING: return value->as.string != NULL;
    case NN_VALUE_ARRAY:
        if (value->as.array.count > 1024 ||
            (value->as.array.count && !value->as.array.items)) return false;
        for (size_t i = 0; i < value->as.array.count; ++i)
            if (!valid_json_value(&value->as.array.items[i], depth + 1)) return false;
        return true;
    case NN_VALUE_OBJECT:
        if (value->as.object.count > 1024 ||
            (value->as.object.count && !value->as.object.items)) return false;
        for (size_t i = 0; i < value->as.object.count; ++i) {
            if (!value->as.object.items[i].key || !value->as.object.items[i].key[0]) return false;
            for (size_t j = 0; j < i; ++j)
                if (!strcmp(value->as.object.items[i].key, value->as.object.items[j].key)) return false;
            if (!valid_json_value(&value->as.object.items[i].value, depth + 1)) return false;
        }
        return true;
    default: return false;
    }
}

static bool valid_value(const NNParameterDef *definition, const NNValue *value,
                        char *error, size_t capacity)
{
    if (!definition || !value) return nn_fail(error, capacity, "invalid parameter");
    if (!strcmp(definition->type, "boolean")) {
        if (value->type != NN_VALUE_BOOL) return nn_fail(error, capacity, "parameter requires a boolean");
    } else if (!strcmp(definition->type, "integer")) {
        if (value->type != NN_VALUE_INT) return nn_fail(error, capacity, "parameter requires an integer");
        if (definition->has_minimum && (double)value->as.integer < definition->minimum)
            return nn_fail(error, capacity, "integer is below minimum");
    } else if (!strcmp(definition->type, "number")) {
        if (value->type != NN_VALUE_REAL || !isfinite(value->as.real))
            return nn_fail(error, capacity, "parameter requires a finite number");
        if (definition->has_minimum && value->as.real < definition->minimum)
            return nn_fail(error, capacity, "number is below minimum");
    } else if (!strcmp(definition->type, "string") ||
               !strcmp(definition->type, "dtype")) {
        if (value->type != NN_VALUE_STRING || !value->as.string)
            return nn_fail(error, capacity, "parameter requires a string");
        if (definition->choice_count) {
            bool found = false;
            for (size_t i = 0; i < definition->choice_count; ++i)
                if (!strcmp(value->as.string, definition->choices[i])) found = true;
            if (!found) return nn_fail(error, capacity, "value is not an allowed choice");
        }
    } else if (!strcmp(definition->type, "json")) {
        if (value->type != NN_VALUE_ARRAY || !valid_json_value(value, 0))
            return nn_fail(error, capacity, "parameter requires a JSON array");
    } else if (!strcmp(definition->type, "stereotype")) {
        if (value->type != NN_VALUE_OBJECT || !valid_json_value(value, 0))
            return nn_fail(error, capacity, "parameter requires a stereotype reference object");
    } else return nn_fail(error, capacity, "unsupported parameter type");
    nn_error_set(error, capacity, "");
    return true;
}

static bool valid_stereotype_parameter(const NNApplication *app,
                                       const NNParameterDef *definition,
                                       const NNValue *value,
                                       char *error, size_t capacity)
{
    NNPackage package = { .parameters = definition, .parameter_count = 1 };
    NNParameter parameter = { .key = (char *)definition->key,
                              .value = *(NNValue *)value };
    NNParameter *effective = NULL;
    size_t count = 0;
    bool okay = nn_catalog_parameters(nn_project_catalog(app->project), &package,
                                      &parameter, 1, &effective, &count,
                                      error, capacity);
    nn_catalog_parameters_free(effective, count);
    return okay;
}

bool nn_app_parse_json_value(const yyjson_val *source, NNValue *value, unsigned depth)
{
    if (depth > 64) return false;
    memset(value, 0, sizeof(*value));
    if (yyjson_is_bool(source)) {
        value->type = NN_VALUE_BOOL;
        value->as.boolean = yyjson_get_bool(source);
    } else if (yyjson_is_int(source)) {
        if (yyjson_is_uint(source) && yyjson_get_uint(source) > LLONG_MAX) return false;
        value->type = NN_VALUE_INT;
        value->as.integer = (long long)yyjson_get_sint(source);
    } else if (yyjson_is_num(source)) {
        value->type = NN_VALUE_REAL;
        value->as.real = yyjson_get_num(source);
        if (!isfinite(value->as.real)) return false;
    } else if (yyjson_is_str(source)) {
        if (strlen(yyjson_get_str(source)) != yyjson_get_len(source)) return false;
        value->type = NN_VALUE_STRING;
        value->as.string = nn_text_copy(yyjson_get_str(source));
        return value->as.string != NULL;
    } else if (yyjson_is_arr(source)) {
        size_t count = yyjson_arr_size(source);
        if (count > 1024 || count > SIZE_MAX / sizeof(NNValue)) return false;
        value->type = NN_VALUE_ARRAY;
        value->as.array.items = calloc(count ? count : 1, sizeof(NNValue));
        if (!value->as.array.items) return false;
        yyjson_arr_iter iter = yyjson_arr_iter_with(source);
        yyjson_val *item;
        while ((item = yyjson_arr_iter_next(&iter))) {
            size_t index = value->as.array.count;
            if (!nn_app_parse_json_value(item, &value->as.array.items[index], depth + 1)) {
                nn_value_dispose(value);
                return false;
            }
            ++value->as.array.count;
        }
    } else if (yyjson_is_obj(source)) {
        size_t count = yyjson_obj_size(source);
        if (count > 1024 || count > SIZE_MAX / sizeof(NNParameter)) return false;
        value->type = NN_VALUE_OBJECT;
        value->as.object.items = calloc(count ? count : 1, sizeof(NNParameter));
        if (!value->as.object.items) return false;
        yyjson_obj_iter iter = yyjson_obj_iter_with(source);
        yyjson_val *key;
        while ((key = yyjson_obj_iter_next(&iter))) {
            size_t index = value->as.object.count;
            const char *text = yyjson_get_str(key);
            if (!text || !text[0] || strlen(text) != yyjson_get_len(key)) {
                nn_value_dispose(value); return false;
            }
            value->as.object.items[index].key = nn_text_copy(text);
            if (!value->as.object.items[index].key ||
                !nn_app_parse_json_value(yyjson_obj_iter_get_val(key),
                                         &value->as.object.items[index].value, depth + 1)) {
                value->as.object.count = index + 1;
                nn_value_dispose(value);
                return false;
            }
            for (size_t i = 0; i < index; ++i)
                if (!strcmp(value->as.object.items[i].key, text)) {
                    value->as.object.count = index + 1;
                    nn_value_dispose(value);
                    return false;
                }
            value->as.object.count = index + 1;
        }
    } else return false;
    return true;
}

static bool parse_parameter_text(const NNParameterDef *definition, const char *text,
                                 NNValue *value, char *error, size_t capacity)
{
    memset(value, 0, sizeof(*value));
    if (!definition || !text) return nn_fail(error, capacity, "invalid parameter text");
    if (!strcmp(definition->type, "boolean")) {
        if (strcmp(text, "true") && strcmp(text, "false"))
            return nn_fail(error, capacity, "boolean must be true or false");
        value->type = NN_VALUE_BOOL;
        value->as.boolean = !strcmp(text, "true");
    } else if (!strcmp(definition->type, "integer")) {
        char *end = NULL;
        errno = 0;
        value->type = NN_VALUE_INT;
        value->as.integer = strtoll(text, &end, 10);
        if (!*text || !end || *end || errno == ERANGE) return nn_fail(error, capacity, "invalid integer");
    } else if (!strcmp(definition->type, "number")) {
        char *end = NULL;
        errno = 0;
        value->type = NN_VALUE_REAL;
        value->as.real = strtod(text, &end);
        if (!*text || !end || *end || errno == ERANGE || !isfinite(value->as.real))
            return nn_fail(error, capacity, "invalid finite number");
    } else if (!strcmp(definition->type, "string") ||
               !strcmp(definition->type, "dtype")) {
        value->type = NN_VALUE_STRING;
        value->as.string = (char *)text;
    } else if (!strcmp(definition->type, "json") ||
               !strcmp(definition->type, "stereotype")) {
        yyjson_doc *document = yyjson_read_opts((char *)(uintptr_t)text, strlen(text), 0,
                                                 NULL, NULL);
        if (!document) return nn_fail(error, capacity, "invalid JSON array");
        const yyjson_val *root = yyjson_doc_get_root(document);
        bool okay = (!strcmp(definition->type, "json") ? yyjson_is_arr(root) : yyjson_is_obj(root)) &&
                    nn_app_parse_json_value(root, value, 0);
        yyjson_doc_free(document);
        if (!okay) {
            nn_value_dispose(value);
            return nn_fail(error, capacity, "parameter has an invalid JSON value");
        }
    } else return nn_fail(error, capacity, "unsupported parameter type");
    if (!valid_value(definition, value, error, capacity)) {
        if (value->type == NN_VALUE_ARRAY || value->type == NN_VALUE_OBJECT) nn_value_dispose(value);
        return false;
    }
    return true;
}

static bool default_value(const NNApplication *app, const NNPackage *package,
                          const NNParameterDef *definition, NNValue *value)
{
    memset(value, 0, sizeof(*value));
    if (definition->has_default) {
        switch (definition->default_value.type) {
        case NN_PARAMETER_BOOLEAN:
            value->type = NN_VALUE_BOOL;
            value->as.boolean = definition->default_value.as.boolean;
            return true;
        case NN_PARAMETER_INTEGER:
            value->type = NN_VALUE_INT;
            value->as.integer = definition->default_value.as.integer;
            return true;
        case NN_PARAMETER_NUMBER:
            value->type = NN_VALUE_REAL;
            value->as.real = definition->default_value.as.number;
            return true;
        case NN_PARAMETER_STRING:
            value->type = NN_VALUE_STRING;
            value->as.string = (char *)definition->default_value.as.string;
            return true;
        case NN_PARAMETER_JSON: {
            const char *text = definition->default_value.as.string;
            yyjson_doc *document = yyjson_read_opts((char *)(uintptr_t)text,
                                                     strlen(text), 0, NULL, NULL);
            if (!document) return false;
            const yyjson_val *root = yyjson_doc_get_root(document);
            bool okay = (!strcmp(definition->type, "stereotype")
                ? yyjson_is_obj(root) : yyjson_is_arr(root)) &&
                nn_app_parse_json_value(root, value, 0);
            yyjson_doc_free(document);
            if (!okay) nn_value_dispose(value);
            return okay;
        }
        default: return false;
        }
    }
    if (!strcmp(definition->type, "integer")) {
        if (definition->has_minimum && definition->minimum > (double)LLONG_MAX)
            return false;
        value->type = NN_VALUE_INT;
        value->as.integer = definition->has_minimum && definition->minimum > 1
            ? (long long)definition->minimum : 1;
    } else if (!strcmp(definition->type, "number")) {
        value->type = NN_VALUE_REAL;
        value->as.real = definition->has_minimum ? definition->minimum : 0;
    } else if (!strcmp(definition->type, "boolean")) {
        value->type = NN_VALUE_BOOL;
    } else if (!strcmp(definition->type, "string") ||
               !strcmp(definition->type, "dtype")) {
        const char *text = "";
        const NNDataset *dataset = nn_project_active_dataset(app->project);
        if (!strcmp(definition->key, "binding") && dataset && dataset->input_count)
            text = dataset->inputs[0].name;
        else if (definition->choice_count) text = definition->choices[0];
        value->type = NN_VALUE_STRING;
        value->as.string = (char *)text;
    } else return false;
    (void)package;
    return true;
}

bool nn_app_add_default(NNApplication *app, NNModel *model, const char *node_id,
                        const NNPackage *package, const NNParameterDef *definition,
                        char *error, size_t capacity)
{
    NNValue value = {0};
    if (!default_value(app, package, definition, &value))
        return nn_fail(error, capacity, "cannot construct parameter default");
    if (!valid_value(definition, &value, error, capacity)) {
        if (definition->has_default &&
            (value.type == NN_VALUE_ARRAY || value.type == NN_VALUE_OBJECT))
            nn_value_dispose(&value);
        return false;
    }
    if (!strcmp(definition->type, "stereotype") &&
        !valid_stereotype_parameter(app, definition, &value, error, capacity)) {
        if (definition->has_default &&
            (value.type == NN_VALUE_ARRAY || value.type == NN_VALUE_OBJECT))
            nn_value_dispose(&value);
        return false;
    }
    bool okay = nn_model_set_parameter(model, node_id, definition->key,
                                       &value, error, capacity);
    if (definition->has_default &&
        (value.type == NN_VALUE_ARRAY || value.type == NN_VALUE_OBJECT))
        nn_value_dispose(&value);
    return okay;
}
bool nn_app_set_parameter(NNApplication *app, const char *node_id, const char *key,
                          const NNValue *value, char *error, size_t cap)
{
    if (!app || !app->project) return nn_app_history_fail(app, error, cap, "no active project");
    const NNNode *node = nn_model_find_node(nn_project_model(app->project), node_id);
    if (!node) return nn_app_history_fail(app, error, cap, "node not found");
    const NNPackage *package = nn_app_find_package(app, node);
    const NNParameterDef *definition = NULL;
    if (!parameter_definition(package, key, &definition))
        return nn_app_history_fail(app, error, cap, "unknown parameter");
    if (!valid_value(definition, value, error, cap)) {
        if (app->history.group_active) app->history.group_failed = true;
        return false;
    }
    if (!strcmp(definition->type, "stereotype") &&
        !valid_stereotype_parameter(app, definition, value, error, cap)) {
        if (app->history.group_active) app->history.group_failed = true;
        return false;
    }
    if (!nn_app_history_prepare(app, error, cap)) return false;
    const bool okay = nn_model_set_parameter(nn_project_model(app->project), node_id,
                                              key, value, error, cap);
    nn_app_history_finish(app, okay);
    if (!okay) return false;
    nn_app_invalidate_analysis(app);
    nn_error_set(error, cap, "");
    return true;
}

bool nn_app_set_parameter_text(NNApplication *app, const char *node, const char *key,
                               const char *text, char *error, size_t cap)
{
    if (!app || !app->project) return nn_app_history_fail(app, error, cap, "no active project");
    const NNNode *entry = nn_model_find_node(nn_project_model(app->project), node);
    const NNParameterDef *definition = NULL;
    if (!entry) return nn_app_history_fail(app, error, cap, "node not found");
    if (!parameter_definition(nn_app_find_package(app, entry), key, &definition))
        return nn_app_history_fail(app, error, cap, "unknown parameter");
    NNValue value = {0};
    if (!parse_parameter_text(definition, text, &value, error, cap)) {
        if (app->history.group_active) app->history.group_failed = true;
        return false;
    }
    bool okay = nn_app_set_parameter(app, node, key, &value, error, cap);
    if (value.type == NN_VALUE_ARRAY || value.type == NN_VALUE_OBJECT) nn_value_dispose(&value);
    return okay;
}

static yyjson_mut_val *parameter_json_value(yyjson_mut_doc *doc, const NNValue *value)
{
    switch (value->type) {
    case NN_VALUE_BOOL: return yyjson_mut_bool(doc, value->as.boolean);
    case NN_VALUE_INT: return yyjson_mut_sint(doc, value->as.integer);
    case NN_VALUE_REAL:
        if (!isfinite(value->as.real)) return NULL;
        if (value->as.real >= (double)LLONG_MIN && value->as.real < -(double)LLONG_MIN &&
            trunc(value->as.real) == value->as.real)
            return yyjson_mut_sint(doc, (long long)value->as.real);
        return yyjson_mut_real(doc, value->as.real);
    case NN_VALUE_STRING: return value->as.string ? yyjson_mut_strcpy(doc, value->as.string) : NULL;
    case NN_VALUE_ARRAY: {
        if (value->as.array.count && !value->as.array.items) return NULL;
        yyjson_mut_val *array = yyjson_mut_arr(doc);
        if (!array) return NULL;
        for (size_t i = 0; i < value->as.array.count; ++i) {
            yyjson_mut_val *item = parameter_json_value(doc, &value->as.array.items[i]);
            if (!item || !yyjson_mut_arr_append(array, item)) return NULL;
        }
        return array;
    }
    case NN_VALUE_OBJECT: {
        if (value->as.object.count && !value->as.object.items) return NULL;
        yyjson_mut_val *object = yyjson_mut_obj(doc);
        if (!object) return NULL;
        for (size_t i = 0; i < value->as.object.count; ++i) {
            const NNParameter *item = &value->as.object.items[i];
            yyjson_mut_val *key = item->key ? yyjson_mut_strcpy(doc, item->key) : NULL;
            yyjson_mut_val *child = parameter_json_value(doc, &item->value);
            if (!key || !child || !yyjson_mut_obj_add(object, key, child)) return NULL;
        }
        return object;
    }
    default: return NULL;
    }
}

char *nn_app_parameter_text(const NNApplication *app, const char *node_id, const char *key)
{
    const NNNode *node = app && app->project
        ? nn_model_find_node(nn_project_model(app->project), node_id) : NULL;
    if (!node || !key) return NULL;
    for (size_t i = 0; i < node->parameter_count; ++i) {
        if (strcmp(node->parameters[i].key, key)) continue;
        if (node->parameters[i].value.type == NN_VALUE_STRING)
            return nn_text_copy(node->parameters[i].value.as.string);
        yyjson_mut_doc *document = yyjson_mut_doc_new(NULL);
        if (!document) return NULL;
        yyjson_mut_val *value = parameter_json_value(document, &node->parameters[i].value);
        if (!value) { yyjson_mut_doc_free(document); return NULL; }
        yyjson_mut_doc_set_root(document, value);
        char *text = yyjson_mut_write(document, 0, NULL);
        yyjson_mut_doc_free(document);
        return text;
    }
    return NULL;
}

void nn_app_free_text(char *text) { free(text); }
