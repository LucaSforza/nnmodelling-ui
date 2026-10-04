#include "catalog_internal.h"
#include "utils/utils.h"

#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

enum { NN_PARAMETER_ITEM_LIMIT = 1024, NN_REFERENCE_DEPTH_LIMIT = 16 };

static void parameters_free(NNParameter *parameters, size_t count)
{
    for (size_t i = 0; parameters && i < count; ++i) {
        free(parameters[i].key);
        nn_value_dispose(&parameters[i].value);
    }
    free(parameters);
}

void nn_catalog_parameters_free(NNParameter *parameters, size_t count)
{
    parameters_free(parameters, count);
}

static const NNParameter *parameter_find(const NNParameter *values, size_t count,
                                         const char *key)
{
    for (size_t i = 0; values && i < count; ++i)
        if (values[i].key && !strcmp(values[i].key, key)) return &values[i];
    return NULL;
}

static bool parse_json_value(yyjson_val *source, NNValue *value, unsigned depth)
{
    if (!source || !value || depth > 64) return false;
    memset(value, 0, sizeof(*value));
    if (yyjson_is_bool(source)) {
        value->type = NN_VALUE_BOOL;
        value->as.boolean = yyjson_get_bool(source);
    } else if (yyjson_is_int(source)) {
        if (yyjson_is_uint(source) && yyjson_get_uint(source) > LLONG_MAX) return false;
        value->type = NN_VALUE_INT;
        value->as.integer = yyjson_get_sint(source);
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
        if (count > NN_PARAMETER_ITEM_LIMIT) return false;
        value->type = NN_VALUE_ARRAY;
        value->as.array.items = calloc(count ? count : 1, sizeof(NNValue));
        if (!value->as.array.items) return false;
        yyjson_arr_iter iter = yyjson_arr_iter_with(source);
        yyjson_val *item;
        while ((item = yyjson_arr_iter_next(&iter))) {
            size_t at = value->as.array.count;
            if (!parse_json_value(item, &value->as.array.items[at], depth + 1)) {
                value->as.array.count = at + 1;
                nn_value_dispose(value);
                return false;
            }
            value->as.array.count = at + 1;
        }
    } else if (yyjson_is_obj(source)) {
        size_t count = yyjson_obj_size(source);
        if (count > NN_PARAMETER_ITEM_LIMIT) return false;
        value->type = NN_VALUE_OBJECT;
        value->as.object.items = calloc(count ? count : 1, sizeof(NNParameter));
        if (!value->as.object.items) return false;
        yyjson_obj_iter iter = yyjson_obj_iter_with(source);
        yyjson_val *key;
        while ((key = yyjson_obj_iter_next(&iter))) {
            size_t at = value->as.object.count;
            const char *name = yyjson_get_str(key);
            if (!name || !name[0] || strlen(name) != yyjson_get_len(key)) {
                nn_value_dispose(value); return false;
            }
            for (size_t i = 0; i < at; ++i)
                if (!strcmp(value->as.object.items[i].key, name)) {
                    nn_value_dispose(value);
                    return false;
                }
            value->as.object.items[at].key = nn_text_copy(name);
            if (!value->as.object.items[at].key ||
                !parse_json_value(yyjson_obj_iter_get_val(key),
                                  &value->as.object.items[at].value, depth + 1)) {
                value->as.object.count = at + 1;
                nn_value_dispose(value);
                return false;
            }
            value->as.object.count = at + 1;
        }
    } else return false;
    return true;
}

static bool value_json(const char *text, bool object, NNValue *value)
{
    if (!text) return false;
    yyjson_doc *document = yyjson_read_opts((char *)(uintptr_t)text, strlen(text), 0,
                                             NULL, NULL);
    if (!document) return false;
    yyjson_val *root = yyjson_doc_get_root(document);
    bool okay = (object ? yyjson_is_obj(root) : yyjson_is_arr(root)) &&
                parse_json_value(root, value, 0);
    yyjson_doc_free(document);
    return okay;
}

static bool make_default(const NNParameterDef *definition, NNValue *value)
{
    memset(value, 0, sizeof(*value));
    if (definition->has_default) {
        switch (definition->default_value.type) {
        case NN_PARAMETER_BOOLEAN:
            value->type = NN_VALUE_BOOL;
            value->as.boolean = definition->default_value.as.boolean;
            break;
        case NN_PARAMETER_INTEGER:
            value->type = NN_VALUE_INT;
            value->as.integer = definition->default_value.as.integer;
            break;
        case NN_PARAMETER_NUMBER:
            value->type = NN_VALUE_REAL;
            value->as.real = definition->default_value.as.number;
            break;
        case NN_PARAMETER_STRING:
            value->type = NN_VALUE_STRING;
            value->as.string = nn_text_copy(definition->default_value.as.string);
            if (!value->as.string) return false;
            break;
        case NN_PARAMETER_JSON:
            return value_json(definition->default_value.as.string,
                              !strcmp(definition->type, "stereotype"), value);
        default: return false;
        }
        return true;
    }
    if (!strcmp(definition->type, "boolean")) {
        value->type = NN_VALUE_BOOL;
    } else if (!strcmp(definition->type, "integer")) {
        if (definition->has_minimum && definition->minimum > (double)LLONG_MAX) return false;
        value->type = NN_VALUE_INT;
        value->as.integer = definition->has_minimum && definition->minimum > 1
            ? (long long)definition->minimum : 1;
    } else if (!strcmp(definition->type, "number")) {
        value->type = NN_VALUE_REAL;
        value->as.real = definition->has_minimum ? definition->minimum : 0;
    } else if (!strcmp(definition->type, "string") || !strcmp(definition->type, "dtype")) {
        value->type = NN_VALUE_STRING;
        value->as.string = nn_text_copy(definition->choice_count ? definition->choices[0] : "");
        if (!value->as.string) return false;
    } else if (!strcmp(definition->type, "stereotype")) {
        return false;
    } else return false;
    return true;
}

static bool valid_json_tree(const NNValue *value, unsigned depth)
{
    if (!value || depth > 64) return false;
    if (value->type == NN_VALUE_BOOL || value->type == NN_VALUE_INT) return true;
    if (value->type == NN_VALUE_REAL) return isfinite(value->as.real);
    if (value->type == NN_VALUE_STRING) return value->as.string != NULL;
    if (value->type == NN_VALUE_ARRAY) {
        if (value->as.array.count > NN_PARAMETER_ITEM_LIMIT ||
            (value->as.array.count && !value->as.array.items)) return false;
        for (size_t i = 0; i < value->as.array.count; ++i)
            if (!valid_json_tree(&value->as.array.items[i], depth + 1)) return false;
        return true;
    }
    if (value->type == NN_VALUE_OBJECT) {
        if (value->as.object.count > NN_PARAMETER_ITEM_LIMIT ||
            (value->as.object.count && !value->as.object.items)) return false;
        for (size_t i = 0; i < value->as.object.count; ++i) {
            const NNParameter *item = &value->as.object.items[i];
            if (!item->key || !item->key[0]) return false;
            for (size_t j = 0; j < i; ++j)
                if (!strcmp(item->key, value->as.object.items[j].key)) return false;
            if (!valid_json_tree(&item->value, depth + 1)) return false;
        }
        return true;
    }
    return false;
}

static const NNValue *object_get(const NNValue *object, const char *key)
{
    if (!object || object->type != NN_VALUE_OBJECT) return NULL;
    for (size_t i = 0; i < object->as.object.count; ++i)
        if (object->as.object.items[i].key &&
            !strcmp(object->as.object.items[i].key, key))
            return &object->as.object.items[i].value;
    return NULL;
}

static bool valid_parameters(const NNCatalog *catalog, const NNPackage *package,
                             const NNParameter *values, size_t count,
                             unsigned depth, char *error, size_t cap);

static bool valid_stereotype(const NNCatalog *catalog, const NNParameterDef *definition,
                             const NNValue *value, unsigned depth,
                             char *error, size_t cap)
{
    if (depth >= NN_REFERENCE_DEPTH_LIMIT || value->type != NN_VALUE_OBJECT ||
        !valid_json_tree(value, 0) ||
        value->as.object.count != 3)
        return nn_fail(error, cap, "invalid or recursively nested stereotype reference");
    const NNValue *id_value = object_get(value, "id");
    const NNValue *version_value = object_get(value, "version");
    const NNValue *parameters_value = object_get(value, "parameters");
    if (!id_value || id_value->type != NN_VALUE_STRING || !id_value->as.string ||
        !version_value || version_value->type != NN_VALUE_STRING || !version_value->as.string ||
        !parameters_value || parameters_value->type != NN_VALUE_OBJECT)
        return nn_fail(error, cap, "stereotype reference requires id, version and parameters");
    const NNPackage *target = nn_catalog_resolve(catalog, id_value->as.string,
                                                  version_value->as.string);
    if (!target)
        return nn_fail(error, cap, "stereotype reference does not resolve uniquely");
    if (definition->kind && (!target->kind || strcmp(definition->kind, target->kind)))
        return nn_fail(error, cap, "stereotype reference has the wrong kind");
    if (!strcmp(target->kind, "subflow"))
        return nn_fail(error, cap, "stereotype references cannot target a subflow");
    return valid_parameters(catalog, target, parameters_value->as.object.items,
                            parameters_value->as.object.count, depth + 1, error, cap);
}

static bool valid_value(const NNCatalog *catalog, const NNParameterDef *definition,
                        const NNValue *value, unsigned depth, char *error, size_t cap)
{
    if (!definition || !value) return nn_fail(error, cap, "invalid parameter");
    if (!strcmp(definition->type, "boolean")) {
        if (value->type != NN_VALUE_BOOL) return nn_fail(error, cap, "parameter requires a boolean");
    } else if (!strcmp(definition->type, "integer")) {
        if (value->type != NN_VALUE_INT) return nn_fail(error, cap, "parameter requires an integer");
        if (definition->has_minimum && (double)value->as.integer < definition->minimum)
            return nn_fail(error, cap, "integer is below minimum");
    } else if (!strcmp(definition->type, "number")) {
        if (value->type != NN_VALUE_REAL || !isfinite(value->as.real))
            return nn_fail(error, cap, "parameter requires a finite number");
        if (definition->has_minimum && value->as.real < definition->minimum)
            return nn_fail(error, cap, "number is below minimum");
    } else if (!strcmp(definition->type, "string") || !strcmp(definition->type, "dtype")) {
        if (value->type != NN_VALUE_STRING || !value->as.string)
            return nn_fail(error, cap, "parameter requires a string");
        for (size_t i = 0; i < definition->choice_count; ++i)
            if (!strcmp(value->as.string, definition->choices[i])) goto valid_string;
        if (definition->choice_count) return nn_fail(error, cap, "value is not an allowed choice");
valid_string:;
    } else if (!strcmp(definition->type, "json")) {
        if (value->type != NN_VALUE_ARRAY || !valid_json_tree(value, 0))
            return nn_fail(error, cap, "parameter requires a JSON array");
    } else if (!strcmp(definition->type, "stereotype")) {
        return valid_stereotype(catalog, definition, value, depth, error, cap);
    } else return nn_fail(error, cap, "unsupported parameter type");
    return true;
}

static bool valid_parameters(const NNCatalog *catalog, const NNPackage *package,
                             const NNParameter *values, size_t count,
                             unsigned depth, char *error, size_t cap)
{
    if (depth > NN_REFERENCE_DEPTH_LIMIT || count > package->parameter_count ||
        (count && !values)) return nn_fail(error, cap, "invalid parameter list");
    for (size_t i = 0; i < count; ++i) {
        if (!values[i].key || !values[i].key[0]) return nn_fail(error, cap, "invalid parameter key");
        for (size_t j = 0; j < i; ++j)
            if (!strcmp(values[i].key, values[j].key)) return nn_fail(error, cap, "duplicate parameter key");
        size_t j;
        for (j = 0; j < package->parameter_count; ++j)
            if (!strcmp(values[i].key, package->parameters[j].key)) break;
        if (j == package->parameter_count) return nn_fail(error, cap, "unknown parameter");
        if (!valid_value(catalog, &package->parameters[j], &values[i].value,
                         depth, error, cap)) return false;
    }
    for (size_t i = 0; i < package->parameter_count; ++i) {
        const NNParameter *value = parameter_find(values, count, package->parameters[i].key);
        NNValue fallback = {0};
        const NNValue *selected = value ? &value->value : &fallback;
        if (!value && !make_default(&package->parameters[i], &fallback))
            return nn_fail(error, cap, "parameter has no usable default");
        bool okay = valid_value(catalog, &package->parameters[i], selected,
                                depth, error, cap);
        if (!value) nn_value_dispose(&fallback);
        if (!okay) return false;
    }
    return true;
}

bool nn_catalog_parameters(const NNCatalog *catalog, const NNPackage *package,
                           const NNParameter *values, size_t count,
                           NNParameter **effective, size_t *effective_count,
                           char *error, size_t cap)
{
    if (effective) *effective = NULL;
    if (effective_count) *effective_count = 0;
    if (!catalog || !package || !effective || !effective_count ||
        package->parameter_count > NN_PARAMETER_ITEM_LIMIT ||
        (count && !values) || count > package->parameter_count)
        return nn_fail(error, cap, "invalid package parameters");
    if (!valid_parameters(catalog, package, values, count, 0, error, cap)) return false;
    NNParameter *result = calloc(package->parameter_count ? package->parameter_count : 1,
                                 sizeof(*result));
    if (!result) return nn_fail(error, cap, "out of memory normalizing parameters");
    size_t copied = 0;
    for (size_t i = 0; i < package->parameter_count; ++i) {
        const NNParameterDef *definition = &package->parameters[i];
        const NNParameter *given = parameter_find(values, count, definition->key);
        NNValue fallback = {0};
        const NNValue *selected = given ? &given->value : &fallback;
        if (!given && !make_default(definition, &fallback)) goto fail;
        result[i].key = nn_text_copy(definition->key);
        if (!result[i].key || !nn_value_copy(&result[i].value, selected)) {
            nn_value_dispose(&fallback);
            goto fail;
        }
        nn_value_dispose(&fallback);
        copied = i + 1;
    }
    *effective = result;
    *effective_count = package->parameter_count;
    nn_error_set(error, cap, "");
    return true;
fail:
    parameters_free(result, copied + (copied < package->parameter_count ? 1 : 0));
    return nn_fail(error, cap, "unable to normalize package parameters");
}
