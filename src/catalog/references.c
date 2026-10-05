#include "catalog/catalog.h"
#include "model/value_json.h"
#include "utils/utils.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

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
            const NNParameter *entry = &value->as.object.items[i];
            if (!entry->key || !*entry->key || !valid_json_value(&entry->value, depth + 1)) return false;
            for (size_t j = 0; j < i; ++j)
                if (!strcmp(entry->key, value->as.object.items[j].key)) return false;
        }
        return true;
    default: return false;
    }
}

bool nn_catalog_validate_value(const NNCatalog *catalog, const NNParameterDef *definition, const NNValue *value,
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
        NNParameter *parameters = NULL;
        size_t count = 0;
        if (!nn_catalog_reference(catalog, value, definition->kind, &parameters, &count,
                                  error, capacity)) return false;
        nn_catalog_parameters_free(parameters, count);
    } else return nn_fail(error, capacity, "unsupported parameter type");
    nn_error_set(error, capacity, "");
    return true;
}


void nn_catalog_parameters_free(NNParameter *parameters, size_t count)
{
    for (size_t i = 0; i < count; ++i) {
        free(parameters[i].key);
        nn_value_dispose(&parameters[i].value);
    }
    free(parameters);
}

const NNPackage *nn_catalog_reference(const NNCatalog *catalog, const NNValue *reference,
                                     const char *kind, NNParameter **parameters, size_t *count,
                                     char *error, size_t capacity)
{
    *parameters = NULL;
    *count = 0;
    if (!reference || reference->type != NN_VALUE_OBJECT || !valid_json_value(reference, 0)) {
        nn_error_set(error, capacity, "stereotype reference requires an object"); return NULL;
    }
    const NNValue *id = nn_value_member(reference, "id");
    const NNValue *version = nn_value_member(reference, "version");
    const NNValue *values = nn_value_member(reference, "parameters");
    if (reference->as.object.count != 3 || !id || id->type != NN_VALUE_STRING ||
        !version || version->type != NN_VALUE_STRING || !values || values->type != NN_VALUE_OBJECT) {
        nn_error_set(error, capacity, "stereotype reference requires id, version and parameters"); return NULL;
    }
    const NNPackage *package = nn_catalog_resolve(catalog, id->as.string, version->as.string);
    if (!package || !strcmp(package->kind, "subflow") ||
        (kind && strcmp(package->kind, kind))) {
        nn_error_set(error, capacity, "stereotype reference has no compatible active package"); return NULL;
    }
    for (size_t i = 0; i < values->as.object.count; ++i) {
        bool found = false;
        for (size_t j = 0; j < package->parameter_count; ++j)
            if (!strcmp(values->as.object.items[i].key, package->parameters[j].key)) found = true;
        if (!found) { nn_error_set(error, capacity, "unknown referenced parameter"); return NULL; }
    }
    NNParameter *result = calloc(package->parameter_count ? package->parameter_count : 1, sizeof(*result));
    if (!result) { nn_error_set(error, capacity, "out of memory resolving reference"); return NULL; }
    for (size_t i = 0; i < package->parameter_count; ++i) {
        const NNParameterDef *def = &package->parameters[i];
        const NNValue *provided = nn_value_member(values, def->key);
        NNValue fallback = {0};
        yyjson_doc *document = NULL;
        bool okay = false;
        if (!strcmp(def->type, "stereotype")) {
            nn_error_set(error, capacity, "recursive stereotype references are unsupported");
            goto failed;
        }
        if (!provided && def->has_default) {
            switch (def->default_value.type) {
            case NN_PARAMETER_BOOLEAN: fallback.type = NN_VALUE_BOOL; fallback.as.boolean = def->default_value.as.boolean; break;
            case NN_PARAMETER_INTEGER: fallback.type = NN_VALUE_INT; fallback.as.integer = def->default_value.as.integer; break;
            case NN_PARAMETER_NUMBER: fallback.type = NN_VALUE_REAL; fallback.as.real = def->default_value.as.number; break;
            case NN_PARAMETER_STRING: fallback.type = NN_VALUE_STRING; fallback.as.string = (char *)def->default_value.as.string; break;
            case NN_PARAMETER_JSON:
                document = yyjson_read(def->default_value.as.string, strlen(def->default_value.as.string), 0);
                if (!document || !nn_value_from_json(yyjson_doc_get_root(document), &fallback)) goto failed;
                break;
            default: goto failed;
            }
            provided = &fallback;
        }
        if (!provided) { nn_error_set(error, capacity, "missing referenced parameter"); goto failed; }
        NNValue numeric;
        if (!strcmp(def->type, "number") && provided->type == NN_VALUE_INT) {
            numeric.type = NN_VALUE_REAL;
            numeric.as.real = (double)provided->as.integer;
            provided = &numeric;
        }
        result[i].key = nn_text_copy(def->key);
        okay = result[i].key && nn_catalog_validate_value(catalog, def, provided, error, capacity) &&
               nn_value_copy(&result[i].value, provided);
failed:
        if (document) { yyjson_doc_free(document); nn_value_dispose(&fallback); }
        if (!okay) {
            nn_catalog_parameters_free(result, package->parameter_count);
            if (error && capacity && !*error) nn_error_set(error, capacity, "invalid referenced parameter or allocation failure");
            return NULL;
        }
    }
    *parameters = result;
    *count = package->parameter_count;
    return package;
}
