#include "application.h"
#include "utils.h"

#include "yyjson.h"
#include "inference.h"

#include <math.h>
#include <stdint.h>
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct NNApplication {
    char *core_root;
    NNProject *project;
    NNInferenceReport *analysis;
};

static void invalidate_analysis(NNApplication *app)
{
    if (!app) return;
    nn_inference_free(app->analysis);
    app->analysis = NULL;
}

static bool kind_is(const NNPackage *package, const char *kind)
{
    return package && package->kind && !strcmp(package->kind, kind);
}

static const NNPackage *find_package(const NNApplication *app, const NNNode *node)
{
    return app && app->project && node
        ? nn_catalog_find(nn_project_catalog(app->project), node->package_id,
                         node->package_version)
        : NULL;
}

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
        return nn_fail(error, capacity, "object-valued stereotype parameters are not supported by native model");
    } else return nn_fail(error, capacity, "unsupported parameter type");
    nn_error_set(error, capacity, "");
    return true;
}

static bool parse_json_value(const yyjson_val *source, NNValue *value, unsigned depth)
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
            if (!parse_json_value(item, &value->as.array.items[index], depth + 1)) {
                nn_value_dispose(value);
                return false;
            }
            ++value->as.array.count;
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
    } else if (!strcmp(definition->type, "json")) {
        yyjson_doc *document = yyjson_read_opts((char *)(uintptr_t)text, strlen(text), 0,
                                                 NULL, NULL);
        if (!document) return nn_fail(error, capacity, "invalid JSON array");
        const yyjson_val *root = yyjson_doc_get_root(document);
        bool okay = yyjson_is_arr(root) && parse_json_value(root, value, 0);
        yyjson_doc_free(document);
        if (!okay) {
            nn_value_dispose(value);
            return nn_fail(error, capacity, "parameter requires a JSON array of primitive values");
        }
    } else return nn_fail(error, capacity, "unsupported parameter type");
    if (!valid_value(definition, value, error, capacity)) {
        if (value->type == NN_VALUE_ARRAY) nn_value_dispose(value);
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
            bool okay = yyjson_is_arr(root) && parse_json_value(root, value, 0);
            yyjson_doc_free(document);
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

static bool add_default(NNApplication *app, NNModel *model, const char *node_id,
                        const NNPackage *package, const NNParameterDef *definition,
                        char *error, size_t capacity)
{
    NNValue value = {0};
    if (!default_value(app, package, definition, &value))
        return nn_fail(error, capacity, "cannot construct parameter default");
    if (!valid_value(definition, &value, error, capacity)) {
        if (definition->has_default && value.type == NN_VALUE_ARRAY) nn_value_dispose(&value);
        return false;
    }
    bool okay = nn_model_set_parameter(model, node_id, definition->key,
                                       &value, error, capacity);
    if (definition->has_default && value.type == NN_VALUE_ARRAY) nn_value_dispose(&value);
    return okay;
}

NNApplication *nn_app_new(const char *core_root)
{
    if (!core_root || !*core_root) return NULL;
    NNApplication *app = calloc(1, sizeof(*app));
    if (!app) return NULL;
    app->core_root = nn_text_copy(core_root);
    if (!app->core_root) { free(app); return NULL; }
    return app;
}

void nn_app_free(NNApplication *app)
{
    if (!app) return;
    invalidate_analysis(app);
    nn_project_close(app->project);
    free(app->core_root);
    free(app);
}

bool nn_app_open(NNApplication *app, const char *directory, char *error, size_t cap)
{
    if (!app || !directory || !*directory) return nn_fail(error, cap, "invalid project path");
    NNProject *staged = nn_project_open(directory, app->core_root, error, cap);
    if (!staged) return false;
    NNProject *old = app->project;
    app->project = staged;
    invalidate_analysis(app);
    nn_project_close(old);
    nn_error_set(error, cap, "");
    return true;
}

bool nn_app_create(NNApplication *app, const char *parent, const char *id,
                   const char *name, bool mnist, char *error, size_t cap)
{
    if (!app) return nn_fail(error, cap, "application is null");
    NNProject *staged = nn_project_create(parent, id, name, mnist,
                                          app->core_root, error, cap);
    if (!staged) return false;
    NNProject *old = app->project;
    app->project = staged;
    invalidate_analysis(app);
    nn_project_close(old);
    nn_error_set(error, cap, "");
    return true;
}

bool nn_app_create_stereotype(NNApplication *app, const char *id, const char *version,
                              const char *definition_json, const char *inference_lua,
                              const char *dependencies_json, char *error, size_t cap)
{
    nn_error_set(error, cap, "");
    if (!app || !app->project) return nn_fail(error, cap, "no active project");
    if (!id || !version || !definition_json || !inference_lua || strlen(inference_lua) > 1024 * 1024)
        return nn_fail(error, cap, "invalid stereotype payload");
    if (!nn_inference_validate_source(inference_lua, error, cap)) return false;
    yyjson_doc *definition = yyjson_read(definition_json, strlen(definition_json), 0);
    yyjson_doc *dependencies = dependencies_json ? yyjson_read(dependencies_json, strlen(dependencies_json), 0) : NULL;
    if (!definition || !yyjson_is_obj(yyjson_doc_get_root(definition)) ||
        (dependencies_json && (!dependencies || !yyjson_is_obj(yyjson_doc_get_root(dependencies))))) {
        if (definition) yyjson_doc_free(definition);
        if (dependencies) yyjson_doc_free(dependencies);
        return nn_fail(error, cap, "definition or dependencies must be JSON objects");
    }
    yyjson_val *root = yyjson_doc_get_root(definition);
    const char *kind = yyjson_get_str(yyjson_obj_get(root, "kind"));
    bool valid_kind = kind && (!strcmp(kind, "input") || !strcmp(kind, "layer") ||
        !strcmp(kind, "join") || !strcmp(kind, "loss") ||
        !strcmp(kind, "output") || !strcmp(kind, "loss-output") ||
        !strcmp(kind, "subflow"));
    yyjson_val *parameters = yyjson_obj_get(root, "parameters");
    bool valid = valid_kind && yyjson_is_obj(parameters);
    if (valid) {
        yyjson_obj_iter iter = yyjson_obj_iter_with(parameters); yyjson_val *key;
        while ((key = yyjson_obj_iter_next(&iter))) {
            const char *name = yyjson_get_str(key);
            yyjson_val *def = yyjson_obj_iter_get_val(key);
            const char *type = yyjson_get_str(yyjson_obj_get(def, "type"));
            if (!name || !*name || !type || (strcmp(type,"boolean") && strcmp(type,"integer") &&
                strcmp(type,"number") && strcmp(type,"string") && strcmp(type,"dtype") && strcmp(type,"json"))) { valid = false; break; }
            yyjson_val *position = yyjson_obj_get(def, "position");
            const char *pos = yyjson_get_str(position);
            if (position && (!pos || (strcmp(pos,"top") && strcmp(pos,"bottom")))) { valid = false; break; }
            yyjson_val *minimum = yyjson_obj_get(def,"minimum");
            if (minimum && (!yyjson_is_num(minimum) || !isfinite(yyjson_get_num(minimum)) || (strcmp(type,"integer") && strcmp(type,"number")))) { valid = false; break; }
            if (minimum && !strcmp(type, "integer") && yyjson_get_num(minimum) > (double)LLONG_MAX) { valid = false; break; }
            yyjson_val *choices = yyjson_obj_get(def,"choices");
            if (choices && (!yyjson_is_arr(choices) || !yyjson_arr_size(choices) || (strcmp(type,"string") && strcmp(type,"dtype")))) { valid = false; break; }
            if (choices) {
                for (size_t i=0; i<yyjson_arr_size(choices); ++i)
                    if (!yyjson_is_str(yyjson_arr_get(choices,i))) { valid=false; break; }
                if (!valid) break;
            }
            yyjson_val *default_value = yyjson_obj_get(def, "default");
            if (!default_value && !strcmp(type, "json")) { valid = false; break; }
            if (minimum && !strcmp(type, "integer") && floor(yyjson_get_num(minimum)) != yyjson_get_num(minimum)) { valid=false; break; }
            if (default_value) {
                if (!strcmp(type, "integer") && yyjson_is_uint(default_value) &&
                    yyjson_get_uint(default_value) > LLONG_MAX) { valid = false; break; }
                bool typed = (!strcmp(type,"boolean") && yyjson_is_bool(default_value)) ||
                    (!strcmp(type,"integer") && yyjson_is_int(default_value)) ||
                    (!strcmp(type,"number") && yyjson_is_num(default_value)) ||
                    ((!strcmp(type,"string") || !strcmp(type,"dtype")) && yyjson_is_str(default_value)) ||
                    (!strcmp(type,"json") && yyjson_is_arr(default_value));
                if (!typed || (yyjson_is_num(default_value) && !isfinite(yyjson_get_num(default_value)))) { valid=false; break; }
                if (minimum && yyjson_is_num(default_value) && yyjson_get_num(default_value) < yyjson_get_num(minimum)) { valid=false; break; }
                if (choices) {
                    bool found=false;
                    for(size_t i=0;i<yyjson_arr_size(choices);++i)
                        if(!strcmp(yyjson_get_str(default_value), yyjson_get_str(yyjson_arr_get(choices,i)))) found=true;
                    if(!found) { valid=false; break; }
                }
                if (!strcmp(type,"json")) {
                    NNValue parsed={0};
                    if(!parse_json_value(default_value,&parsed,0)) { valid=false; break; }
                    nn_value_dispose(&parsed);
                }
            }
            if (!valid) break;
        }
    }
    yyjson_doc_free(definition); if (dependencies) yyjson_doc_free(dependencies);
    if (!valid) return nn_fail(error, cap, "invalid stereotype schema, parameter, type or position");
    bool okay = nn_project_create_stereotype(app->project, id, version, definition_json,
                                             inference_lua, dependencies_json, error, cap);
    if (okay) { invalidate_analysis(app); nn_error_set(error, cap, ""); }
    return okay;
}

bool nn_app_create_dataset(NNApplication *app, const char *id, const char *version,
                           const char *definition_json, bool select,
                           char *error, size_t cap)
{
    nn_error_set(error, cap, "");
    if (!app || !app->project) return nn_fail(error, cap, "no active project");
    if (!id || !version || !definition_json) return nn_fail(error, cap, "invalid dataset payload");
    yyjson_doc *doc = yyjson_read(definition_json, strlen(definition_json), 0);
    if (!doc || !yyjson_is_obj(yyjson_doc_get_root(doc))) { if (doc) yyjson_doc_free(doc); return nn_fail(error, cap, "dataset definition must be an object"); }
    yyjson_val *root = yyjson_doc_get_root(doc), *batch = yyjson_obj_get(root, "batch");
    const char *dtype_names[] = {"float16","bfloat16","float32","float64","int8","int16","int32","int64","uint8","bool"};
    bool valid = yyjson_is_obj(batch);
    const char *parts[] = {"inputs","targets"};
    const char *seen_slots[128];
    size_t seen_count = 0;
    for (size_t p=0; valid && p<2; ++p) {
        yyjson_val *slots = yyjson_obj_get(batch, parts[p]);
        if (!yyjson_is_obj(slots)) { valid=false; break; }
        size_t count=yyjson_obj_size(slots);
        if (count > 64 || (p==0 && count==0)) { valid=false; break; }
        yyjson_obj_iter it=yyjson_obj_iter_with(slots); yyjson_val *key;
        while ((key=yyjson_obj_iter_next(&it))) {
            const char *slot=yyjson_get_str(key); yyjson_val *spec=yyjson_obj_iter_get_val(key);
            const char *dtype=yyjson_get_str(yyjson_obj_get(spec,"dtype")); yyjson_val *shape=yyjson_obj_get(spec,"shape");
            bool dtype_ok=false; for(size_t i=0;dtype && i<10;++i) if(!strcmp(dtype,dtype_names[i])) dtype_ok=true;
            if(!slot || !*slot || !dtype_ok || !yyjson_is_arr(shape) || !yyjson_arr_size(shape) || yyjson_arr_size(shape)>64) { valid=false; break; }
            for (size_t i = 0; i < seen_count; ++i)
                if (!strcmp(slot, seen_slots[i])) { valid = false; break; }
            if (!valid) break;
            seen_slots[seen_count++] = slot;
            for(size_t i=0;i<yyjson_arr_size(shape);++i) { yyjson_val *dim=yyjson_arr_get(shape,i); if(!(yyjson_is_uint(dim) && yyjson_get_uint(dim)>0) && !(yyjson_is_str(dim) && !strcmp(yyjson_get_str(dim),"B"))) { valid=false; break; } }
            if(!valid) break;
        }
    }
    yyjson_doc_free(doc);
    if (!valid) return nn_fail(error, cap, "invalid dataset slots, dtype or shape");
    bool okay = nn_project_create_dataset(app->project, id, version, definition_json, select, error, cap);
    if (okay) { invalidate_analysis(app); nn_error_set(error, cap, ""); }
    return okay;
}

bool nn_app_select_dataset(NNApplication *app, const char *id, const char *version,
                           char *error, size_t cap)
{
    nn_error_set(error, cap, "");
    if (!app || !app->project) return nn_fail(error, cap, "no active project");
    bool okay = nn_project_select_dataset(app->project, id, version, error, cap);
    if (okay) invalidate_analysis(app);
    return okay;
}

bool nn_app_create_vae(NNApplication *app, const char *parent, const char *id,
                       const char *name, char *error, size_t cap)
{
    if (!app) return nn_fail(error, cap, "application is null");
    NNProject *staged = NULL;
    if (!nn_project_create_vae(parent, id, name, app->core_root, &staged, error, cap)) return false;
    NNProject *old = app->project; app->project = staged; invalidate_analysis(app); nn_project_close(old);
    nn_error_set(error, cap, ""); return true;
}

bool nn_app_save(NNApplication *app, char *error, size_t cap)
{
    if (!app || !app->project) return nn_fail(error, cap, "no active project");
    return nn_project_save(app->project, error, cap);
}

bool nn_app_close(NNApplication *app, bool discard, char *error, size_t cap)
{
    if (!app) return nn_fail(error, cap, "application is null");
    if (!app->project) { nn_error_set(error, cap, ""); return true; }
    if (nn_project_dirty(app->project) && !discard &&
        !nn_project_save(app->project, error, cap)) return false;
    nn_project_close(app->project);
    app->project = NULL;
    invalidate_analysis(app);
    nn_error_set(error, cap, "");
    return true;
}

const NNProject *nn_app_project(const NNApplication *app)
{
    return app ? app->project : NULL;
}

const NNModel *nn_app_model(const NNApplication *app)
{
    return app && app->project ? nn_project_model(app->project) : NULL;
}

const NNInferenceReport *nn_app_analysis(NNApplication *app, char *error, size_t cap)
{
    if (!app || !app->project) {
        nn_fail(error, cap, "no active project");
        return NULL;
    }
    if (!app->analysis) {
        app->analysis = nn_infer_project(app->project);
        if (!app->analysis) {
            nn_fail(error, cap, "unable to construct project analysis report");
            return NULL;
        }
    }
    nn_error_set(error, cap, "");
    return app->analysis;
}

bool nn_app_add_node(NNApplication *app, const char *id, const char *package_id,
                     const char *version, const char *scope, double x, double y,
                     char *error, size_t cap)
{
    if (!app || !app->project) return nn_fail(error, cap, "no active project");
    if (!id || !*id || !package_id || !version || !scope || !isfinite(x) || !isfinite(y))
        return nn_fail(error, cap, "invalid node fields or non-finite position");
    const NNPackage *package = nn_catalog_find(nn_project_catalog(app->project),
                                                package_id, version);
    if (!package) return nn_fail(error, cap, "package is not active in this project");
    NNModel *model = nn_project_model(app->project);
    if (*scope) {
        const NNNode *owner = nn_model_find_node(model, scope);
        if (!owner || !kind_is(find_package(app, owner), "subflow"))
            return nn_fail(error, cap, "scope must name an existing subflow");
    }
    char local_error[256] = "";
    if (!nn_model_add_node(model, id, package->name, package->id, package->version,
                           scope, x, y, local_error, sizeof(local_error)))
        return nn_fail(error, cap, local_error);
    for (size_t i = 0; i < package->parameter_count; ++i) {
        const NNParameterDef *definition = &package->parameters[i];
        /* Preserve the existing editor behavior for object-valued defaults. */
        if (!strcmp(definition->type, "stereotype") && definition->has_default &&
            definition->default_value.type == NN_PARAMETER_JSON) continue;
        if (!add_default(app, model, id, package, definition, error, cap)) {
            char ignored[64];
            (void)nn_model_remove_node(model, id, ignored, sizeof(ignored));
            return false;
        }
    }
    if (kind_is(package, "subflow")) {
        char spawned_ids[2][256] = {{0}};
        size_t spawned = 0;
        for (size_t i = 0; i < package->output_count; ++i) {
            const NNOutputDef *output = &package->outputs[i];
            const char *terminal_id = !strcmp(output->type, "loss")
                ? "core.loss-output" : "core.output";
            const NNCatalog *catalog = nn_project_catalog(app->project);
            const NNPackage *terminal = nn_catalog_find(catalog, terminal_id, "0.1.0");
            bool unique = false;
            bool current_created = false;
            double terminal_x = x + 220.0;
            double terminal_y = y + (double)spawned * 100.0;
            bool position_valid = isfinite(terminal_x) && isfinite(terminal_y);
            char failure[256] = "";
            if (!position_valid)
                nn_error_set(failure, sizeof(failure), "subflow terminal position must be finite");
            for (unsigned attempt = 0; attempt < 10000 && !unique; ++attempt) {
                int n = attempt
                    ? snprintf(spawned_ids[spawned], sizeof(spawned_ids[spawned]),
                               "%s-boundary-%s-%u", id, output->id, attempt)
                    : snprintf(spawned_ids[spawned], sizeof(spawned_ids[spawned]),
                               "%s-boundary-%s", id, output->id);
                if (n < 0 || (size_t)n >= sizeof(spawned_ids[spawned])) {
                    nn_error_set(failure, sizeof(failure), "generated subflow terminal ID is too long");
                    break;
                }
                unique = nn_model_find_node(model, spawned_ids[spawned]) == NULL;
                for (size_t e = 0; unique && e < nn_model_edge_count(model); ++e)
                    if (!strcmp(nn_model_edge_at(model, e)->id, spawned_ids[spawned])) unique = false;
            }
            if (!terminal && !failure[0])
                nn_error_set(failure, sizeof(failure), "required boundary package is unavailable");
            if (!unique && !failure[0])
                nn_error_set(failure, sizeof(failure), "unable to generate a unique subflow terminal ID");
            if (!position_valid && !failure[0])
                nn_error_set(failure, sizeof(failure), "subflow terminal position must be finite");
            bool ready = terminal && unique && position_valid;
            if (ready) {
                bool added = nn_model_add_node(model, spawned_ids[spawned], output->id,
                                               terminal->id, terminal->version, id,
                                               terminal_x, terminal_y,
                                               failure, sizeof(failure));
                current_created = added;
                bool mapped = added && nn_model_set_boundary_handle(
                    model, spawned_ids[spawned], output->id, failure, sizeof(failure));
                ready = added && mapped;
            }
            if (!ready) {
                char ignored[64];
                if (current_created)
                    (void)nn_model_remove_node(model, spawned_ids[spawned], ignored, sizeof(ignored));
                while (spawned) (void)nn_model_remove_node(model, spawned_ids[--spawned], ignored, sizeof(ignored));
                (void)nn_model_remove_node(model, id, ignored, sizeof(ignored));
                if (!failure[0])
                    nn_error_set(failure, sizeof(failure), "unable to spawn subflow output terminal");
                return nn_fail(error, cap, failure);
            }
            ++spawned;
        }
    }
    nn_project_mark_dirty(app->project);
    invalidate_analysis(app);
    nn_error_set(error, cap, "");
    return true;
}

bool nn_app_remove_node(NNApplication *app, const char *id, char *error, size_t cap)
{
    if (!app || !app->project || !id) return nn_fail(error, cap, "no active project or invalid node ID");
    NNModel *model = nn_project_model(app->project);
    const NNNode *node = nn_model_find_node(model, id);
    if (!node) return nn_fail(error, cap, "node not found");
    if (kind_is(find_package(app, node), "subflow")) {
        for (size_t i = 0; i < nn_model_node_count(model); ++i) {
            const NNNode *child = nn_model_node_at(model, i);
            if (!strcmp(child->scope_id, id))
                return nn_fail(error, cap, "subflow still contains nodes");
        }
    }
    if (!nn_model_remove_node(model, id, error, cap)) return false;
    nn_project_mark_dirty(app->project);
    invalidate_analysis(app);
    nn_error_set(error, cap, "");
    return true;
}

bool nn_app_move_node(NNApplication *app, const char *id, double x, double y,
                      char *error, size_t cap)
{
    if (!app || !app->project) return nn_fail(error, cap, "no active project");
    if (!isfinite(x) || !isfinite(y)) return nn_fail(error, cap, "position must be finite");
    if (!nn_model_move_node(nn_project_model(app->project), id, x, y, error, cap)) return false;
    nn_project_mark_dirty(app->project);
    nn_error_set(error, cap, "");
    return true;
}

bool nn_app_rename_node(NNApplication *app, const char *id, const char *label,
                        char *error, size_t cap)
{
    if (!app || !app->project) return nn_fail(error, cap, "no active project");
    if (!nn_model_rename_node(nn_project_model(app->project), id, label, error, cap)) return false;
    nn_project_mark_dirty(app->project);
    nn_error_set(error, cap, "");
    return true;
}

static bool valid_output_handle(const NNPackage *package, const char *handle)
{
    if (!package || !handle) return false;
    for (size_t i = 0; i < package->output_count; ++i)
        if (!strcmp(package->outputs[i].id, handle)) return true;
    return false;
}

static bool valid_input_handle(const NNPackage *package, const char *handle)
{
    if (kind_is(package, "input")) return false;
    if (kind_is(package, "join")) {
        size_t suffix = 0;
        return nn_join_handle_order(handle, &suffix);
    }
    return handle && !strcmp(handle, "in");
}

bool nn_app_connect(NNApplication *app, const char *id, const char *source,
                    const char *source_handle, const char *target,
                    const char *target_handle, char *error, size_t cap)
{
    if (!app || !app->project) return nn_fail(error, cap, "no active project");
    NNModel *model = nn_project_model(app->project);
    const NNNode *source_node = nn_model_find_node(model, source);
    const NNNode *target_node = nn_model_find_node(model, target);
    if (!source_node || !target_node) return nn_fail(error, cap, "edge endpoint not found");
    const NNPackage *source_package = find_package(app, source_node);
    const NNPackage *target_package = find_package(app, target_node);
    if (!source_package || !target_package) return nn_fail(error, cap, "edge package is unresolved");
    if (!valid_output_handle(source_package, source_handle))
        return nn_fail(error, cap, "invalid output handle");
    if (!valid_input_handle(target_package, target_handle))
        return nn_fail(error, cap, "invalid input handle");
    const char *type = nn_app_output_type(app, source, source_handle);
    if ((kind_is(target_package, "output") && (!type || strcmp(type, "output"))) ||
        (kind_is(target_package, "loss-output") && (!type || strcmp(type, "loss"))))
        return nn_fail(error, cap, "output type is incompatible with terminal");
    if (kind_is(target_package, "join")) {
        size_t requested;
        (void)nn_join_handle_order(target_handle, &requested);
        for (size_t i = 0; i < nn_model_edge_count(model); ++i) {
            const NNEdge *edge = nn_model_edge_at(model, i);
            size_t occupied;
            if (!strcmp(edge->target_id, target) &&
                nn_join_handle_order(edge->target_handle_id, &occupied) && occupied == requested)
                return nn_fail(error, cap, "join input position is already occupied");
        }
    }
    if (!nn_model_connect(model, id, source, source_handle, target, target_handle, error, cap))
        return false;
    nn_project_mark_dirty(app->project);
    invalidate_analysis(app);
    nn_error_set(error, cap, "");
    return true;
}

bool nn_app_disconnect(NNApplication *app, const char *id, char *error, size_t cap)
{
    if (!app || !app->project) return nn_fail(error, cap, "no active project");
    if (!nn_model_disconnect(nn_project_model(app->project), id, error, cap)) return false;
    nn_project_mark_dirty(app->project);
    invalidate_analysis(app);
    nn_error_set(error, cap, "");
    return true;
}

bool nn_app_set_parameter(NNApplication *app, const char *node_id, const char *key,
                          const NNValue *value, char *error, size_t cap)
{
    if (!app || !app->project) return nn_fail(error, cap, "no active project");
    const NNNode *node = nn_model_find_node(nn_project_model(app->project), node_id);
    if (!node) return nn_fail(error, cap, "node not found");
    const NNPackage *package = find_package(app, node);
    const NNParameterDef *definition = NULL;
    if (!parameter_definition(package, key, &definition))
        return nn_fail(error, cap, "unknown parameter");
    if (!valid_value(definition, value, error, cap)) return false;
    if (!nn_model_set_parameter(nn_project_model(app->project), node_id, key,
                                value, error, cap)) return false;
    nn_project_mark_dirty(app->project);
    invalidate_analysis(app);
    nn_error_set(error, cap, "");
    return true;
}

bool nn_app_set_parameter_text(NNApplication *app, const char *node, const char *key,
                               const char *text, char *error, size_t cap)
{
    if (!app || !app->project) return nn_fail(error, cap, "no active project");
    const NNNode *entry = nn_model_find_node(nn_project_model(app->project), node);
    const NNParameterDef *definition = NULL;
    if (!entry) return nn_fail(error, cap, "node not found");
    if (!parameter_definition(find_package(app, entry), key, &definition))
        return nn_fail(error, cap, "unknown parameter");
    if (!strcmp(definition->type, "stereotype"))
        return nn_fail(error, cap, "object-valued stereotype parameters are not supported by native model");
    NNValue value = {0};
    if (!parse_parameter_text(definition, text, &value, error, cap)) return false;
    bool okay = nn_app_set_parameter(app, node, key, &value, error, cap);
    if (value.type == NN_VALUE_ARRAY) nn_value_dispose(&value);
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

typedef struct { const char *id; size_t number; bool numeric, owned; } JoinHandle;

static int compare_join_handle(const void *left, const void *right)
{
    const JoinHandle *a = left, *b = right;
    if (a->numeric != b->numeric) return a->numeric ? -1 : 1;
    if (a->numeric && a->number != b->number) return a->number < b->number ? -1 : 1;
    return strcmp(a->id, b->id);
}

static bool join_number_used(const JoinHandle *handles, size_t count, size_t number)
{
    for (size_t i = 0; i < count; ++i)
        if (handles[i].numeric && handles[i].number == number) return true;
    return false;
}

static void join_handles_free(JoinHandle *handles, size_t count)
{
    if (!handles) return;
    for (size_t i = 0; i < count; ++i)
        if (handles[i].owned) free((void *)handles[i].id);
    free(handles);
}

static JoinHandle *join_inputs(const NNModel *model, const char *node_id, size_t *count)
{
    size_t edge_count = nn_model_edge_count(model);
    if (edge_count > (SIZE_MAX / sizeof(JoinHandle)) - 2) return NULL;
    JoinHandle *handles = calloc(edge_count + 2, sizeof(*handles));
    if (!handles) return NULL;
    size_t used = 0;
    for (size_t i = 0; i < edge_count; ++i) {
        const NNEdge *edge = nn_model_edge_at(model, i);
        size_t suffix = 0;
        if (strcmp(edge->target_id, node_id)) continue;
        bool numeric = nn_join_handle_order(edge->target_handle_id, &suffix);
        bool duplicate = false;
        for (size_t j = 0; j < used; ++j)
            if (!strcmp(handles[j].id, edge->target_handle_id)) duplicate = true;
        if (!duplicate) handles[used++] = (JoinHandle){ edge->target_handle_id, suffix, numeric, false };
    }
    size_t existing = used;
    size_t candidate = 1;
    size_t desired = existing < 2 ? 2 : existing + 1;
    while (used < desired) {
        while (join_number_used(handles, existing, candidate)) {
            if (candidate == SIZE_MAX) { join_handles_free(handles, used); return NULL; }
            ++candidate;
        }
        char *generated = malloc(3 + 3 * sizeof(size_t) + 1);
        if (!generated) { join_handles_free(handles, used); return NULL; }
        snprintf(generated, 3 + 3 * sizeof(size_t) + 1, "in-%zu", candidate);
        handles[used++] = (JoinHandle){ generated, candidate, true, true };
        if (candidate == SIZE_MAX) { join_handles_free(handles, used); return NULL; }
        ++candidate;
    }
    qsort(handles, used, sizeof(*handles), compare_join_handle);
    *count = used;
    return handles;
}

size_t nn_app_port_count(const NNApplication *app, const char *node_id, bool output)
{
    const NNModel *model = nn_app_model(app);
    const NNNode *node = model ? nn_model_find_node(model, node_id) : NULL;
    const NNPackage *package = find_package(app, node);
    if (!package) return 0;
    if (output) return package->output_count;
    if (kind_is(package, "input")) return 0;
    if (kind_is(package, "join")) {
        size_t count = 0;
        JoinHandle *inputs = join_inputs(model, node_id, &count);
        join_handles_free(inputs, count);
        return count;
    }
    return 1;
}

bool nn_app_port_id(const NNApplication *app, const char *node_id, bool output,
                    size_t index, char *buffer, size_t capacity)
{
    const NNModel *model = nn_app_model(app);
    const NNNode *node = model ? nn_model_find_node(model, node_id) : NULL;
    const NNPackage *package = find_package(app, node);
    if (!package || !buffer || !capacity) return false;
    if (output) {
        if (index >= package->output_count) return false;
        int written = snprintf(buffer, capacity, "%s", package->outputs[index].id);
        return written >= 0 && (size_t)written < capacity;
    }
    if (kind_is(package, "input")) return false;
    if (!kind_is(package, "join")) {
        if (index != 0) return false;
        return snprintf(buffer, capacity, "in") < (int)capacity;
    }
    size_t count = 0;
    JoinHandle *inputs = join_inputs(model, node_id, &count);
    if (!inputs || index >= count) { join_handles_free(inputs, count); return false; }
    int written = snprintf(buffer, capacity, "%s", inputs[index].id);
    join_handles_free(inputs, count);
    return written >= 0 && (size_t)written < capacity;
}

bool nn_app_node_is_subflow(const NNApplication *app, const char *node_id)
{
    const NNModel *model = nn_app_model(app);
    const NNNode *node = model ? nn_model_find_node(model, node_id) : NULL;
    return kind_is(find_package(app, node), "subflow");
}

const char *nn_app_output_type(const NNApplication *app, const char *node_id,
                               const char *handle_id)
{
    const NNModel *model = nn_app_model(app);
    const NNNode *node = model ? nn_model_find_node(model, node_id) : NULL;
    const NNPackage *package = find_package(app, node);
    if (!package || !handle_id) return NULL;
    for (size_t i = 0; i < package->output_count; ++i)
        if (!strcmp(package->outputs[i].id, handle_id)) return package->outputs[i].type;
    return NULL;
}

bool nn_app_set_boundary_handle(NNApplication *app, const char *node_id,
                                const char *handle_id, char *error, size_t cap)
{
    if (!app || !app->project || !node_id || !handle_id || !*handle_id)
        return nn_fail(error, cap, "invalid boundary mapping");
    NNModel *model = nn_project_model(app->project);
    const NNNode *node = nn_model_find_node(model, node_id);
    const NNPackage *terminal = find_package(app, node);
    if (!node || (!kind_is(terminal, "output") && !kind_is(terminal, "loss-output")))
        return nn_fail(error, cap, "boundary mapping requires an output terminal");
    const NNNode *owner = *node->scope_id ? nn_model_find_node(model, node->scope_id) : NULL;
    const NNPackage *owner_package = find_package(app, owner);
    if (!owner || !kind_is(owner_package, "subflow"))
        return nn_fail(error, cap, "root terminals cannot have boundary mappings");
    const NNOutputDef *mapping = NULL;
    for (size_t i = 0; i < owner_package->output_count; ++i)
        if (!strcmp(owner_package->outputs[i].id, handle_id)) mapping = &owner_package->outputs[i];
    if (!mapping) return nn_fail(error, cap, "unknown subflow output handle");
    const char *expected_kind = !strcmp(mapping->type, "loss") ? "loss-output" : "output";
    if (!kind_is(terminal, expected_kind))
        return nn_fail(error, cap, "boundary handle type does not match terminal");
    if (node->boundary_handle_id && !strcmp(node->boundary_handle_id, handle_id)) {
        nn_error_set(error, cap, "");
        return true;
    }
    if (!nn_model_set_boundary_handle(model, node_id, handle_id, error, cap)) return false;
    nn_project_mark_dirty(app->project);
    invalidate_analysis(app);
    nn_error_set(error, cap, "");
    return true;
}
