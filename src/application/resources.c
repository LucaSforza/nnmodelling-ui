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
                    if(!nn_app_parse_json_value(default_value,&parsed,0)) { valid=false; break; }
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
    if (okay) { nn_app_invalidate_analysis(app); nn_error_set(error, cap, ""); }
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
    if (okay) { nn_app_invalidate_analysis(app); nn_error_set(error, cap, ""); }
    return okay;
}

bool nn_app_select_dataset(NNApplication *app, const char *id, const char *version,
                           char *error, size_t cap)
{
    nn_error_set(error, cap, "");
    if (!app || !app->project) return nn_fail(error, cap, "no active project");
    bool okay = nn_project_select_dataset(app->project, id, version, error, cap);
    if (okay) nn_app_invalidate_analysis(app);
    return okay;
}

bool nn_app_create_vae(NNApplication *app, const char *parent, const char *id,
                       const char *name, char *error, size_t cap)
{
    if (!app) return nn_fail(error, cap, "application is null");
    NNProject *staged = NULL;
    if (!nn_project_create_vae(parent, id, name, app->core_root, &staged, error, cap)) return false;
    NNProject *old = app->project; app->project = staged; nn_app_invalidate_analysis(app); nn_project_close(old);
    nn_error_set(error, cap, ""); return true;
}
