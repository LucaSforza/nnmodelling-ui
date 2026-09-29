#include "application.h"

#include "yyjson.h"

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
};

static void set_error(char *error, size_t capacity, const char *message)
{
    if (error && capacity) snprintf(error, capacity, "%s", message ? message : "");
}

static bool fail(char *error, size_t capacity, const char *message)
{
    set_error(error, capacity, message);
    return false;
}

static char *copy_text(const char *text)
{
    if (!text) return NULL;
    size_t size = strlen(text) + 1;
    char *copy = malloc(size);
    if (copy) memcpy(copy, text, size);
    return copy;
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
    if (!definition || !value) return fail(error, capacity, "invalid parameter");
    if (!strcmp(definition->type, "boolean")) {
        if (value->type != NN_VALUE_BOOL) return fail(error, capacity, "parameter requires a boolean");
    } else if (!strcmp(definition->type, "integer")) {
        if (value->type != NN_VALUE_INT) return fail(error, capacity, "parameter requires an integer");
        if (definition->has_minimum && (double)value->as.integer < definition->minimum)
            return fail(error, capacity, "integer is below minimum");
    } else if (!strcmp(definition->type, "number")) {
        if (value->type != NN_VALUE_REAL || !isfinite(value->as.real))
            return fail(error, capacity, "parameter requires a finite number");
        if (definition->has_minimum && value->as.real < definition->minimum)
            return fail(error, capacity, "number is below minimum");
    } else if (!strcmp(definition->type, "string") ||
               !strcmp(definition->type, "dtype")) {
        if (value->type != NN_VALUE_STRING || !value->as.string)
            return fail(error, capacity, "parameter requires a string");
        if (definition->choice_count) {
            bool found = false;
            for (size_t i = 0; i < definition->choice_count; ++i)
                if (!strcmp(value->as.string, definition->choices[i])) found = true;
            if (!found) return fail(error, capacity, "value is not an allowed choice");
        }
    } else if (!strcmp(definition->type, "json")) {
        if (value->type != NN_VALUE_ARRAY || !valid_json_value(value, 0))
            return fail(error, capacity, "parameter requires a JSON array");
    } else if (!strcmp(definition->type, "stereotype")) {
        return fail(error, capacity, "object-valued stereotype parameters are not supported by native model");
    } else return fail(error, capacity, "unsupported parameter type");
    set_error(error, capacity, "");
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
        value->type = NN_VALUE_INT;
        value->as.integer = (long long)yyjson_get_sint(source);
    } else if (yyjson_is_num(source)) {
        value->type = NN_VALUE_REAL;
        value->as.real = yyjson_get_num(source);
        if (!isfinite(value->as.real)) return false;
    } else if (yyjson_is_str(source)) {
        value->type = NN_VALUE_STRING;
        value->as.string = copy_text(yyjson_get_str(source));
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
    if (!definition || !text) return fail(error, capacity, "invalid parameter text");
    if (!strcmp(definition->type, "boolean")) {
        if (strcmp(text, "true") && strcmp(text, "false"))
            return fail(error, capacity, "boolean must be true or false");
        value->type = NN_VALUE_BOOL;
        value->as.boolean = !strcmp(text, "true");
    } else if (!strcmp(definition->type, "integer")) {
        char *end = NULL;
        errno = 0;
        value->type = NN_VALUE_INT;
        value->as.integer = strtoll(text, &end, 10);
        if (!*text || !end || *end || errno == ERANGE) return fail(error, capacity, "invalid integer");
    } else if (!strcmp(definition->type, "number")) {
        char *end = NULL;
        errno = 0;
        value->type = NN_VALUE_REAL;
        value->as.real = strtod(text, &end);
        if (!*text || !end || *end || errno == ERANGE || !isfinite(value->as.real))
            return fail(error, capacity, "invalid finite number");
    } else if (!strcmp(definition->type, "string") ||
               !strcmp(definition->type, "dtype")) {
        value->type = NN_VALUE_STRING;
        value->as.string = (char *)text;
    } else if (!strcmp(definition->type, "json")) {
        yyjson_doc *document = yyjson_read_opts((char *)(uintptr_t)text, strlen(text), 0,
                                                 NULL, NULL);
        if (!document) return fail(error, capacity, "invalid JSON array");
        const yyjson_val *root = yyjson_doc_get_root(document);
        bool okay = yyjson_is_arr(root) && parse_json_value(root, value, 0);
        yyjson_doc_free(document);
        if (!okay) {
            nn_value_dispose(value);
            return fail(error, capacity, "parameter requires a JSON array of primitive values");
        }
    } else return fail(error, capacity, "unsupported parameter type");
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
        return fail(error, capacity, "cannot construct parameter default");
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
    app->core_root = copy_text(core_root);
    if (!app->core_root) { free(app); return NULL; }
    return app;
}

void nn_app_free(NNApplication *app)
{
    if (!app) return;
    nn_project_close(app->project);
    free(app->core_root);
    free(app);
}

bool nn_app_open(NNApplication *app, const char *directory, char *error, size_t cap)
{
    if (!app || !directory || !*directory) return fail(error, cap, "invalid project path");
    NNProject *staged = nn_project_open(directory, app->core_root, error, cap);
    if (!staged) return false;
    NNProject *old = app->project;
    app->project = staged;
    nn_project_close(old);
    set_error(error, cap, "");
    return true;
}

bool nn_app_create(NNApplication *app, const char *parent, const char *id,
                   const char *name, bool mnist, char *error, size_t cap)
{
    if (!app) return fail(error, cap, "application is null");
    NNProject *staged = nn_project_create(parent, id, name, mnist,
                                          app->core_root, error, cap);
    if (!staged) return false;
    NNProject *old = app->project;
    app->project = staged;
    nn_project_close(old);
    set_error(error, cap, "");
    return true;
}

bool nn_app_save(NNApplication *app, char *error, size_t cap)
{
    if (!app || !app->project) return fail(error, cap, "no active project");
    return nn_project_save(app->project, error, cap);
}

bool nn_app_close(NNApplication *app, bool discard, char *error, size_t cap)
{
    if (!app) return fail(error, cap, "application is null");
    if (!app->project) { set_error(error, cap, ""); return true; }
    if (nn_project_dirty(app->project) && !discard &&
        !nn_project_save(app->project, error, cap)) return false;
    nn_project_close(app->project);
    app->project = NULL;
    set_error(error, cap, "");
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

bool nn_app_add_node(NNApplication *app, const char *id, const char *package_id,
                     const char *version, const char *scope, double x, double y,
                     char *error, size_t cap)
{
    if (!app || !app->project) return fail(error, cap, "no active project");
    if (!id || !*id || !package_id || !version || !scope || !isfinite(x) || !isfinite(y))
        return fail(error, cap, "invalid node fields or non-finite position");
    const NNPackage *package = nn_catalog_find(nn_project_catalog(app->project),
                                                package_id, version);
    if (!package) return fail(error, cap, "package is not active in this project");
    NNModel *model = nn_project_model(app->project);
    if (*scope) {
        const NNNode *owner = nn_model_find_node(model, scope);
        if (!owner || !kind_is(find_package(app, owner), "subflow"))
            return fail(error, cap, "scope must name an existing subflow");
    }
    char local_error[256] = "";
    if (!nn_model_add_node(model, id, package->name, package->id, package->version,
                           scope, x, y, local_error, sizeof(local_error)))
        return fail(error, cap, local_error);
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
    nn_project_mark_dirty(app->project);
    set_error(error, cap, "");
    return true;
}

bool nn_app_remove_node(NNApplication *app, const char *id, char *error, size_t cap)
{
    if (!app || !app->project || !id) return fail(error, cap, "no active project or invalid node ID");
    NNModel *model = nn_project_model(app->project);
    const NNNode *node = nn_model_find_node(model, id);
    if (!node) return fail(error, cap, "node not found");
    if (kind_is(find_package(app, node), "subflow")) {
        for (size_t i = 0; i < nn_model_node_count(model); ++i) {
            const NNNode *child = nn_model_node_at(model, i);
            if (!strcmp(child->scope_id, id))
                return fail(error, cap, "subflow still contains nodes");
        }
    }
    if (!nn_model_remove_node(model, id, error, cap)) return false;
    nn_project_mark_dirty(app->project);
    set_error(error, cap, "");
    return true;
}

bool nn_app_move_node(NNApplication *app, const char *id, double x, double y,
                      char *error, size_t cap)
{
    if (!app || !app->project) return fail(error, cap, "no active project");
    if (!isfinite(x) || !isfinite(y)) return fail(error, cap, "position must be finite");
    if (!nn_model_move_node(nn_project_model(app->project), id, x, y, error, cap)) return false;
    nn_project_mark_dirty(app->project);
    set_error(error, cap, "");
    return true;
}

bool nn_app_rename_node(NNApplication *app, const char *id, const char *label,
                        char *error, size_t cap)
{
    if (!app || !app->project) return fail(error, cap, "no active project");
    if (!nn_model_rename_node(nn_project_model(app->project), id, label, error, cap)) return false;
    nn_project_mark_dirty(app->project);
    set_error(error, cap, "");
    return true;
}

static bool join_suffix(const char *handle, size_t *suffix)
{
    if (!handle || strncmp(handle, "in-", 3) || !handle[3] || handle[3] == '0') return false;
    size_t number = 0;
    for (const char *p = handle + 3; *p; ++p) {
        if (*p < '0' || *p > '9') return false;
        unsigned digit = (unsigned)(*p - '0');
        if (number > (SIZE_MAX - digit) / 10) return false;
        number = number * 10 + digit;
    }
    if (!number) return false;
    *suffix = number;
    return true;
}

static bool valid_output_handle(const NNPackage *package, const char *handle)
{
    return !kind_is(package, "output") && handle && !strcmp(handle, "out");
}

static bool valid_input_handle(const NNPackage *package, const char *handle)
{
    if (kind_is(package, "input")) return false;
    if (kind_is(package, "join")) {
        size_t suffix = 0;
        return join_suffix(handle, &suffix) && handle[3] != '0';
    }
    return handle && !strcmp(handle, "in");
}

bool nn_app_connect(NNApplication *app, const char *id, const char *source,
                    const char *source_handle, const char *target,
                    const char *target_handle, char *error, size_t cap)
{
    if (!app || !app->project) return fail(error, cap, "no active project");
    NNModel *model = nn_project_model(app->project);
    const NNNode *source_node = nn_model_find_node(model, source);
    const NNNode *target_node = nn_model_find_node(model, target);
    if (!source_node || !target_node) return fail(error, cap, "edge endpoint not found");
    const NNPackage *source_package = find_package(app, source_node);
    const NNPackage *target_package = find_package(app, target_node);
    if (!source_package || !target_package) return fail(error, cap, "edge package is unresolved");
    if (!valid_output_handle(source_package, source_handle))
        return fail(error, cap, "invalid output handle");
    if (!valid_input_handle(target_package, target_handle))
        return fail(error, cap, "invalid input handle");
    if (kind_is(target_package, "join")) {
        size_t requested;
        (void)join_suffix(target_handle, &requested);
        for (size_t i = 0; i < nn_model_edge_count(model); ++i) {
            const NNEdge *edge = nn_model_edge_at(model, i);
            size_t occupied;
            if (!strcmp(edge->target_id, target) &&
                join_suffix(edge->target_handle_id, &occupied) && occupied == requested)
                return fail(error, cap, "join input position is already occupied");
        }
    }
    if (!nn_model_connect(model, id, source, source_handle, target, target_handle, error, cap))
        return false;
    nn_project_mark_dirty(app->project);
    set_error(error, cap, "");
    return true;
}

bool nn_app_disconnect(NNApplication *app, const char *id, char *error, size_t cap)
{
    if (!app || !app->project) return fail(error, cap, "no active project");
    if (!nn_model_disconnect(nn_project_model(app->project), id, error, cap)) return false;
    nn_project_mark_dirty(app->project);
    set_error(error, cap, "");
    return true;
}

bool nn_app_set_parameter(NNApplication *app, const char *node_id, const char *key,
                          const NNValue *value, char *error, size_t cap)
{
    if (!app || !app->project) return fail(error, cap, "no active project");
    const NNNode *node = nn_model_find_node(nn_project_model(app->project), node_id);
    if (!node) return fail(error, cap, "node not found");
    const NNPackage *package = find_package(app, node);
    const NNParameterDef *definition = NULL;
    if (!parameter_definition(package, key, &definition))
        return fail(error, cap, "unknown parameter");
    if (!valid_value(definition, value, error, cap)) return false;
    if (!nn_model_set_parameter(nn_project_model(app->project), node_id, key,
                                value, error, cap)) return false;
    nn_project_mark_dirty(app->project);
    set_error(error, cap, "");
    return true;
}

bool nn_app_set_parameter_text(NNApplication *app, const char *node, const char *key,
                               const char *text, char *error, size_t cap)
{
    if (!app || !app->project) return fail(error, cap, "no active project");
    const NNNode *entry = nn_model_find_node(nn_project_model(app->project), node);
    const NNParameterDef *definition = NULL;
    if (!entry) return fail(error, cap, "node not found");
    if (!parameter_definition(find_package(app, entry), key, &definition))
        return fail(error, cap, "unknown parameter");
    if (!strcmp(definition->type, "stereotype"))
        return fail(error, cap, "object-valued stereotype parameters are not supported by native model");
    NNValue value = {0};
    if (!parse_parameter_text(definition, text, &value, error, cap)) return false;
    bool okay = nn_app_set_parameter(app, node, key, &value, error, cap);
    if (value.type == NN_VALUE_ARRAY) nn_value_dispose(&value);
    return okay;
}

typedef struct { char *data; size_t length, capacity; } TextBuffer;

static bool text_append(TextBuffer *buffer, const char *text, size_t length)
{
    if (length > SIZE_MAX - buffer->length - 1) return false;
    size_t needed = buffer->length + length + 1;
    if (needed > buffer->capacity) {
        size_t next = buffer->capacity ? buffer->capacity : 64;
        while (next < needed) {
            if (next > SIZE_MAX / 2) { next = needed; break; }
            next *= 2;
        }
        char *grown = realloc(buffer->data, next);
        if (!grown) return false;
        buffer->data = grown;
        buffer->capacity = next;
    }
    memcpy(buffer->data + buffer->length, text, length);
    buffer->length += length;
    buffer->data[buffer->length] = '\0';
    return true;
}

static bool json_string(TextBuffer *buffer, const char *text)
{
    if (!text_append(buffer, "\"", 1)) return false;
    for (const unsigned char *p = (const unsigned char *)text; *p; ++p) {
        char escaped[7];
        const char *out = (const char *)p;
        size_t length = 1;
        if (*p == '"' || *p == '\\') { escaped[0] = '\\'; escaped[1] = (char)*p; out = escaped; length = 2; }
        else if (*p < 0x20) { snprintf(escaped, sizeof(escaped), "\\u%04x", *p); out = escaped; length = 6; }
        if (!text_append(buffer, out, length)) return false;
    }
    return text_append(buffer, "\"", 1);
}

static bool serialize_value(TextBuffer *buffer, const NNValue *value)
{
    char number[64];
    switch (value->type) {
    case NN_VALUE_BOOL: return text_append(buffer, value->as.boolean ? "true" : "false", value->as.boolean ? 4 : 5);
    case NN_VALUE_INT:
        snprintf(number, sizeof(number), "%lld", value->as.integer);
        return text_append(buffer, number, strlen(number));
    case NN_VALUE_REAL:
        snprintf(number, sizeof(number), "%.17g", value->as.real);
        return text_append(buffer, number, strlen(number));
    case NN_VALUE_STRING: return value->as.string && json_string(buffer, value->as.string);
    case NN_VALUE_ARRAY:
        if (!text_append(buffer, "[", 1)) return false;
        for (size_t i = 0; i < value->as.array.count; ++i) {
            if (i && !text_append(buffer, ",", 1)) return false;
            if (!serialize_value(buffer, &value->as.array.items[i])) return false;
        }
        return text_append(buffer, "]", 1);
    default: return false;
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
            return copy_text(node->parameters[i].value.as.string);
        TextBuffer buffer = {0};
        if (!serialize_value(&buffer, &node->parameters[i].value)) { free(buffer.data); return NULL; }
        return buffer.data;
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
        bool numeric = join_suffix(edge->target_handle_id, &suffix);
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
            if (candidate == SIZE_MAX) { free(handles); return NULL; }
            ++candidate;
        }
        char *generated = malloc(3 + 3 * sizeof(size_t) + 1);
        if (!generated) { free(handles); return NULL; }
        snprintf(generated, 3 + 3 * sizeof(size_t) + 1, "in-%zu", candidate);
        handles[used++] = (JoinHandle){ generated, candidate, true, true };
        if (candidate == SIZE_MAX) { free(handles); return NULL; }
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
    if (output) return kind_is(package, "output") ? 0 : 1;
    if (kind_is(package, "input")) return 0;
    if (kind_is(package, "join")) {
        size_t count = 0;
        JoinHandle *inputs = join_inputs(model, node_id, &count);
        if (inputs) for (size_t i = 0; i < count; ++i)
            if (inputs[i].owned) free((void *)inputs[i].id);
        free(inputs);
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
        if (kind_is(package, "output") || index != 0) return false;
        return snprintf(buffer, capacity, "out") < (int)capacity;
    }
    if (kind_is(package, "input")) return false;
    if (!kind_is(package, "join")) {
        if (index != 0) return false;
        return snprintf(buffer, capacity, "in") < (int)capacity;
    }
    size_t count = 0;
    JoinHandle *inputs = join_inputs(model, node_id, &count);
    if (!inputs || index >= count) { free(inputs); return false; }
    int written = snprintf(buffer, capacity, "%s", inputs[index].id);
    for (size_t i = 0; i < count; ++i)
        if (inputs[i].owned) free((void *)inputs[i].id);
    free(inputs);
    return written >= 0 && (size_t)written < capacity;
}

bool nn_app_node_is_subflow(const NNApplication *app, const char *node_id)
{
    const NNModel *model = nn_app_model(app);
    const NNNode *node = model ? nn_model_find_node(model, node_id) : NULL;
    return kind_is(find_package(app, node), "subflow");
}
