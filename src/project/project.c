#include "project_internal.h"
#include "utils/utils.h"

#include <stdlib.h>
#include <string.h>

#define NN_OPERATION_LIMIT 256
static bool json_text(yyjson_val *object, const char *key, const char **text);

static const char *const python_keywords[] = {
    "False", "None", "True", "and", "as", "assert", "async", "await",
    "break", "class", "continue", "def", "del", "elif", "else", "except",
    "finally", "for", "from", "global", "if", "import", "in", "is",
    "lambda", "nonlocal", "not", "or", "pass", "raise", "return", "try",
    "while", "with", "yield"
};

void nn_project_operations_dispose(NNModelOperation *operations, size_t count)
{
    if (!operations) return;
    for (size_t i = 0; i < count; ++i) {
        free(operations[i].name);
        free(operations[i].input.node);
        free(operations[i].input.handle);
        free(operations[i].input.codec);
        free(operations[i].output.node);
        free(operations[i].output.handle);
        free(operations[i].output.codec);
    }
    free(operations);
}

void nn_project_metavariables_dispose(NNMetavariable *items, size_t count)
{
    if (!items) return;
    for (size_t i = 0; i < count; ++i) {
        free(items[i].name);
        free(items[i].type);
        nn_value_dispose(&items[i].value);
    }
    free(items);
}

static bool valid_metavariable_name(const char *name)
{
    if (!name || !*name || strlen(name) > 128 ||
        !(name[0] == '_' || (name[0] >= 'A' && name[0] <= 'Z') ||
          (name[0] >= 'a' && name[0] <= 'z'))) return false;
    for (const unsigned char *p = (const unsigned char *)name + 1; *p; ++p)
        if (!( (*p >= 'A' && *p <= 'Z') || (*p >= 'a' && *p <= 'z') ||
               (*p >= '0' && *p <= '9') || *p == '_' || *p == '-')) return false;
    return true;
}

static bool contains_binding(const NNValue *value)
{
    if (!value) return false;
    if (value->type == NN_VALUE_OBJECT) {
        if (value->as.object.count == 1 && value->as.object.items) {
            const NNParameter *item = &value->as.object.items[0];
            if (item->key && (!strcmp(item->key, "$ref") || !strcmp(item->key, "$lua")))
                return true;
        }
        for (size_t i = 0; i < value->as.object.count; ++i)
            if (contains_binding(&value->as.object.items[i].value)) return true;
    } else if (value->type == NN_VALUE_ARRAY) {
        for (size_t i = 0; i < value->as.array.count; ++i)
            if (contains_binding(&value->as.array.items[i])) return true;
    }
    return false;
}

bool nn_project_parse_metavariables(NNProject *project, yyjson_val *value,
                                    char *error, size_t capacity)
{
    if (!project) return nn_fail(error, capacity, "project is null");
    if (!value) return true;
    if (!yyjson_is_arr(value) || yyjson_arr_size(value) > 256)
        return nn_fail(error, capacity, "invalid metavariable list");
    size_t count = yyjson_arr_size(value);
    NNMetavariable *items = calloc(count ? count : 1, sizeof(*items));
    if (!items) return nn_fail(error, capacity, "out of memory loading metavariables");
    bool okay = true;
    for (size_t i = 0; okay && i < count; ++i) {
        yyjson_val *entry = yyjson_arr_get(value, i);
        const char *name = NULL, *type = NULL;
        yyjson_val *raw = yyjson_obj_get(entry, "value");
        if (!yyjson_is_obj(entry) || yyjson_obj_size(entry) != 3 || !json_text(entry, "name", &name) ||
            !json_text(entry, "type", &type) || !raw || !valid_metavariable_name(name) ||
            (strcmp(type, "boolean") && strcmp(type, "integer") &&
             strcmp(type, "number") && strcmp(type, "string") &&
             strcmp(type, "dtype") && strcmp(type, "json") &&
             strcmp(type, "stereotype"))) {
            nn_errorf(error, capacity, "invalid metavariable at index %zu", i);
            okay = false;
            break;
        }
        for (size_t j = 0; j < i; ++j)
            if (!strcmp(name, items[j].name)) {
                nn_errorf(error, capacity, "duplicate metavariable name: %s", name);
                okay = false;
                break;
            }
        if (!okay) break;
        items[i].name = nn_text_copy(name);
        items[i].type = nn_text_copy(type);
        okay = items[i].name && items[i].type && nn_project_parse_value(raw, &items[i].value);
        if (okay && !strcmp(type, "number") && items[i].value.type == NN_VALUE_INT) {
            long long integer = items[i].value.as.integer;
            if (integer > 9007199254740992LL || integer < -9007199254740992LL) {
                nn_errorf(error, capacity, "number metavariable '%s' is not exactly representable", name);
                okay = false;
            } else {
                items[i].value.type = NN_VALUE_REAL;
                items[i].value.as.real = (double)integer;
            }
        }
        if (okay && contains_binding(&items[i].value)) {
            nn_errorf(error, capacity, "metavariable '%s' must contain literal values", name);
            okay = false;
        }
        NNParameterDef definition = { .key = name, .type = type };
        if (okay) okay = nn_catalog_validate_value(project->catalog, &definition,
                                                     &items[i].value, error, capacity);
        if (!okay && (!error || !capacity || !error[0]))
            nn_errorf(error, capacity, "invalid value for metavariable '%s'", name);
    }
    if (!okay) {
        nn_project_metavariables_dispose(items, count);
        return false;
    }
    project->metavariables = items;
    project->metavariable_count = count;
    return true;
}

static bool values_equal(const NNValue *a, const NNValue *b)
{
    if (!a || !b || a->type != b->type) return false;
    switch (a->type) {
    case NN_VALUE_BOOL: return a->as.boolean == b->as.boolean;
    case NN_VALUE_INT: return a->as.integer == b->as.integer;
    case NN_VALUE_REAL: return a->as.real == b->as.real;
    case NN_VALUE_STRING: return a->as.string && b->as.string && !strcmp(a->as.string, b->as.string);
    case NN_VALUE_ARRAY:
        if (a->as.array.count != b->as.array.count) return false;
        for (size_t i = 0; i < a->as.array.count; ++i)
            if (!values_equal(&a->as.array.items[i], &b->as.array.items[i])) return false;
        return true;
    case NN_VALUE_OBJECT:
        if (a->as.object.count != b->as.object.count) return false;
        for (size_t i = 0; i < a->as.object.count; ++i) {
            const NNValue *right = nn_value_member(b, a->as.object.items[i].key);
            if (!right || !values_equal(&a->as.object.items[i].value, right)) return false;
        }
        return true;
    default: return false;
    }
}

bool nn_project_set_metavariables_json(NNProject *project, const char *json, bool *changed,
                                       char *error, size_t capacity)
{
    if (changed) *changed = false;
    if (error && capacity) error[0] = '\0';
    if (!project || !json) return nn_fail(error, capacity, "invalid metavariable JSON");
    yyjson_doc *document = yyjson_read(json, strlen(json), 0);
    if (!document) return nn_fail(error, capacity, "invalid metavariable JSON");
    NNProject parsed = { .catalog = project->catalog };
    bool okay = nn_project_parse_metavariables(&parsed, yyjson_doc_get_root(document), error, capacity);
    yyjson_doc_free(document);
    if (!okay) { nn_project_metavariables_dispose(parsed.metavariables, parsed.metavariable_count); return false; }
    bool same = project->metavariable_count == parsed.metavariable_count;
    for (size_t i = 0; same && i < parsed.metavariable_count; ++i) {
        const NNMetavariable *a = &project->metavariables[i], *b = &parsed.metavariables[i];
        same = !strcmp(a->name, b->name) && !strcmp(a->type, b->type) && values_equal(&a->value, &b->value);
    }
    if (!same) {
        nn_project_metavariables_dispose(project->metavariables, project->metavariable_count);
        project->metavariables = parsed.metavariables;
        project->metavariable_count = parsed.metavariable_count;
        parsed.metavariables = NULL;
        parsed.metavariable_count = 0;
        project->dirty = true;
        if (changed) *changed = true;
    }
    nn_project_metavariables_dispose(parsed.metavariables, parsed.metavariable_count);
    return true;
}

const NNMetavariable *nn_project_metavariables(const NNProject *project, size_t *count)
{
    if (count) *count = project ? project->metavariable_count : 0;
    return project ? project->metavariables : NULL;
}

static yyjson_mut_val *write_value(yyjson_mut_doc *doc, const NNValue *value)
{
    switch (value->type) {
    case NN_VALUE_BOOL: return yyjson_mut_bool(doc, value->as.boolean);
    case NN_VALUE_INT: return yyjson_mut_sint(doc, value->as.integer);
    case NN_VALUE_REAL: return yyjson_mut_real(doc, value->as.real);
    case NN_VALUE_STRING: return value->as.string ? yyjson_mut_strcpy(doc, value->as.string) : NULL;
    case NN_VALUE_ARRAY: {
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
            yyjson_mut_val *item = write_value(doc, &value->as.object.items[i].value);
            if (!item || !yyjson_mut_obj_add_val(doc, object, value->as.object.items[i].key, item)) return NULL;
        }
        return object;
    }
    default: return NULL;
    }
}

yyjson_mut_val *nn_project_write_metavariables(yyjson_mut_doc *doc, const NNProject *project)
{
    yyjson_mut_val *array = yyjson_mut_arr(doc);
    if (!array) return NULL;
    for (size_t i = 0; i < project->metavariable_count; ++i) {
        const NNMetavariable *variable = &project->metavariables[i];
        yyjson_mut_val *entry = yyjson_mut_obj(doc);
        yyjson_mut_val *value = write_value(doc, &variable->value);
        if (!entry || !value || !yyjson_mut_obj_add_str(doc, entry, "name", variable->name) ||
            !yyjson_mut_obj_add_str(doc, entry, "type", variable->type) ||
            !yyjson_mut_obj_add_val(doc, entry, "value", value) || !yyjson_mut_arr_append(array, entry)) return NULL;
    }
    return array;
}

char *nn_project_metavariables_json(const NNProject *project, char *error, size_t capacity)
{
    if (error && capacity) error[0] = '\0';
    if (!project) { nn_errorf(error, capacity, "no active project"); return NULL; }
    yyjson_mut_doc *doc = yyjson_mut_doc_new(NULL);
    if (!doc) { nn_errorf(error, capacity, "out of memory serializing metavariables"); return NULL; }
    yyjson_mut_val *array = nn_project_write_metavariables(doc, project);
    if (!array) { yyjson_mut_doc_free(doc); nn_errorf(error, capacity, "unable to serialize metavariables"); return NULL; }
    yyjson_mut_doc_set_root(doc, array);
    char *text = yyjson_mut_write_opts(doc, YYJSON_WRITE_PRETTY_TWO_SPACES, NULL, NULL, NULL);
    yyjson_mut_doc_free(doc);
    if (!text) nn_errorf(error, capacity, "unable to serialize metavariables");
    return text;
}

static bool json_text(yyjson_val *object, const char *key, const char **text)
{
    yyjson_val *value = yyjson_obj_get(object, key);
    if (!yyjson_is_str(value)) return false;
    const char *raw = yyjson_get_str(value);
    size_t length = yyjson_get_len(value);
    if (!raw || !length || strlen(raw) != length) return false;
    *text = raw;
    return true;
}

static bool valid_operation_name(const char *name)
{
    if (!name || !(*name == '_' || (*name >= 'A' && *name <= 'Z') ||
                   (*name >= 'a' && *name <= 'z')) || name[0] == '_') return false;
    for (const unsigned char *p = (const unsigned char *)name + 1; *p; ++p)
        if (!((*p >= 'A' && *p <= 'Z') || (*p >= 'a' && *p <= 'z') ||
              (*p >= '0' && *p <= '9') || *p == '_')) return false;
    if (!strcmp(name, "infer") || !strcmp(name, "inference") ||
        !strcmp(name, "run_operation")) return false;
    for (size_t i = 0; i < sizeof(python_keywords) / sizeof(python_keywords[0]); ++i)
        if (!strcmp(name, python_keywords[i])) return false;
    return true;
}

static bool parse_endpoint(yyjson_val *value, NNOperationEndpoint *endpoint)
{
    const char *node = NULL, *handle = NULL, *codec = NULL;
    if (!yyjson_is_obj(value) || !json_text(value, "node", &node) ||
        !json_text(value, "handle", &handle) || !json_text(value, "codec", &codec) ||
        (strcmp(codec, "dataset") && strcmp(codec, "tensor"))) return false;
    endpoint->node = nn_text_copy(node);
    endpoint->handle = nn_text_copy(handle);
    endpoint->codec = nn_text_copy(codec);
    return endpoint->node && endpoint->handle && endpoint->codec;
}

bool nn_project_parse_operations(NNProject *project, yyjson_val *value,
                                 char *error, size_t capacity)
{
    if (!project) return nn_fail(error, capacity, "project is null");
    if (!value) return true;
    if (!yyjson_is_arr(value) || yyjson_arr_size(value) > NN_OPERATION_LIMIT)
        return nn_fail(error, capacity, "invalid operation list");
    size_t count = yyjson_arr_size(value);
    NNModelOperation *items = calloc(count ? count : 1, sizeof(*items));
    if (!items) return nn_fail(error, capacity, "out of memory loading operations");
    bool okay = true;
    for (size_t i = 0; okay && i < count; ++i) {
        yyjson_val *entry = yyjson_arr_get(value, i);
        const char *name = NULL;
        if (!yyjson_is_obj(entry) || !json_text(entry, "name", &name) ||
            !valid_operation_name(name)) {
            nn_errorf(error, capacity, "invalid operation name at index %zu", i);
            okay = false;
        }
        for (size_t j = 0; okay && j < i; ++j)
            if (!strcmp(name, items[j].name)) {
                nn_errorf(error, capacity, "duplicate operation name: %s", name);
                okay = false;
            }
        if (okay) {
            items[i].name = nn_text_copy(name);
            okay = items[i].name &&
                parse_endpoint(yyjson_obj_get(entry, "input"), &items[i].input) &&
                parse_endpoint(yyjson_obj_get(entry, "output"), &items[i].output);
            if (!okay && (!error || !capacity || !error[0]))
                nn_errorf(error, capacity, "invalid operation endpoint at index %zu", i);
            else if (!okay && error && capacity && !error[0])
                nn_errorf(error, capacity, "out of memory loading operation at index %zu", i);
        }
    }
    if (!okay) {
        nn_project_operations_dispose(items, count);
        return false;
    }
    project->operations = items;
    project->operation_count = count;
    return true;
}

static bool same_operations(const NNProject *project, const NNModelOperation *items,
                            size_t count)
{
    if (project->operation_count != count) return false;
    for (size_t i = 0; i < count; ++i) {
        const NNModelOperation *a = &project->operations[i], *b = &items[i];
        if (strcmp(a->name, b->name) || strcmp(a->input.node, b->input.node) ||
            strcmp(a->input.handle, b->input.handle) || strcmp(a->input.codec, b->input.codec) ||
            strcmp(a->output.node, b->output.node) || strcmp(a->output.handle, b->output.handle) ||
            strcmp(a->output.codec, b->output.codec)) return false;
    }
    return true;
}

bool nn_project_set_operations_json(NNProject *project, const char *json, bool *changed,
                                    char *error, size_t capacity)
{
    if (changed) *changed = false;
    if (error && capacity) error[0] = '\0';
    if (!project || !json) return nn_fail(error, capacity, "invalid operation JSON");
    yyjson_doc *document = yyjson_read(json, strlen(json), 0);
    if (!document) return nn_fail(error, capacity, "invalid operation JSON");
    NNProject parsed = {0};
    bool okay = nn_project_parse_operations(&parsed, yyjson_doc_get_root(document), error, capacity);
    yyjson_doc_free(document);
    if (!okay) { nn_project_operations_dispose(parsed.operations, parsed.operation_count); return false; }
    if (!same_operations(project, parsed.operations, parsed.operation_count)) {
        nn_project_operations_dispose(project->operations, project->operation_count);
        project->operations = parsed.operations;
        project->operation_count = parsed.operation_count;
        parsed.operations = NULL;
        parsed.operation_count = 0;
        project->dirty = true;
        if (changed) *changed = true;
    }
    nn_project_operations_dispose(parsed.operations, parsed.operation_count);
    return true;
}

yyjson_mut_val *nn_project_write_operations(yyjson_mut_doc *doc,
                                           const NNProject *project)
{
    yyjson_mut_val *array = yyjson_mut_arr(doc);
    if (!array) return NULL;
    for (size_t i = 0; i < project->operation_count; ++i) {
        const NNModelOperation *operation = &project->operations[i];
        yyjson_mut_val *entry = yyjson_mut_obj(doc);
        yyjson_mut_val *input = yyjson_mut_obj(doc);
        yyjson_mut_val *output = yyjson_mut_obj(doc);
        if (!entry || !input || !output ||
            !yyjson_mut_obj_add_str(doc, entry, "name", operation->name) ||
            !yyjson_mut_obj_add_str(doc, input, "node", operation->input.node) ||
            !yyjson_mut_obj_add_str(doc, input, "handle", operation->input.handle) ||
            !yyjson_mut_obj_add_str(doc, input, "codec", operation->input.codec) ||
            !yyjson_mut_obj_add_str(doc, output, "node", operation->output.node) ||
            !yyjson_mut_obj_add_str(doc, output, "handle", operation->output.handle) ||
            !yyjson_mut_obj_add_str(doc, output, "codec", operation->output.codec) ||
            !yyjson_mut_obj_add_val(doc, entry, "input", input) ||
            !yyjson_mut_obj_add_val(doc, entry, "output", output) ||
            !yyjson_mut_arr_append(array, entry)) return NULL;
    }
    return array;
}

char *nn_project_operations_json(const NNProject *project, char *error, size_t capacity)
{
    if (error && capacity) error[0] = '\0';
    if (!project) { nn_errorf(error, capacity, "no active project"); return NULL; }
    yyjson_mut_doc *doc = yyjson_mut_doc_new(NULL);
    if (!doc) { nn_errorf(error, capacity, "out of memory serializing operations"); return NULL; }
    yyjson_mut_val *array = nn_project_write_operations(doc, project);
    if (!array) { yyjson_mut_doc_free(doc); nn_errorf(error, capacity, "unable to serialize operations"); return NULL; }
    yyjson_mut_doc_set_root(doc, array);
    size_t length = 0;
    char *text = yyjson_mut_write_opts(doc, YYJSON_WRITE_PRETTY_TWO_SPACES, NULL, &length, NULL);
    yyjson_mut_doc_free(doc);
    if (!text) nn_errorf(error, capacity, "unable to serialize operations");
    return text;
}

void nn_project_close(NNProject *project)
{
    if (!project) return;
    nn_model_free(project->model);
    nn_catalog_free(project->catalog);
    for (size_t i = 0; project->packages && i < project->package_count; ++i) {
        free((char *)project->packages[i].id);
        free((char *)project->packages[i].version);
        free((char *)project->packages[i].path);
    }
    for (size_t i = 0; project->datasets && i < project->dataset_count; ++i) nn_project_dataset_dispose(&project->datasets[i]);
    nn_project_operations_dispose(project->operations, project->operation_count);
    nn_project_metavariables_dispose(project->metavariables, project->metavariable_count);
    free(project->packages); free(project->datasets);
    free(project->directory); free(project->core_root); free(project->id); free(project->version);
    free(project->name); free(project->description); free(project->layout_direction);
    free(project->active_dataset_id); free(project->active_dataset_version);
    free(project);
}

void nn_project_mark_dirty(NNProject *project) { if (project) project->dirty = true; }

void nn_project_set_dirty(NNProject *project, bool dirty) { if (project) project->dirty = dirty; }

const char *nn_project_directory(const NNProject *project) { return project ? project->directory : NULL; }

const char *nn_project_id(const NNProject *project) { return project ? project->id : NULL; }

const char *nn_project_version(const NNProject *project) { return project ? project->version : NULL; }

const char *nn_project_name(const NNProject *project) { return project ? project->name : NULL; }

bool nn_project_dirty(const NNProject *project) { return project && project->dirty; }

NNModel *nn_project_model(NNProject *project) { return project ? project->model : NULL; }

const NNCatalog *nn_project_catalog(const NNProject *project) { return project ? project->catalog : NULL; }

size_t nn_project_dataset_count(const NNProject *project) { return project ? project->dataset_count : 0; }

const NNDataset *nn_project_dataset_at(const NNProject *project, size_t index)
{
    return project && index < project->dataset_count ? &project->datasets[index] : NULL;
}

const NNDataset *nn_project_active_dataset(const NNProject *project)
{
    if (!project || !project->active_dataset_id) return NULL;
    for (size_t i = 0; i < project->dataset_count; ++i)
        if (!strcmp(project->datasets[i].id, project->active_dataset_id) &&
            !strcmp(project->datasets[i].version, project->active_dataset_version))
            return &project->datasets[i];
    return NULL;
}
