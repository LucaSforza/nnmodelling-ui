#include "automation_internal.h"
#include "utils/utils.h"

#include <stdlib.h>
#include <string.h>

static const char *string_arg(NNAutomationQuery *q, const char *key, const char *fallback)
{
    yyjson_val *value = yyjson_obj_get(q->args, key);
    if (!value && fallback) return fallback;
    if (!yyjson_is_str(value)) {
        nn_errorf(q->error, sizeof(q->error), "argument '%s' must be a string", key);
        return NULL;
    }
    return yyjson_get_str(value);
}

static bool bool_arg(NNAutomationQuery *q, const char *key, bool fallback)
{
    yyjson_val *value = yyjson_obj_get(q->args, key);
    if (!value) return fallback;
    if (!yyjson_is_bool(value)) nn_errorf(q->error, sizeof(q->error), "argument '%s' must be boolean", key);
    return yyjson_get_bool(value);
}

static double number_arg(NNAutomationQuery *q, const char *key, bool required)
{
    yyjson_val *value = yyjson_obj_get(q->args, key);
    if (!value && !required) return 0;
    if (!yyjson_is_num(value)) nn_errorf(q->error, sizeof(q->error), "argument '%s' must be numeric", key);
    return yyjson_get_num(value);
}

static char *object_arg(NNAutomationQuery *q, const char *key, bool required)
{
    yyjson_val *value = yyjson_obj_get(q->args, key);
    if (!value && !required) {
        char *text = nn_text_copy("{}");
        if (!text) nn_error_set(q->error, sizeof(q->error), "out of memory copying arguments");
        return text;
    }
    if (!yyjson_is_obj(value)) {
        nn_errorf(q->error, sizeof(q->error), "argument '%s' must be an object", key);
        return NULL;
    }
    char *text = yyjson_val_write(value, 0, NULL);
    if (!text) nn_error_set(q->error, sizeof(q->error), "out of memory serializing arguments");
    return text;
}

bool nn_automation_execute(NNApplication *app, const char *op, NNAutomationQuery *q)
{
    const char *id = NULL;
    if (!strcmp(op, "project.save")) return nn_app_save(app, q->error, sizeof(q->error));
    if (!strcmp(op, "project.close")) {
        bool discard = bool_arg(q, "discard", false);
        return !q->error[0] && nn_app_close(app, discard, q->error, sizeof(q->error));
    }
    if (!strcmp(op, "project.open")) {
        const char *path = string_arg(q, "path", NULL);
        if (!q->error[0] && nn_project_dirty(nn_app_project(app)))
            nn_error_set(q->error, sizeof(q->error), "save or explicitly discard the dirty project before replacement");
        return !q->error[0] && nn_app_open(app, path, q->error, sizeof(q->error));
    }
    if (!strcmp(op, "project.create")) {
        const char *parent = string_arg(q, "parent", NULL);
        id = string_arg(q, "id", NULL);
        const char *name = string_arg(q, "name", NULL);
        const char *kind = string_arg(q, "template", "blank");
        if (q->error[0]) return false;
        if (nn_project_dirty(nn_app_project(app))) {
            nn_error_set(q->error, sizeof(q->error), "save or explicitly discard the dirty project before replacement"); return false;
        }
        if (!strcmp(kind, "mnist-vae")) return nn_app_create_vae(app, parent, id, name, q->error, sizeof(q->error));
        if (strcmp(kind, "blank") && strcmp(kind, "mnist-mlp")) {
            nn_error_set(q->error, sizeof(q->error), "unknown project template"); return false;
        }
        return nn_app_create(app, parent, id, name, !strcmp(kind, "mnist-mlp"), q->error, sizeof(q->error));
    }
    if (!strcmp(op, "stereotype.create") || !strcmp(op, "dataset.create")) {
        id = string_arg(q, "id", NULL);
        const char *version = string_arg(q, "version", NULL);
        char *definition = object_arg(q, "definition", true);
        bool okay = false;
        if (!strcmp(op, "stereotype.create")) {
            const char *source = string_arg(q, "inference", NULL);
            char *dependencies = object_arg(q, "dependencies", false);
            if (!q->error[0] && definition && dependencies)
                okay = nn_app_create_stereotype(app, id, version, definition, source, dependencies, q->error, sizeof(q->error));
            free(dependencies);
        } else {
            bool select = bool_arg(q, "select", false);
            if (!q->error[0] && definition)
                okay = nn_app_create_dataset(app, id, version, definition, select, q->error, sizeof(q->error));
        }
        free(definition);
        return okay;
    }
    if (!strcmp(op, "dataset.select")) {
        id = string_arg(q, "id", NULL);
        const char *version = string_arg(q, "version", NULL);
        return !q->error[0] && nn_app_select_dataset(app, id, version, q->error, sizeof(q->error));
    }
    if (!strncmp(op, "node.", 5) || !strncmp(op, "edge.", 5)) id = string_arg(q, "id", NULL);
    if (!strcmp(op, "node.add")) {
        const char *package = string_arg(q, "package", NULL), *version = string_arg(q, "version", NULL);
        const char *scope = string_arg(q, "scope", "");
        double x = number_arg(q, "x", false), y = number_arg(q, "y", false);
        return !q->error[0] && nn_app_add_node(app, id, package, version, scope, x, y, q->error, sizeof(q->error));
    }
    if (!strcmp(op, "node.remove")) return !q->error[0] && nn_app_remove_node(app, id, q->error, sizeof(q->error));
    if (!strcmp(op, "node.move")) {
        double x = number_arg(q, "x", true), y = number_arg(q, "y", true);
        return !q->error[0] && nn_app_move_node(app, id, x, y, q->error, sizeof(q->error));
    }
    if (!strcmp(op, "node.rename")) {
        const char *name = string_arg(q, "name", NULL);
        return !q->error[0] && nn_app_rename_node(app, id, name, q->error, sizeof(q->error));
    }
    if (!strcmp(op, "node.parameter")) {
        const char *key = string_arg(q, "key", NULL), *value = string_arg(q, "value", NULL);
        return !q->error[0] && nn_app_set_parameter_text(app, id, key, value, q->error, sizeof(q->error));
    }
    if (!strcmp(op, "node.boundary")) {
        const char *handle = string_arg(q, "handle", NULL);
        return !q->error[0] && nn_app_set_boundary_handle(app, id, handle, q->error, sizeof(q->error));
    }
    if (!strcmp(op, "edge.disconnect")) return !q->error[0] && nn_app_disconnect(app, id, q->error, sizeof(q->error));
    if (!strcmp(op, "edge.connect")) {
        const char *source = string_arg(q, "source", NULL), *target = string_arg(q, "target", NULL);
        const char *sh = string_arg(q, "sourceHandle", NULL), *th = string_arg(q, "targetHandle", NULL);
        return !q->error[0] && nn_app_connect(app, id, source, sh, target, th, q->error, sizeof(q->error));
    }
    nn_error_set(q->error, sizeof(q->error), "unknown operation");
    return false;
}
