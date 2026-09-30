#define _GNU_SOURCE
#include "automation.h"
#include "yyjson.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>

#define REQUEST_LIMIT (2u * 1024u * 1024u)
#define CLIENT_LIMIT 8

typedef struct {
    int fd;
    char *input, *output;
    size_t used, sent, length;
    double deadline;
} Client;

struct NNAutomation {
    NNApplication *app;
    NNAutomationUiCallback callback;
    void *user;
    int fd;
    char *path;
    dev_t device;
    ino_t inode;
    bool polling;
    Client clients[CLIENT_LIMIT];
};

typedef struct { yyjson_val *args; char error[512]; } Query;

static void error_text(char *error, size_t cap, const char *text)
{
    if (error && cap) snprintf(error, cap, "%s", text);
}

static const char *string_arg(Query *q, const char *key, const char *fallback)
{
    yyjson_val *value = yyjson_obj_get(q->args, key);
    if (!value && fallback) return fallback;
    if (!yyjson_is_str(value)) {
        snprintf(q->error, sizeof(q->error), "argument '%s' must be a string", key);
        return NULL;
    }
    return yyjson_get_str(value);
}

static bool bool_arg(Query *q, const char *key, bool fallback)
{
    yyjson_val *value = yyjson_obj_get(q->args, key);
    if (!value) return fallback;
    if (!yyjson_is_bool(value)) snprintf(q->error, sizeof(q->error), "argument '%s' must be boolean", key);
    return yyjson_get_bool(value);
}

static double number_arg(Query *q, const char *key, bool required)
{
    yyjson_val *value = yyjson_obj_get(q->args, key);
    if (!value && !required) return 0;
    if (!yyjson_is_num(value)) snprintf(q->error, sizeof(q->error), "argument '%s' must be numeric", key);
    return yyjson_get_num(value);
}

static char *object_arg(Query *q, const char *key, bool required)
{
    yyjson_val *value = yyjson_obj_get(q->args, key);
    if (!value && !required) return strdup("{}");
    if (!yyjson_is_obj(value)) {
        snprintf(q->error, sizeof(q->error), "argument '%s' must be an object", key);
        return NULL;
    }
    return yyjson_val_write(value, 0, NULL);
}

static bool valid_json_tree(yyjson_val *value, unsigned depth)
{
    if (depth > 64) return false;
    if (yyjson_is_str(value)) return strlen(yyjson_get_str(value)) == yyjson_get_len(value);
    if (yyjson_is_arr(value)) {
        size_t index, count; yyjson_val *item;
        yyjson_arr_foreach(value, index, count, item)
            if (!valid_json_tree(item, depth + 1)) return false;
    } else if (yyjson_is_obj(value)) {
        size_t index, count; yyjson_val *key, *item;
        yyjson_obj_foreach(value, index, count, key, item) {
            if (strlen(yyjson_get_str(key)) != yyjson_get_len(key) ||
                yyjson_obj_get(value, yyjson_get_str(key)) != item ||
                !valid_json_tree(item, depth + 1)) return false;
        }
    }
    return true;
}

static yyjson_mut_val *snapshot_value(yyjson_mut_doc *doc, const NNValue *value)
{
    switch (value->type) {
    case NN_VALUE_BOOL: return yyjson_mut_bool(doc, value->as.boolean);
    case NN_VALUE_INT: return yyjson_mut_sint(doc, value->as.integer);
    case NN_VALUE_REAL: return yyjson_mut_real(doc, value->as.real);
    case NN_VALUE_STRING: return yyjson_mut_strcpy(doc, value->as.string);
    case NN_VALUE_ARRAY: {
        yyjson_mut_val *array = yyjson_mut_arr(doc);
        for (size_t i = 0; i < value->as.array.count; ++i)
            yyjson_mut_arr_append(array, snapshot_value(doc, &value->as.array.items[i]));
        return array;
    }
    default: return yyjson_mut_null(doc);
    }
}

static void json_string(yyjson_mut_doc *doc, yyjson_mut_val *obj, const char *key, const char *value)
{
    yyjson_mut_obj_add_strcpy(doc, obj, key, value ? value : "");
}

static yyjson_mut_val *identity(yyjson_mut_doc *doc, const char *id, const char *version)
{
    yyjson_mut_val *value = yyjson_mut_obj(doc);
    json_string(doc, value, "id", id);
    json_string(doc, value, "version", version);
    return value;
}

static yyjson_mut_val *slots_json(yyjson_mut_doc *doc, const NNTensorSlot *slots, size_t count)
{
    yyjson_mut_val *array = yyjson_mut_arr(doc);
    for (size_t i = 0; i < count; ++i) {
        yyjson_mut_val *slot = yyjson_mut_obj(doc);
        json_string(doc, slot, "name", slots[i].name);
        json_string(doc, slot, "dtype", slots[i].dtype);
        yyjson_mut_obj_add_val(doc, slot, "shape", snapshot_value(doc, &slots[i].shape));
        yyjson_mut_arr_append(array, slot);
    }
    return array;
}

static yyjson_mut_val *snapshot(yyjson_mut_doc *doc, NNApplication *app)
{
    const NNProject *project = nn_app_project(app);
    if (!project) return yyjson_mut_null(doc);
    yyjson_mut_val *result = yyjson_mut_obj(doc);
    json_string(doc, result, "id", nn_project_id(project));
    json_string(doc, result, "version", nn_project_version(project));
    json_string(doc, result, "name", nn_project_name(project));
    yyjson_mut_obj_add_bool(doc, result, "dirty", nn_project_dirty(project));
    const NNDataset *active = nn_project_active_dataset(project);
    yyjson_mut_obj_add_val(doc, result, "activeDataset", active
        ? identity(doc, active->id, active->version) : yyjson_mut_null(doc));
    yyjson_mut_val *packages = yyjson_mut_arr(doc), *datasets = yyjson_mut_arr(doc);
    const NNCatalog *catalog = nn_project_catalog(project);
    for (size_t i = 0; i < nn_catalog_count(catalog); ++i) {
        const NNPackage *p = nn_catalog_at(catalog, i);
        yyjson_mut_val *item = identity(doc, p->id, p->version);
        json_string(doc, item, "name", p->name);
        json_string(doc, item, "kind", p->kind);
        yyjson_mut_arr_append(packages, item);
    }
    for (size_t i = 0; i < nn_project_dataset_count(project); ++i) {
        const NNDataset *d = nn_project_dataset_at(project, i);
        yyjson_mut_val *item = identity(doc, d->id, d->version);
        json_string(doc, item, "name", d->name);
        yyjson_mut_obj_add_val(doc, item, "inputs", slots_json(doc, d->inputs, d->input_count));
        yyjson_mut_obj_add_val(doc, item, "targets", slots_json(doc, d->targets, d->target_count));
        yyjson_mut_arr_append(datasets, item);
    }
    yyjson_mut_obj_add_val(doc, result, "packages", packages);
    yyjson_mut_obj_add_val(doc, result, "datasets", datasets);
    yyjson_mut_val *nodes = yyjson_mut_arr(doc), *edges = yyjson_mut_arr(doc);
    const NNModel *model = nn_app_model(app);
    for (size_t i = 0; i < nn_model_node_count(model); ++i) {
        const NNNode *n = nn_model_node_at(model, i);
        yyjson_mut_val *item = yyjson_mut_obj(doc), *params = yyjson_mut_obj(doc);
        json_string(doc, item, "id", n->id);
        json_string(doc, item, "name", n->label);
        json_string(doc, item, "scope", n->scope_id);
        yyjson_mut_obj_add_val(doc, item, "package", identity(doc, n->package_id, n->package_version));
        yyjson_mut_obj_add_double(doc, item, "x", n->x);
        yyjson_mut_obj_add_double(doc, item, "y", n->y);
        for (size_t j = 0; j < n->parameter_count; ++j) {
            yyjson_mut_val *key = yyjson_mut_strcpy(doc, n->parameters[j].key);
            yyjson_mut_obj_add(params, key, snapshot_value(doc, &n->parameters[j].value));
        }
        yyjson_mut_obj_add_val(doc, item, "parameters", params);
        yyjson_mut_arr_append(nodes, item);
    }
    for (size_t i = 0; i < nn_model_edge_count(model); ++i) {
        const NNEdge *e = nn_model_edge_at(model, i);
        yyjson_mut_val *item = yyjson_mut_obj(doc);
        json_string(doc, item, "id", e->id);
        json_string(doc, item, "source", e->source_id);
        json_string(doc, item, "sourceHandle", e->source_handle_id);
        json_string(doc, item, "target", e->target_id);
        json_string(doc, item, "targetHandle", e->target_handle_id);
        json_string(doc, item, "scope", e->scope_id);
        yyjson_mut_arr_append(edges, item);
    }
    yyjson_mut_obj_add_val(doc, result, "nodes", nodes);
    yyjson_mut_obj_add_val(doc, result, "edges", edges);
    return result;
}

static bool execute(NNApplication *app, const char *op, Query *q)
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
            error_text(q->error, sizeof(q->error), "save or explicitly discard the dirty project before replacement");
        return !q->error[0] && nn_app_open(app, path, q->error, sizeof(q->error));
    }
    if (!strcmp(op, "project.create")) {
        const char *parent = string_arg(q, "parent", NULL);
        id = string_arg(q, "id", NULL);
        const char *name = string_arg(q, "name", NULL);
        const char *kind = string_arg(q, "template", "blank");
        if (q->error[0]) return false;
        if (nn_project_dirty(nn_app_project(app))) {
            error_text(q->error, sizeof(q->error), "save or explicitly discard the dirty project before replacement"); return false;
        }
        if (!strcmp(kind, "mnist-vae")) return nn_app_create_vae(app, parent, id, name, q->error, sizeof(q->error));
        if (strcmp(kind, "blank") && strcmp(kind, "mnist-mlp")) {
            error_text(q->error, sizeof(q->error), "unknown project template"); return false;
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
    if (!strcmp(op, "edge.disconnect")) return !q->error[0] && nn_app_disconnect(app, id, q->error, sizeof(q->error));
    if (!strcmp(op, "edge.connect")) {
        const char *source = string_arg(q, "source", NULL), *target = string_arg(q, "target", NULL);
        const char *sh = string_arg(q, "sourceHandle", NULL), *th = string_arg(q, "targetHandle", NULL);
        return !q->error[0] && nn_app_connect(app, id, source, sh, target, th, q->error, sizeof(q->error));
    }
    error_text(q->error, sizeof(q->error), "unknown operation");
    return false;
}

char *nn_automation_dispatch(NNApplication *app, const char *request,
                             NNAutomationUiCallback callback, void *user)
{
    yyjson_doc *input = request && strlen(request) <= REQUEST_LIMIT
        ? yyjson_read(request, strlen(request), 0) : NULL;
    yyjson_val *root = input ? yyjson_doc_get_root(input) : NULL;
    const char *op = yyjson_get_str(yyjson_obj_get(root, "operation"));
    Query query = { .args = yyjson_obj_get(root, "args") };
    yyjson_mut_doc *out = yyjson_mut_doc_new(NULL);
    if (!out) { yyjson_doc_free(input); return NULL; }
    yyjson_mut_val *response = yyjson_mut_obj(out), *result = yyjson_mut_null(out);
    yyjson_doc *ui_result = NULL;
    bool okay = false;
    if (!app || !yyjson_is_obj(root) || !op || !yyjson_is_obj(query.args) || !valid_json_tree(root, 0))
        error_text(query.error, sizeof(query.error), "request requires operation string and args object");
    else if (!strcmp(op, "project.snapshot")) { result = snapshot(out, app); okay = true; }
    else if (!strncmp(op, "ui.", 3)) {
        if (!strcmp(op, "ui.inspect") || !strcmp(op, "ui.arrange") || !strcmp(op, "ui.scope") || !strcmp(op, "ui.screenshot")) {
            if (!callback) error_text(query.error, sizeof(query.error), "UI callback unavailable");
            else {
                char *args = yyjson_val_write(query.args, 0, NULL);
                char *json = args ? callback(user, op, args, query.error, sizeof(query.error)) : NULL;
                free(args);
                if (json) ui_result = yyjson_read(json, strlen(json), 0);
                free(json);
                if (ui_result) { result = yyjson_val_mut_copy(out, yyjson_doc_get_root(ui_result)); okay = result != NULL; }
            }
        } else error_text(query.error, sizeof(query.error), "unknown UI operation");
    } else okay = execute(app, op, &query);
    if (!okay && !query.error[0]) error_text(query.error, sizeof(query.error), "operation failed");
    yyjson_mut_obj_add_bool(out, response, "ok", okay);
    if (okay) yyjson_mut_obj_add_val(out, response, "result", result);
    else json_string(out, response, "error", query.error);
    yyjson_mut_doc_set_root(out, response);
    char *json = yyjson_mut_write(out, YYJSON_WRITE_NEWLINE_AT_END, NULL);
    yyjson_doc_free(ui_result); yyjson_doc_free(input); yyjson_mut_doc_free(out);
    return json;
}

static double now(void)
{
    struct timespec time;
    clock_gettime(CLOCK_MONOTONIC, &time);
    return (double)time.tv_sec + (double)time.tv_nsec / 1000000000;
}

static void close_client(Client *client)
{
    if (client->fd >= 0) close(client->fd);
    free(client->input); free(client->output);
    memset(client, 0, sizeof(*client)); client->fd = -1;
}

NNAutomation *nn_automation_start(NNApplication *app, const char *path,
                                NNAutomationUiCallback callback, void *user,
                                char *error, size_t cap)
{
    struct sockaddr_un address = { .sun_family = AF_UNIX };
    struct stat info;
    if (!app || !path || !*path || strlen(path) >= sizeof(address.sun_path)) {
        error_text(error, cap, "invalid or overlong local socket path"); return NULL;
    }
    if (!lstat(path, &info) || errno != ENOENT) {
        error_text(error, cap, "socket path already exists or cannot be inspected"); return NULL;
    }
    NNAutomation *service = calloc(1, sizeof(*service));
    if (!service) return NULL;
    service->fd = -1;
    for (size_t i = 0; i < CLIENT_LIMIT; ++i) service->clients[i].fd = -1;
    service->app = app; service->callback = callback; service->user = user;
    service->path = strdup(path);
    strcpy(address.sun_path, path);
    service->fd = socket(AF_UNIX, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
    if (!service->path || service->fd < 0) goto fail;
    mode_t mask = umask(0077);
    int bound = bind(service->fd, (struct sockaddr *)&address, sizeof(address));
    umask(mask);
    if (bound) goto fail;
    if (lstat(path, &info)) goto fail;
    service->device = info.st_dev; service->inode = info.st_ino;
    if (!S_ISSOCK(info.st_mode) || info.st_uid != getuid() || chmod(path, 0600) || listen(service->fd, CLIENT_LIMIT)) goto fail;
    error_text(error, cap, "");
    return service;
fail:
    error_text(error, cap, "unable to bind private local socket");
    nn_automation_stop(service);
    return NULL;
}

bool nn_automation_poll(NNAutomation *service)
{
    if (!service || service->polling) return false;
    service->polling = true;
    bool changed = false;
    for (size_t accepted = 0; accepted < CLIENT_LIMIT; ++accepted) {
        int fd = accept4(service->fd, NULL, NULL, SOCK_NONBLOCK | SOCK_CLOEXEC);
        if (fd < 0) break;
        struct ucred peer; socklen_t length = sizeof(peer);
        if (getsockopt(fd, SOL_SOCKET, SO_PEERCRED, &peer, &length) || peer.uid != getuid()) { close(fd); continue; }
        size_t slot = 0;
        while (slot < CLIENT_LIMIT && service->clients[slot].fd >= 0) ++slot;
        if (slot == CLIENT_LIMIT) { close(fd); continue; }
        Client *client = &service->clients[slot];
        client->fd = fd; client->deadline = now() + 5;
        client->input = malloc(REQUEST_LIMIT + 1);
        if (!client->input) close_client(client);
    }
    for (size_t i = 0; i < CLIENT_LIMIT; ++i) {
        Client *client = &service->clients[i];
        if (client->fd < 0) continue;
        if (now() > client->deadline) { close_client(client); continue; }
        if (!client->output) {
            size_t capacity = REQUEST_LIMIT - client->used;
            if (capacity > 65536) capacity = 65536;
            ssize_t count = recv(client->fd, client->input + client->used, capacity, 0);
            if (count > 0) {
                client->used += (size_t)count; client->input[client->used] = '\0';
                char *newline = memchr(client->input, '\n', client->used);
                if (newline) {
                    *newline = '\0';
                    if (memchr(client->input, '\0', (size_t)(newline - client->input)))
                        client->output = strdup("{\"ok\":false,\"error\":\"embedded NUL in request\"}\n");
                    else client->output = nn_automation_dispatch(service->app, client->input, service->callback, service->user);
                    if (client->output) {
                        yyjson_doc *response = yyjson_read(client->output, strlen(client->output), 0);
                        yyjson_doc *request = yyjson_read(client->input, strlen(client->input), 0);
                        const char *op = request ? yyjson_get_str(yyjson_obj_get(yyjson_doc_get_root(request), "operation")) : NULL;
                        bool readonly = op && (!strcmp(op, "project.snapshot") || !strcmp(op, "ui.inspect") || !strcmp(op, "ui.screenshot"));
                        changed |= !readonly && response && yyjson_get_bool(yyjson_obj_get(yyjson_doc_get_root(response), "ok"));
                        yyjson_doc_free(request);
                        yyjson_doc_free(response);
                    }
                } else if (client->used == REQUEST_LIMIT)
                    client->output = strdup("{\"ok\":false,\"error\":\"request too large\"}\n");
                if (client->output) client->length = strlen(client->output);
            } else if (!count || (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR)) { close_client(client); continue; }
        }
        if (client->output) {
            size_t pending = client->length - client->sent;
            if (pending > 65536) pending = 65536;
            ssize_t count = send(client->fd, client->output + client->sent, pending, MSG_NOSIGNAL);
            if (count > 0) client->sent += (size_t)count;
            else if (count < 0 && errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) { close_client(client); continue; }
            if (client->sent == client->length) close_client(client);
        }
    }
    service->polling = false;
    return changed;
}

void nn_automation_stop(NNAutomation *service)
{
    if (!service) return;
    for (size_t i = 0; i < CLIENT_LIMIT; ++i) close_client(&service->clients[i]);
    if (service->fd >= 0) close(service->fd);
    struct stat info;
    if (service->path && service->inode && !lstat(service->path, &info) &&
        info.st_dev == service->device && info.st_ino == service->inode && S_ISSOCK(info.st_mode)) unlink(service->path);
    free(service->path); free(service);
}
