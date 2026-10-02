#define _GNU_SOURCE
#include "automation.h"
#include "automation_internal.h"
#include "utils/utils.h"

#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>

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
        nn_error_set(error, cap, "invalid or overlong local socket path"); return NULL;
    }
    if (!lstat(path, &info) || errno != ENOENT) {
        nn_error_set(error, cap, "socket path already exists or cannot be inspected"); return NULL;
    }
    NNAutomation *service = calloc(1, sizeof(*service));
    if (!service) { nn_error_set(error, cap, "out of memory starting local service"); return NULL; }
    service->fd = -1;
    for (size_t i = 0; i < CLIENT_LIMIT; ++i) service->clients[i].fd = -1;
    service->app = app; service->callback = callback; service->user = user;
    service->path = nn_text_copy(path);
    if (!service->path) {
        nn_error_set(error, cap, "out of memory copying local socket path");
        nn_automation_stop(service);
        return NULL;
    }
    strcpy(address.sun_path, path);
    service->fd = socket(AF_UNIX, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
    if (service->fd < 0) goto fail;
    mode_t mask = umask(0077);
    int bound = bind(service->fd, (struct sockaddr *)&address, sizeof(address));
    umask(mask);
    if (bound) goto fail;
    if (lstat(path, &info)) goto fail;
    service->device = info.st_dev; service->inode = info.st_ino;
    if (!S_ISSOCK(info.st_mode) || info.st_uid != getuid() || chmod(path, 0600) || listen(service->fd, CLIENT_LIMIT)) goto fail;
    nn_error_set(error, cap, "");
    return service;
fail:
    nn_error_set(error, cap, "unable to bind private local socket");
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
        client->input = malloc(NN_AUTOMATION_REQUEST_LIMIT + 1);
        if (!client->input) close_client(client);
    }
    for (size_t i = 0; i < CLIENT_LIMIT; ++i) {
        Client *client = &service->clients[i];
        if (client->fd < 0) continue;
        if (now() > client->deadline) { close_client(client); continue; }
        if (!client->output) {
            size_t capacity = NN_AUTOMATION_REQUEST_LIMIT - client->used;
            if (capacity > 65536) capacity = 65536;
            ssize_t count = recv(client->fd, client->input + client->used, capacity, 0);
            if (count > 0) {
                client->used += (size_t)count; client->input[client->used] = '\0';
                char *newline = memchr(client->input, '\n', client->used);
                if (newline) {
                    *newline = '\0';
                    if (memchr(client->input, '\0', (size_t)(newline - client->input)))
                        client->output = nn_text_copy("{\"ok\":false,\"error\":\"embedded NUL in request\"}\n");
                    else client->output = nn_automation_dispatch(service->app, client->input, service->callback, service->user);
                    if (client->output) {
                        yyjson_doc *response = yyjson_read(client->output, strlen(client->output), 0);
                        yyjson_doc *request = yyjson_read(client->input, strlen(client->input), 0);
                        const char *op = request ? yyjson_get_str(yyjson_obj_get(yyjson_doc_get_root(request), "operation")) : NULL;
                        bool readonly = op && (!strcmp(op, "project.snapshot") || !strcmp(op, "analysis.diagnostics") || !strcmp(op, "ui.inspect") || !strcmp(op, "ui.screenshot"));
                        changed |= !readonly && response && yyjson_get_bool(yyjson_obj_get(yyjson_doc_get_root(response), "ok"));
                        yyjson_doc_free(request);
                        yyjson_doc_free(response);
                    }
                } else if (client->used == NN_AUTOMATION_REQUEST_LIMIT)
                    client->output = nn_text_copy("{\"ok\":false,\"error\":\"request too large\"}\n");
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
