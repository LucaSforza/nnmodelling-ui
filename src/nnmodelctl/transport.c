#define _POSIX_C_SOURCE 200809L
#include "cli_internal.h"
#include "yyjson.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/un.h>
#include <unistd.h>

int nn_cli_exchange(const char *path, char *text, size_t length)
{
    struct sockaddr_un address = { .sun_family = AF_UNIX };
    struct stat info;
    if (lstat(path, &info) || !S_ISSOCK(info.st_mode) || info.st_uid != getuid() || (info.st_mode & 077)) {
        free(text); fputs("socket must be private and owned by current user\n", stderr); return 2;
    }
    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    struct timeval timeout = { .tv_sec = 10 };
    if (fd < 0) { free(text); perror("socket"); return 2; }
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
    strcpy(address.sun_path, path);
    if (connect(fd, (struct sockaddr *)&address, sizeof(address))) {
        perror("connect"); close(fd); free(text); return 2;
    }
    size_t sent = 0;
    while (sent < length) {
        ssize_t count = send(fd, text + sent, length - sent, MSG_NOSIGNAL);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) { perror("send"); close(fd); free(text); return 2; }
        sent += (size_t)count;
    }
    free(text);
    char *response = malloc(NN_CLI_RESPONSE_LIMIT + 1);
    if (!response) { close(fd); return 2; }
    size_t used = 0;
    bool complete = false;
    while (used < NN_CLI_RESPONSE_LIMIT) {
        ssize_t count = recv(fd, response + used, NN_CLI_RESPONSE_LIMIT - used, 0);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) break;
        used += (size_t)count;
        if (memchr(response, '\n', used)) { complete = true; break; }
    }
    close(fd); response[used] = '\0';
    yyjson_doc *doc = complete ? yyjson_read(response, used, 0) : NULL;
    yyjson_val *ok = doc ? yyjson_obj_get(yyjson_doc_get_root(doc), "ok") : NULL;
    int exit_code = yyjson_is_bool(ok) ? (yyjson_get_bool(ok) ? 0 : 1) : 2;
    if (exit_code == 2) fputs("invalid, incomplete or timed out command response\n", stderr);
    else fputs(response, stdout);
    yyjson_doc_free(doc); free(response);
    return exit_code;
}
