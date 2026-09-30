#define _POSIX_C_SOURCE 200809L
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

#define REQUEST_LIMIT (2u * 1024u * 1024u)
#define RESPONSE_LIMIT (16u * 1024u * 1024u)

static void help(void)
{
    puts("Usage: nnmodelctl [--socket PATH] OPERATION [JSON_OBJECT|-]\n"
         "Socket defaults to NNMODELLING_SOCKET. Start UI with --socket PATH.\n"
         "Use - to read the argument object from stdin. Exit 0 success, 1 rejected, 2 transport/usage.\n"
         "project.open {path}; project.create {parent,id,name,template:blank|mnist-mlp|mnist-vae}\n"
          "project.save {}; project.close {discard:false}; project.snapshot {}\n"
          "analysis.diagnostics {} returns model/Lua problems and successful tensors\n"
         "stereotype.create {id,version,definition:{name,kind,view:{color,width,height},parameters:{}},\n"
         "                   inference:LuaSource,dependencies:{packageId:versionConstraint}}\n"
         "dataset.create {id,version,definition:{name,batch:{inputs:{slot:{dtype,shape}},targets:{}}},select:false}\n"
         "dataset.select {id,version}\n"
         "node.add {id,package,version,scope:\"\",x:0,y:0}; node.remove {id}; node.move {id,x,y}\n"
         "node.rename {id,name}; node.parameter {id,key,value:text}\n"
         "edge.connect {id,source,sourceHandle,target,targetHandle}; edge.disconnect {id}\n"
          "ui.inspect {}; ui.scope {id}; ui.reveal {id}; ui.arrange {}; ui.screenshot {path}\n"
         "See docs/knowledge/contracts/automation.md for complete payloads and transaction semantics.");
}

static char *stdin_json(void)
{
    char *text = malloc(REQUEST_LIMIT + 1);
    if (!text) return NULL;
    size_t length = fread(text, 1, REQUEST_LIMIT, stdin);
    if (ferror(stdin) || (length == REQUEST_LIMIT && fgetc(stdin) != EOF) || memchr(text, '\0', length)) {
        free(text); return NULL;
    }
    text[length] = '\0';
    return text;
}

int main(int argc, char **argv)
{
    const char *path = getenv("NNMODELLING_SOCKET"), *operation = NULL, *payload = "{}";
    bool have_payload = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) { help(); return 0; }
        if (!strcmp(argv[i], "--socket")) {
            if (++i == argc) { help(); return 2; }
            path = argv[i];
        } else if (!operation) operation = argv[i];
        else if (!have_payload) { payload = argv[i]; have_payload = true; }
        else { help(); return 2; }
    }
    if (!operation || !path || !*path) { help(); return 2; }
    struct sockaddr_un address = { .sun_family = AF_UNIX };
    if (strlen(path) >= sizeof(address.sun_path)) { fputs("socket path too long\n", stderr); return 2; }
    char *owned_payload = !strcmp(payload, "-") ? stdin_json() : NULL;
    if (!strcmp(payload, "-") && !owned_payload) { fputs("invalid or oversized stdin\n", stderr); return 2; }
    if (owned_payload) payload = owned_payload;
    yyjson_doc *args = strlen(payload) <= REQUEST_LIMIT ? yyjson_read(payload, strlen(payload), 0) : NULL;
    yyjson_mut_doc *request = yyjson_mut_doc_new(NULL);
    if (!args || !yyjson_is_obj(yyjson_doc_get_root(args)) || !request) {
        fputs("arguments must be a JSON object\n", stderr);
        yyjson_doc_free(args); yyjson_mut_doc_free(request); free(owned_payload); return 2;
    }
    yyjson_mut_val *root = yyjson_mut_obj(request);
    yyjson_mut_val *arguments = yyjson_val_mut_copy(request, yyjson_doc_get_root(args));
    if (!root || !arguments || !yyjson_mut_obj_add_strcpy(request, root, "operation", operation) ||
        !yyjson_mut_obj_add_val(request, root, "args", arguments)) {
        fputs("out of memory building request\n", stderr);
        yyjson_doc_free(args); yyjson_mut_doc_free(request); free(owned_payload); return 2;
    }
    yyjson_mut_doc_set_root(request, root);
    size_t length = 0;
    char *text = yyjson_mut_write(request, YYJSON_WRITE_NEWLINE_AT_END, &length);
    yyjson_doc_free(args); yyjson_mut_doc_free(request); free(owned_payload);
    if (!text || length > REQUEST_LIMIT) { free(text); fputs("request too large\n", stderr); return 2; }
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
    char *response = malloc(RESPONSE_LIMIT + 1);
    if (!response) { close(fd); return 2; }
    size_t used = 0;
    bool complete = false;
    while (used < RESPONSE_LIMIT) {
        ssize_t count = recv(fd, response + used, RESPONSE_LIMIT - used, 0);
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
