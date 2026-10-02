#include "cli_internal.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/un.h>

int main(int argc, char **argv)
{
    const char *path = getenv("NNMODELLING_SOCKET"), *operation = NULL, *payload = "{}";
    bool have_payload = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) { nn_cli_help(); return 0; }
        if (!strcmp(argv[i], "--socket")) {
            if (++i == argc) { nn_cli_help(); return 2; }
            path = argv[i];
        } else if (!operation) operation = argv[i];
        else if (!have_payload) { payload = argv[i]; have_payload = true; }
        else { nn_cli_help(); return 2; }
    }
    if (!operation || !path || !*path) { nn_cli_help(); return 2; }
    struct sockaddr_un address;
    if (strlen(path) >= sizeof(address.sun_path)) { fputs("socket path too long\n", stderr); return 2; }
    size_t length = 0;
    char *text = nn_cli_request(operation, payload, &length);
    if (!text) return 2;
    return nn_cli_exchange(path, text, length);
}
