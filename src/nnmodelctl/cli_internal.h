#ifndef NN_CLI_INTERNAL_H
#define NN_CLI_INTERNAL_H

#include <stddef.h>

#define NN_CLI_REQUEST_LIMIT (2u * 1024u * 1024u)
#define NN_CLI_RESPONSE_LIMIT (16u * 1024u * 1024u)

void nn_cli_help(void);
char *nn_cli_request(const char *operation, const char *payload, size_t *length);
/* Consumes the malloc-owned request on every path; prints response/errors. */
int nn_cli_exchange(const char *path, char *request, size_t length);

#endif
