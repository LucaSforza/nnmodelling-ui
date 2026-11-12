#ifndef NN_ACTIVITY_H
#define NN_ACTIVITY_H

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct NNActivityLog NNActivityLog;

/* false + empty error when disabled; false + error on failure; true + path. */
bool nn_activity_resolve_path(char *buffer, size_t capacity, char *error, size_t cap);
/* Appends to path (mode 0600). One log per process; not thread-safe. */
NNActivityLog *nn_activity_open(const char *path, char *error, size_t cap);
void nn_activity_close(NNActivityLog *log);
const char *nn_activity_session(const NNActivityLog *log);
/* data_json: NULL or JSON object text. Returns false on write failure. */
bool nn_activity_emit(NNActivityLog *log, const char *source, const char *event,
                      bool ok, const char *error_text, const char *data_json);
/* Owned copy with $HOME prefix replaced by "~"; release with free. */
char *nn_activity_redact_path(const char *path);

#ifdef __cplusplus
}
#endif
#endif
