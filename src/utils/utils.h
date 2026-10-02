#ifndef NN_UTILS_H
#define NN_UTILS_H

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Error buffers are optional, bounded and never allocate. */
void nn_error_set(char *error, size_t capacity, const char *message);
void nn_errorf(char *error, size_t capacity, const char *format, ...);
bool nn_fail(char *error, size_t capacity, const char *message);
/* Malloc-owned ABI text, not a dynamic string builder. NULL remains NULL. */
char *nn_text_copy(const char *text);
/* Mechanical joining only; callers enforce filesystem confinement. */
char *nn_path_join(const char *left, const char *right);
bool nn_join_handle_order(const char *handle, size_t *order);

#ifdef __cplusplus
}
#endif
#endif
