#include "utils.h"

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void nn_error_set(char *error, size_t capacity, const char *message)
{
    if (error && capacity) snprintf(error, capacity, "%s", message ? message : "");
}

void nn_errorf(char *error, size_t capacity, const char *format, ...)
{
    if (!error || !capacity) return;
    va_list args;
    va_start(args, format);
    int written = vsnprintf(error, capacity, format, args);
    va_end(args);
    if (written < 0) nn_error_set(error, capacity, "Unable to format error");
}

bool nn_fail(char *error, size_t capacity, const char *message)
{
    nn_error_set(error, capacity, message);
    return false;
}

char *nn_text_copy(const char *text)
{
    if (!text) return NULL;
    size_t length = strlen(text);
    if (length == SIZE_MAX) return NULL;
    char *copy = malloc(length + 1);
    if (copy) memcpy(copy, text, length + 1);
    return copy;
}

char *nn_path_join(const char *left, const char *right)
{
    if (!left || !right) return NULL;
    size_t a = strlen(left), b = strlen(right);
    const size_t limit = 1024u * 1024u;
    if (b > limit || a > limit - b || a + b >= limit) return NULL;
    size_t length = a + 1 + b;
    if (length == SIZE_MAX) return NULL;
    char *path = malloc(length + 1);
    if (!path) return NULL;
    memcpy(path, left, a);
    path[a] = '/';
    memcpy(path + a + 1, right, b);
    path[length] = '\0';
    return path;
}

bool nn_join_handle_order(const char *handle, size_t *order)
{
    if (!handle || !order || strncmp(handle, "in-", 3) || !handle[3] || handle[3] == '0') return false;
    size_t number = 0;
    for (const char *p = handle + 3; *p; ++p) {
        if (*p < '0' || *p > '9') return false;
        unsigned digit = (unsigned)(*p - '0');
        if (number > (SIZE_MAX - digit) / 10) return false;
        number = number * 10 + digit;
    }
    *order = number;
    return number != 0;
}
