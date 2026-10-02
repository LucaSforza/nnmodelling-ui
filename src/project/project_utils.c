#define _XOPEN_SOURCE 700
#define _POSIX_C_SOURCE 200809L
#include "project_internal.h"
#include "utils/utils.h"

#include <dirent.h>
#include <errno.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

const char *nn_project_string_field(yyjson_val *object, const char *key)
{
    return yyjson_get_str(yyjson_obj_get(object, key));
}

bool nn_project_valid_id(const char *id)
{
    if (!id || !*id || strlen(id) > 96 || !strcmp(id, ".") || !strcmp(id, "..")) return false;
    for (const unsigned char *c = (const unsigned char *)id; *c; ++c)
        if (!((*c >= 'a' && *c <= 'z') || (*c >= 'A' && *c <= 'Z') ||
              (*c >= '0' && *c <= '9') || *c == '-' || *c == '_' || *c == '.'))
            return false;
    return true;
}

bool nn_project_valid_semver(const char *text)
{
    if (!text || !*text) return false;
    for (unsigned part = 0; part < 3; ++part) {
        unsigned long value = 0;
        if (*text < '0' || *text > '9' || (*text == '0' && text[1] >= '0' && text[1] <= '9')) return false;
        while (*text >= '0' && *text <= '9') {
            unsigned digit = (unsigned)(*text++ - '0');
            if (value > (ULONG_MAX - digit) / 10) return false;
            value = value * 10 + digit;
        }
        if (part < 2) { if (*text++ != '.') return false; }
        else if (*text) return false;
    }
    return true;
}

char *nn_project_safe_resource_dir(const char *root, const char *relative,
                               char *error, size_t capacity)
{
    if (!relative || !*relative || relative[0] == '/' || strlen(relative) > 512) {
        nn_errorf(error, capacity, "invalid resource path");
        return NULL;
    }
    char *path = nn_text_copy(root);
    if (!path) { nn_error_set(error, capacity, "out of memory building resource path"); return NULL; }
    const char *part = relative;
    for (const char *end = relative;; ++end) {
        if (*end != '/' && *end != '\0') continue;
        size_t length = (size_t)(end - part);
        if (!length || (length == 1 && part[0] == '.') ||
            (length == 2 && part[0] == '.' && part[1] == '.')) {
            nn_errorf(error, capacity, "resource path traversal rejected: %s", relative);
            free(path);
            return NULL;
        }
        for (size_t i = 0; i < length; ++i) {
            unsigned char c = (unsigned char)part[i];
            if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                  (c >= '0' && c <= '9') || c == '-' || c == '_' || c == '.')) {
                nn_errorf(error, capacity, "invalid resource path: %s", relative);
                free(path);
                return NULL;
            }
        }
        char *segment = malloc(length + 1);
        if (!segment) { free(path); nn_error_set(error, capacity, "out of memory building resource path"); return NULL; }
        memcpy(segment, part, length);
        segment[length] = '\0';
        char *next = nn_path_join(path, segment);
        free(segment);
        free(path);
        path = next;
        if (!path) { nn_error_set(error, capacity, "unable to build resource path"); return NULL; }
        struct stat st;
        if (lstat(path, &st) || !S_ISDIR(st.st_mode)) {
            nn_errorf(error, capacity, "resource directory missing or symlinked: %s", relative);
            free(path);
            return NULL;
        }
        if (*end == '\0') break;
        part = end + 1;
    }
    return path;
}

yyjson_doc *nn_project_read_document(const char *path, char *error, size_t capacity)
{
    struct stat st;
    if (lstat(path, &st) || !S_ISREG(st.st_mode) || st.st_size > 16 * 1024 * 1024) {
        nn_errorf(error, capacity, "missing, symlinked or oversized JSON: %s", path);
        return NULL;
    }
    yyjson_read_err parse_error;
    yyjson_doc *document = yyjson_read_file(path, 0, NULL, &parse_error);
    if (!document) nn_errorf(error, capacity, "JSON parse error in %s: %s", path, parse_error.msg);
    return document;
}

bool nn_project_write_all(int fd, const char *data, size_t length)
{
    while (length) {
        ssize_t written = write(fd, data, length);
        if (written < 0 && errno == EINTR) continue;
        if (written <= 0) return false;
        data += written;
        length -= (size_t)written;
    }
    return true;
}

void nn_project_remove_tree(const char *path)
{
    struct stat st;
    if (!path || lstat(path, &st)) return;
    if (S_ISDIR(st.st_mode)) {
        DIR *dir = opendir(path);
        if (dir) {
            struct dirent *entry;
            while ((entry = readdir(dir))) {
                if (strcmp(entry->d_name, ".") && strcmp(entry->d_name, "..")) {
                    char *child = nn_path_join(path, entry->d_name);
                    if (child) { nn_project_remove_tree(child); free(child); }
                }
            }
            closedir(dir);
        }
        rmdir(path);
    } else unlink(path);
}
