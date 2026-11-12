#define _DEFAULT_SOURCE
#ifdef __APPLE__
#define _DARWIN_C_SOURCE
#endif
#define _POSIX_C_SOURCE 200809L
#include "activity/activity.h"
#include "utils/utils.h"
#include "yyjson.h"

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <time.h>
#include <unistd.h>
#ifdef __linux__
#include <sys/random.h>
#endif
#ifdef __APPLE__
#include <stdlib.h>
#endif

enum { NN_ACTIVITY_STRING_LIMIT = 1024, NN_ACTIVITY_DEPTH_LIMIT = 64 };

struct NNActivityLog {
    int fd;
    char session[37];
    unsigned long long seq;
    long long start_ms;
};

static long long monotonic_ms(void)
{
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now)) return 0;
    return (long long)now.tv_sec * 1000 + now.tv_nsec / 1000000;
}

bool nn_activity_resolve_path(char *buffer, size_t capacity, char *error, size_t cap)
{
    nn_error_set(error, cap, "");
    if (!buffer || !capacity) return nn_fail(error, cap, "activity log path buffer is empty");
    buffer[0] = '\0';
    const char *value = getenv("NNMODELLING_ACTIVITY_LOG");
    const char *text = value && *value ? value : NULL;
    if (text && (!strcasecmp(text, "off") || !strcasecmp(text, "0") ||
                 !strcasecmp(text, "false") || !strcasecmp(text, "none"))) return false;
    int length;
    if (text) {
        length = snprintf(buffer, capacity, "%s", text);
    } else {
        const char *home = getenv("HOME");
        if (!home || !*home) return false;
        length = snprintf(buffer, capacity, "%s/nnmodelling.log", home);
    }
    if (length < 0 || (size_t)length >= capacity) {
        buffer[0] = '\0';
        return nn_fail(error, cap, "activity log path is too long");
    }
    return true;
}

static void random_bytes(unsigned char *bytes, size_t count)
{
    size_t done = 0;
#ifdef __APPLE__
    arc4random_buf(bytes, count);
    done = count;
#endif
#ifdef __linux__
    while (done < count) {
        ssize_t got = getrandom(bytes + done, count - done, 0);
        if (got < 0 && errno == EINTR) continue;
        if (got <= 0) break;
        done += (size_t)got;
    }
#endif
    if (done < count) {
        int fd = open("/dev/urandom", O_RDONLY | O_CLOEXEC);
        while (fd >= 0 && done < count) {
            ssize_t got = read(fd, bytes + done, count - done);
            if (got < 0 && errno == EINTR) continue;
            if (got <= 0) break;
            done += (size_t)got;
        }
        if (fd >= 0) close(fd);
    }
    if (done < count) {
        /* Last resort: session IDs only need to differ between processes. */
        struct timespec now = {0};
        clock_gettime(CLOCK_REALTIME, &now);
        uint64_t state = (uint64_t)now.tv_nsec ^ ((uint64_t)now.tv_sec << 20) ^
                         ((uint64_t)getpid() << 40) ^ (uint64_t)(uintptr_t)bytes;
        for (; done < count; ++done) {
            state = state * 6364136223846793005ULL + 1442695040888963407ULL;
            bytes[done] = (unsigned char)(state >> 56);
        }
    }
}

static void make_uuid(char out[37])
{
    unsigned char b[16];
    random_bytes(b, sizeof(b));
    b[6] = (unsigned char)((b[6] & 0x0f) | 0x40);
    b[8] = (unsigned char)((b[8] & 0x3f) | 0x80);
    snprintf(out, 37, "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
             b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
             b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]);
}

NNActivityLog *nn_activity_open(const char *path, char *error, size_t cap)
{
    nn_error_set(error, cap, "");
    if (!path || !*path) { nn_fail(error, cap, "activity log path is empty"); return NULL; }
    NNActivityLog *log = calloc(1, sizeof(*log));
    if (!log) { nn_fail(error, cap, "out of memory opening activity log"); return NULL; }
    log->fd = open(path, O_WRONLY | O_APPEND | O_CREAT | O_CLOEXEC, 0600);
    if (log->fd < 0) {
        nn_errorf(error, cap, "cannot open activity log: %s", strerror(errno));
        free(log);
        return NULL;
    }
    make_uuid(log->session);
    log->start_ms = monotonic_ms();
    return log;
}

void nn_activity_close(NNActivityLog *log)
{
    if (!log) return;
    if (log->fd >= 0) close(log->fd);
    free(log);
}

const char *nn_activity_session(const NNActivityLog *log)
{
    return log ? log->session : "";
}

static void format_timestamp(char *out, size_t capacity)
{
    struct timespec now;
    struct tm local;
    if (clock_gettime(CLOCK_REALTIME, &now) || !localtime_r(&now.tv_sec, &local)) {
        snprintf(out, capacity, "1970-01-01T00:00:00.000+00:00");
        return;
    }
    char date[32];
    if (!strftime(date, sizeof(date), "%Y-%m-%dT%H:%M:%S", &local)) {
        snprintf(out, capacity, "1970-01-01T00:00:00.000+00:00");
        return;
    }
    long offset = (long)local.tm_gmtoff;
    char sign = offset < 0 ? '-' : '+';
    if (offset < 0) offset = -offset;
    /* Bounded ints let compilers prove the record fits (-Wformat-truncation). */
    unsigned millis = (unsigned)(now.tv_nsec / 1000000) % 1000u;
    unsigned hours = (unsigned)(offset / 3600) % 100u;
    unsigned minutes = (unsigned)(offset % 3600 / 60) % 60u;
    snprintf(out, capacity, "%s.%03u%c%02u:%02u", date, millis, sign, hours, minutes);
}

static uint64_t fnv1a64(const char *bytes, size_t length)
{
    uint64_t hash = 14695981039346656037ULL;
    for (size_t i = 0; i < length; ++i) {
        hash ^= (unsigned char)bytes[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

static yyjson_mut_val *copy_value(yyjson_mut_doc *doc, yyjson_val *value, unsigned depth)
{
    if (depth > NN_ACTIVITY_DEPTH_LIMIT) return NULL;
    if (yyjson_is_str(value) && yyjson_get_len(value) > NN_ACTIVITY_STRING_LIMIT) {
        char hash[17];
        snprintf(hash, sizeof(hash), "%016llx",
                 (unsigned long long)fnv1a64(yyjson_get_str(value), yyjson_get_len(value)));
        yyjson_mut_val *object = yyjson_mut_obj(doc);
        if (!object || !yyjson_mut_obj_add_bool(doc, object, "truncated", true) ||
            !yyjson_mut_obj_add_uint(doc, object, "len", yyjson_get_len(value)) ||
            !yyjson_mut_obj_add_strcpy(doc, object, "fnv1a64", hash)) return NULL;
        return object;
    }
    if (yyjson_is_arr(value)) {
        yyjson_mut_val *array = yyjson_mut_arr(doc);
        size_t index, count;
        yyjson_val *item;
        if (!array) return NULL;
        yyjson_arr_foreach(value, index, count, item) {
            yyjson_mut_val *copy = copy_value(doc, item, depth + 1);
            if (!copy || !yyjson_mut_arr_append(array, copy)) return NULL;
        }
        return array;
    }
    if (yyjson_is_obj(value)) {
        yyjson_mut_val *object = yyjson_mut_obj(doc);
        size_t index, count;
        yyjson_val *key, *item;
        if (!object) return NULL;
        yyjson_obj_foreach(value, index, count, key, item) {
            yyjson_mut_val *name = yyjson_val_mut_copy(doc, key);
            yyjson_mut_val *copy = copy_value(doc, item, depth + 1);
            if (!name || !copy || !yyjson_mut_obj_add(object, name, copy)) return NULL;
        }
        return object;
    }
    return yyjson_val_mut_copy(doc, value);
}

static yyjson_mut_val *data_value(yyjson_mut_doc *doc, const char *data_json)
{
    yyjson_mut_val *result = NULL;
    if (!data_json) return yyjson_mut_obj(doc);
    yyjson_doc *parsed = yyjson_read(data_json, strlen(data_json), 0);
    yyjson_val *root = parsed ? yyjson_doc_get_root(parsed) : NULL;
    if (yyjson_is_obj(root)) result = copy_value(doc, root, 0);
    yyjson_doc_free(parsed);
    if (result) return result;
    result = yyjson_mut_obj(doc);
    if (result) yyjson_mut_obj_add_bool(doc, result, "invalid_data", true);
    return result;
}

static bool write_all(int fd, const char *bytes, size_t length)
{
    while (length) {
        ssize_t written = write(fd, bytes, length);
        if (written < 0 && errno == EINTR) continue;
        if (written <= 0) return false;
        bytes += written;
        length -= (size_t)written;
    }
    return true;
}

static char *build_line(NNActivityLog *log, const char *source, const char *event,
                        bool ok, const char *error_text, const char *data_json, size_t *length)
{
    yyjson_mut_doc *doc = yyjson_mut_doc_new(NULL);
    if (!doc) return NULL;
    char timestamp[48];
    format_timestamp(timestamp, sizeof(timestamp));
    long long elapsed = monotonic_ms() - log->start_ms;
    yyjson_mut_val *root = yyjson_mut_obj(doc);
    bool built = root &&
        yyjson_mut_obj_add_int(doc, root, "v", 1) &&
        yyjson_mut_obj_add_strcpy(doc, root, "ts", timestamp) &&
        yyjson_mut_obj_add_int(doc, root, "t", elapsed < 0 ? 0 : elapsed) &&
        yyjson_mut_obj_add_strcpy(doc, root, "session", log->session) &&
        yyjson_mut_obj_add_uint(doc, root, "seq", log->seq + 1) &&
        yyjson_mut_obj_add_strcpy(doc, root, "src", source ? source : "core") &&
        yyjson_mut_obj_add_strcpy(doc, root, "event", event) &&
        yyjson_mut_obj_add_bool(doc, root, "ok", ok);
    if (built && !ok)
        built = yyjson_mut_obj_add_strcpy(doc, root, "error", error_text ? error_text : "");
    yyjson_mut_val *data = built ? data_value(doc, data_json) : NULL;
    char *line = NULL;
    if (data && yyjson_mut_obj_add_val(doc, root, "data", data)) {
        yyjson_mut_doc_set_root(doc, root);
        line = yyjson_mut_write(doc, YYJSON_WRITE_NEWLINE_AT_END, length);
    }
    yyjson_mut_doc_free(doc);
    return line;
}

bool nn_activity_emit(NNActivityLog *log, const char *source, const char *event,
                      bool ok, const char *error_text, const char *data_json)
{
    if (!log) return true;
    if (log->fd < 0) return false;
    if (!event) return true;
    size_t length = 0;
    char *line = build_line(log, source, event, ok, error_text, data_json, &length);
    if (!line) return true; /* unserializable record: skipped, log stays usable */
    ++log->seq;
    bool written = write_all(log->fd, line, length);
    free(line);
    if (!written) {
        fprintf(stderr, "nnmodelling: activity log disabled after write failure: %s\n",
                strerror(errno));
        close(log->fd);
        log->fd = -1;
    }
    return written;
}

char *nn_activity_redact_path(const char *path)
{
    if (!path) return NULL;
    const char *home = getenv("HOME");
    size_t length = home ? strlen(home) : 0;
    while (length > 1 && home[length - 1] == '/') --length;
    if (length > 1 && !strncmp(path, home, length) && (!path[length] || path[length] == '/')) {
        size_t rest = strlen(path + length);
        char *result = malloc(rest + 2);
        if (!result) return NULL;
        result[0] = '~';
        memcpy(result + 1, path + length, rest + 1);
        return result;
    }
    return nn_text_copy(path);
}
