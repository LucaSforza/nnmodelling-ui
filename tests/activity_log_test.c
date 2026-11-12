#define _XOPEN_SOURCE 700
#include "activity/activity.h"
#include "application/application.h"
#include "yyjson.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

typedef struct { yyjson_doc **docs; size_t count; char *text; } Lines;

static char *slurp(const char *path)
{
    FILE *file = fopen(path, "rb");
    assert(file);
    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    fseek(file, 0, SEEK_SET);
    char *text = calloc((size_t)size + 1, 1);
    assert(text && fread(text, 1, (size_t)size, file) == (size_t)size);
    fclose(file);
    return text;
}

static Lines read_lines(const char *path)
{
    Lines lines = {.text = slurp(path)};
    char *cursor = lines.text;
    while (*cursor) {
        char *end = strchr(cursor, '\n');
        assert(end); /* every record is newline terminated */
        lines.docs = realloc(lines.docs, (lines.count + 1) * sizeof(*lines.docs));
        lines.docs[lines.count] = yyjson_read(cursor, (size_t)(end - cursor), 0);
        assert(lines.docs[lines.count] && yyjson_is_obj(yyjson_doc_get_root(lines.docs[lines.count])));
        ++lines.count;
        cursor = end + 1;
    }
    return lines;
}

static void free_lines(Lines *lines)
{
    for (size_t i = 0; i < lines->count; ++i) yyjson_doc_free(lines->docs[i]);
    free(lines->docs);
    free(lines->text);
}

static yyjson_val *rec(const Lines *lines, size_t index) { return yyjson_doc_get_root(lines->docs[index]); }
static const char *event_of(const Lines *lines, size_t index)
{
    return yyjson_get_str(yyjson_obj_get(rec(lines, index), "event"));
}
static yyjson_val *data_of(const Lines *lines, size_t index)
{
    return yyjson_obj_get(rec(lines, index), "data");
}
static const char *text_of(yyjson_val *object, const char *key)
{
    return yyjson_get_str(yyjson_obj_get(object, key));
}
static size_t find_event(const Lines *lines, size_t from, const char *name)
{
    for (size_t i = from; i < lines->count; ++i)
        if (!strcmp(event_of(lines, i), name)) return i;
    assert(!"event not found");
    return 0;
}

static void test_paths(void)
{
    char path[256], error[128];
    unsetenv("NNMODELLING_ACTIVITY_LOG");
    setenv("HOME", "/home/someone", 1);
    assert(nn_activity_resolve_path(path, sizeof(path), error, sizeof(error)));
    assert(!strcmp(path, "/home/someone/nnmodelling.log"));
    setenv("NNMODELLING_ACTIVITY_LOG", "", 1);
    assert(nn_activity_resolve_path(path, sizeof(path), error, sizeof(error)));
    assert(!strcmp(path, "/home/someone/nnmodelling.log"));
    setenv("NNMODELLING_ACTIVITY_LOG", "/tmp/x.log", 1);
    assert(nn_activity_resolve_path(path, sizeof(path), error, sizeof(error)) && !strcmp(path, "/tmp/x.log"));
    const char *off[] = {"off", "OFF", "0", "False", "NONE"};
    for (size_t i = 0; i < 5; ++i) {
        setenv("NNMODELLING_ACTIVITY_LOG", off[i], 1);
        strcpy(error, "stale");
        assert(!nn_activity_resolve_path(path, sizeof(path), error, sizeof(error)) && !error[0]);
    }
    unsetenv("NNMODELLING_ACTIVITY_LOG");
    unsetenv("HOME");
    strcpy(error, "stale");
    assert(!nn_activity_resolve_path(path, sizeof(path), error, sizeof(error)) && !error[0]);
    setenv("NNMODELLING_ACTIVITY_LOG", "/tmp/a-very-long-name.log", 1);
    assert(!nn_activity_resolve_path(path, 8, error, sizeof(error)) && error[0]);
    unsetenv("NNMODELLING_ACTIVITY_LOG");
    setenv("HOME", "/home/someone", 1);
}

static void test_redaction(void)
{
    setenv("HOME", "/home/someone", 1);
    char *a = nn_activity_redact_path("/home/someone/projects/x");
    char *b = nn_activity_redact_path("/home/someone");
    char *c = nn_activity_redact_path("/home/someone2/x");
    char *d = nn_activity_redact_path("/opt/x");
    assert(!strcmp(a, "~/projects/x") && !strcmp(b, "~") && !strcmp(c, "/home/someone2/x") &&
           !strcmp(d, "/opt/x"));
    free(a); free(b); free(c); free(d);
}

static void test_format(const char *dir)
{
    char path[512], error[256];
    snprintf(path, sizeof(path), "%s/format.log", dir);
    NNActivityLog *first = nn_activity_open(path, error, sizeof(error));
    assert(first && strlen(nn_activity_session(first)) == 36);
    assert(nn_activity_emit(first, "ui", "one", true, NULL, NULL));
    assert(nn_activity_emit(first, "core", "two", false, "bad \"thing\"\n\xc3\xa9", "{\"a\":\"q\\\"\\n\"}"));
    assert(nn_activity_emit(first, "ipc", "three", true, NULL, "not json"));
    assert(nn_activity_emit(first, "ui", "four", true, NULL, "[1,2]"));
    char big[2000];
    memset(big, 'a', sizeof(big) - 1);
    big[sizeof(big) - 1] = '\0';
    char json[4200];
    snprintf(json, sizeof(json), "{\"s\":\"%s\",\"n\":[{\"d\":\"%s\"}],\"ok\":\"%s\"}", big, big, "short");
    assert(nn_activity_emit(first, "ui", "five", true, NULL, json));
    assert(nn_activity_emit(NULL, "ui", "x", true, NULL, NULL));
    char session_one[37];
    strcpy(session_one, nn_activity_session(first));
    nn_activity_close(first);
    nn_activity_close(NULL);
    struct stat info;
    assert(stat(path, &info) == 0 && (info.st_mode & 0777) == 0600);

    NNActivityLog *second = nn_activity_open(path, error, sizeof(error));
    assert(second && strcmp(session_one, nn_activity_session(second)));
    assert(nn_activity_emit(second, NULL, "six", true, NULL, "{}"));
    nn_activity_close(second);

    Lines lines = read_lines(path);
    assert(lines.count == 6);
    /* Exact key order of the first record. */
    const char *first_line = lines.text;
    const char *keys[] = {"\"v\":1,\"ts\":\"", "\"t\":", "\"session\":\"", "\"seq\":1,\"src\":\"ui\",\"event\":\"one\",\"ok\":true,\"data\":{}}\n"};
    const char *at = first_line;
    for (size_t i = 0; i < 4; ++i) { at = strstr(at, keys[i]); assert(at); }
    assert(!strstr(first_line, "\"error\"") || strstr(first_line, "\"error\"") > strchr(first_line, '\n'));
    yyjson_val *r0 = rec(&lines, 0);
    assert(yyjson_get_int(yyjson_obj_get(r0, "v")) == 1);
    const char *ts = text_of(r0, "ts");
    assert(strlen(ts) == 29 && ts[4] == '-' && ts[10] == 'T' && ts[19] == '.' &&
           (ts[23] == '+' || ts[23] == '-') && ts[26] == ':');
    assert(yyjson_get_sint(yyjson_obj_get(r0, "t")) >= 0);
    assert(!strcmp(text_of(r0, "session"), session_one));
    assert(yyjson_get_uint(yyjson_obj_get(rec(&lines, 4), "seq")) == 5);
    yyjson_val *r1 = rec(&lines, 1);
    assert(!yyjson_get_bool(yyjson_obj_get(r1, "ok")));
    assert(!strcmp(text_of(r1, "error"), "bad \"thing\"\n\xc3\xa9"));
    assert(!strcmp(text_of(data_of(&lines, 1), "a"), "q\"\n"));
    assert(yyjson_obj_get(data_of(&lines, 2), "invalid_data"));
    assert(yyjson_obj_get(data_of(&lines, 3), "invalid_data"));
    yyjson_val *truncated = yyjson_obj_get(data_of(&lines, 4), "s");
    assert(yyjson_is_obj(truncated) && yyjson_get_bool(yyjson_obj_get(truncated, "truncated")) &&
           yyjson_get_uint(yyjson_obj_get(truncated, "len")) == 1999);
    /* FNV-1a 64 of 1999 'a' bytes, computed independently. */
    unsigned long long hash = 14695981039346656037ULL;
    for (size_t i = 0; i < 1999; ++i) { hash ^= 'a'; hash *= 1099511628211ULL; }
    char expected[17];
    snprintf(expected, sizeof(expected), "%016llx", hash);
    assert(!strcmp(text_of(truncated, "fnv1a64"), expected));
    yyjson_val *nested = yyjson_obj_get(yyjson_arr_get(yyjson_obj_get(data_of(&lines, 4), "n"), 0), "d");
    assert(yyjson_is_obj(nested) && yyjson_obj_get(nested, "truncated"));
    assert(!strcmp(text_of(data_of(&lines, 4), "ok"), "short"));
    assert(!strcmp(text_of(rec(&lines, 5), "src"), "core"));
    assert(yyjson_get_uint(yyjson_obj_get(rec(&lines, 5), "seq")) == 1);
    assert(strcmp(text_of(rec(&lines, 5), "session"), session_one));
    free_lines(&lines);

    /* Open failure reports an error and returns no log. */
    char bad[512];
    snprintf(bad, sizeof(bad), "%s/format.log/child/x.log", dir); /* file used as directory */
    error[0] = '\0';
    assert(!nn_activity_open(bad, error, sizeof(error)) && error[0]);
}

static void test_application(const char *dir)
{
    char cwd[4096], core[4200], path[512], error[512] = "", parent[512];
    assert(getcwd(cwd, sizeof(cwd)));
    snprintf(core, sizeof(core), "%s/stereotype-packages/core", cwd);
    snprintf(path, sizeof(path), "%s/app.log", dir);
    snprintf(parent, sizeof(parent), "%s/projects", dir);
    assert(mkdir(parent, 0700) == 0);
    NNActivityLog *log = nn_activity_open(path, error, sizeof(error));
    assert(log);
    NNApplication *app = nn_app_new(core);
    assert(app);
    nn_app_set_activity_log(app, log);
    assert(nn_app_create(app, parent, "demo", "Demo", "0.1.0", false, error, sizeof(error)));
    assert(nn_app_add_node(app, "lin", "core.linear", "0.1.0", "", 10, 20, error, sizeof(error)));
    assert(nn_app_connect(app, "e1", "lin", "out", "output", "in", error, sizeof(error)));
    assert(nn_app_set_parameter_text(app, "lin", "in_features", "8", error, sizeof(error)));
    assert(nn_app_move_node(app, "lin", 40, 60, error, sizeof(error)));
    /* Failed operation: logged with its error, returned text unchanged. */
    assert(!nn_app_remove_node(app, "ghost", error, sizeof(error)) && !strcmp(error, "node not found"));
    assert(!nn_app_remove_node(app, "ghost", NULL, 0));
    assert(nn_app_analysis(app, error, sizeof(error)));
    assert(nn_app_analysis(app, error, sizeof(error)));
    assert(nn_app_undo(app, error, sizeof(error)));
    /* Edit group: snapshots are deferred to end_edit. */
    assert(nn_app_begin_edit(app, error, sizeof(error)));
    assert(nn_app_add_node(app, "relu", "core.relu", "0.1.0", "", 0, 0, error, sizeof(error)));
    assert(nn_app_rename_node(app, "relu", "R", error, sizeof(error)));
    assert(nn_app_end_edit(app, true, error, sizeof(error)));
    assert(nn_app_begin_edit(app, error, sizeof(error)));
    assert(nn_app_remove_node(app, "relu", error, sizeof(error)));
    assert(nn_app_end_edit(app, false, error, sizeof(error)));
    nn_app_set_activity_source(app, "ipc");
    assert(nn_app_disconnect(app, "e1", error, sizeof(error)));
    nn_app_set_activity_source(app, "core");
    assert(nn_app_save(app, error, sizeof(error)));
    nn_app_set_activity_log(app, NULL);
    assert(nn_app_add_node(app, "silent", "core.relu", "0.1.0", "", 0, 0, error, sizeof(error)));
    nn_app_free(app);
    nn_activity_close(log);

    Lines lines = read_lines(path);
    size_t i = 0;
    assert(!strcmp(event_of(&lines, i), "project.create"));
    yyjson_val *data = data_of(&lines, i);
    assert(!strcmp(text_of(data, "id"), "demo") && !strcmp(text_of(data, "template"), "blank"));
    assert(!strcmp(event_of(&lines, ++i), "model.snapshot"));
    data = data_of(&lines, i);
    assert(!strcmp(text_of(data, "project"), "demo") && !yyjson_get_bool(yyjson_obj_get(data, "dirty")));
    assert(yyjson_arr_size(yyjson_obj_get(data, "nodes")) == 2);
    yyjson_val *node = yyjson_arr_get(yyjson_obj_get(data, "nodes"), 0);
    assert(yyjson_obj_get(node, "params") && !yyjson_obj_get(node, "x") && !yyjson_obj_get(node, "position"));
    assert(yyjson_is_null(yyjson_obj_get(data, "active_dataset")) || yyjson_is_obj(yyjson_obj_get(data, "active_dataset")));
    assert(!strcmp(event_of(&lines, ++i), "node.add"));
    assert(!strcmp(text_of(rec(&lines, i), "src"), "core"));
    data = data_of(&lines, i);
    assert(!strcmp(text_of(data, "id"), "lin") && !strcmp(text_of(data, "package"), "core.linear") &&
           yyjson_get_num(yyjson_obj_get(data, "x")) == 10);
    assert(!strcmp(event_of(&lines, ++i), "model.snapshot"));
    assert(!strcmp(event_of(&lines, ++i), "edge.connect"));
    assert(!strcmp(text_of(data_of(&lines, i), "target_handle"), "in"));
    assert(!strcmp(event_of(&lines, ++i), "model.snapshot"));
    assert(yyjson_arr_size(yyjson_obj_get(data_of(&lines, i), "edges")) == 1);
    assert(!strcmp(event_of(&lines, ++i), "param.set"));
    data = data_of(&lines, i);
    assert(!strcmp(text_of(data, "node"), "lin") && !strcmp(text_of(data, "key"), "in_features") &&
           !strcmp(text_of(data, "new"), "8") && yyjson_obj_get(data, "old"));
    assert(!strcmp(event_of(&lines, ++i), "model.snapshot"));
    assert(!strcmp(event_of(&lines, ++i), "node.move"));
    assert(!strcmp(event_of(&lines, ++i), "node.remove"));
    yyjson_val *failed = rec(&lines, i);
    assert(!yyjson_get_bool(yyjson_obj_get(failed, "ok")) && !strcmp(text_of(failed, "error"), "node not found"));
    assert(!yyjson_obj_get(data_of(&lines, i), "package"));
    assert(!strcmp(event_of(&lines, ++i), "node.remove")); /* NULL error buffer still logged */
    assert(!strcmp(text_of(rec(&lines, i), "error"), "node not found"));
    assert(!strcmp(event_of(&lines, ++i), "analysis.result"));
    assert(yyjson_is_uint(yyjson_obj_get(data_of(&lines, i), "problems")));
    assert(yyjson_obj_get(data_of(&lines, i), "root_status") && yyjson_obj_get(data_of(&lines, i), "nodes"));
    assert(!strcmp(event_of(&lines, ++i), "edit.undo"));
    assert(!strcmp(event_of(&lines, ++i), "model.snapshot"));
    assert(!strcmp(event_of(&lines, ++i), "edit.begin"));
    assert(!strcmp(event_of(&lines, ++i), "node.add")); /* deferred: no snapshot yet */
    assert(!strcmp(event_of(&lines, ++i), "node.rename"));
    assert(!strcmp(event_of(&lines, ++i), "edit.end"));
    assert(yyjson_get_bool(yyjson_obj_get(data_of(&lines, i), "commit")));
    assert(!strcmp(event_of(&lines, ++i), "model.snapshot"));
    assert(!strcmp(event_of(&lines, ++i), "edit.begin"));
    assert(!strcmp(event_of(&lines, ++i), "node.remove"));
    assert(!strcmp(text_of(data_of(&lines, i), "package"), "core.relu"));
    assert(!strcmp(event_of(&lines, ++i), "edit.end"));
    assert(!yyjson_get_bool(yyjson_obj_get(data_of(&lines, i), "commit")));
    assert(!strcmp(event_of(&lines, ++i), "model.snapshot"));
    assert(!strcmp(event_of(&lines, ++i), "edge.disconnect"));
    assert(!strcmp(text_of(rec(&lines, i), "src"), "ipc"));
    assert(!strcmp(event_of(&lines, ++i), "model.snapshot"));
    assert(!strcmp(text_of(rec(&lines, i), "src"), "ipc"));
    assert(!strcmp(event_of(&lines, ++i), "project.save"));
    assert(++i == lines.count); /* nothing after save, nothing after log detach */
    /* seq is strictly increasing from 1 within the session. */
    for (size_t n = 0; n < lines.count; ++n)
        assert(yyjson_get_uint(yyjson_obj_get(rec(&lines, n), "seq")) == n + 1);
    (void)find_event;
    free_lines(&lines);
}

int main(void)
{
    const char *base = getenv("TMPDIR");
    char temporary[512];
    snprintf(temporary, sizeof(temporary), "%s/nn-activity-test-XXXXXX", base && *base ? base : "/tmp");
    char *dir = mkdtemp(temporary);
    assert(dir);
    test_paths();
    test_redaction();
    test_format(dir);
    test_application(dir);
    return 0;
}
