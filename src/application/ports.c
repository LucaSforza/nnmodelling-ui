#include "application_internal.h"
#include "application/application.h"
#include "utils/utils.h"
#include "inference/inference.h"
#include "yyjson.h"

#include <math.h>
#include <stdint.h>
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

bool nn_app_valid_output_handle(const NNPackage *package, const char *handle)
{
    if (!package || !handle) return false;
    for (size_t i = 0; i < package->output_count; ++i)
        if (!strcmp(package->outputs[i].id, handle)) return true;
    return false;
}

bool nn_app_valid_input_handle(const NNPackage *package, const char *handle)
{
    if (nn_app_kind_is(package, "input")) return false;
    if (nn_app_kind_is(package, "join")) {
        size_t suffix = 0;
        return nn_join_handle_order(handle, &suffix);
    }
    return handle && !strcmp(handle, "in");
}
typedef struct { const char *id; size_t number; bool numeric, owned; } JoinHandle;

static int compare_join_handle(const void *left, const void *right)
{
    const JoinHandle *a = left, *b = right;
    if (a->numeric != b->numeric) return a->numeric ? -1 : 1;
    if (a->numeric && a->number != b->number) return a->number < b->number ? -1 : 1;
    return strcmp(a->id, b->id);
}

static bool join_number_used(const JoinHandle *handles, size_t count, size_t number)
{
    for (size_t i = 0; i < count; ++i)
        if (handles[i].numeric && handles[i].number == number) return true;
    return false;
}

static void join_handles_free(JoinHandle *handles, size_t count)
{
    if (!handles) return;
    for (size_t i = 0; i < count; ++i)
        if (handles[i].owned) free((void *)handles[i].id);
    free(handles);
}

static JoinHandle *join_inputs(const NNModel *model, const char *node_id, size_t *count)
{
    size_t edge_count = nn_model_edge_count(model);
    if (edge_count > (SIZE_MAX / sizeof(JoinHandle)) - 2) return NULL;
    JoinHandle *handles = calloc(edge_count + 2, sizeof(*handles));
    if (!handles) return NULL;
    size_t used = 0;
    for (size_t i = 0; i < edge_count; ++i) {
        const NNEdge *edge = nn_model_edge_at(model, i);
        size_t suffix = 0;
        if (strcmp(edge->target_id, node_id)) continue;
        bool numeric = nn_join_handle_order(edge->target_handle_id, &suffix);
        bool duplicate = false;
        for (size_t j = 0; j < used; ++j)
            if (!strcmp(handles[j].id, edge->target_handle_id)) duplicate = true;
        if (!duplicate) handles[used++] = (JoinHandle){ edge->target_handle_id, suffix, numeric, false };
    }
    size_t existing = used;
    size_t candidate = 1;
    size_t desired = existing < 2 ? 2 : existing + 1;
    while (used < desired) {
        while (join_number_used(handles, existing, candidate)) {
            if (candidate == SIZE_MAX) { join_handles_free(handles, used); return NULL; }
            ++candidate;
        }
        char *generated = malloc(3 + 3 * sizeof(size_t) + 1);
        if (!generated) { join_handles_free(handles, used); return NULL; }
        snprintf(generated, 3 + 3 * sizeof(size_t) + 1, "in-%zu", candidate);
        handles[used++] = (JoinHandle){ generated, candidate, true, true };
        if (candidate == SIZE_MAX) { join_handles_free(handles, used); return NULL; }
        ++candidate;
    }
    qsort(handles, used, sizeof(*handles), compare_join_handle);
    *count = used;
    return handles;
}

size_t nn_app_port_count(const NNApplication *app, const char *node_id, bool output)
{
    const NNModel *model = nn_app_model(app);
    const NNNode *node = model ? nn_model_find_node(model, node_id) : NULL;
    const NNPackage *package = nn_app_find_package(app, node);
    if (!package) return 0;
    if (output) return package->output_count;
    if (nn_app_kind_is(package, "input")) return 0;
    if (nn_app_kind_is(package, "join")) {
        size_t count = 0;
        JoinHandle *inputs = join_inputs(model, node_id, &count);
        join_handles_free(inputs, count);
        return count;
    }
    return 1;
}

bool nn_app_port_id(const NNApplication *app, const char *node_id, bool output,
                    size_t index, char *buffer, size_t capacity)
{
    const NNModel *model = nn_app_model(app);
    const NNNode *node = model ? nn_model_find_node(model, node_id) : NULL;
    const NNPackage *package = nn_app_find_package(app, node);
    if (!package || !buffer || !capacity) return false;
    if (output) {
        if (index >= package->output_count) return false;
        int written = snprintf(buffer, capacity, "%s", package->outputs[index].id);
        return written >= 0 && (size_t)written < capacity;
    }
    if (nn_app_kind_is(package, "input")) return false;
    if (!nn_app_kind_is(package, "join")) {
        if (index != 0) return false;
        return snprintf(buffer, capacity, "in") < (int)capacity;
    }
    size_t count = 0;
    JoinHandle *inputs = join_inputs(model, node_id, &count);
    if (!inputs || index >= count) { join_handles_free(inputs, count); return false; }
    int written = snprintf(buffer, capacity, "%s", inputs[index].id);
    join_handles_free(inputs, count);
    return written >= 0 && (size_t)written < capacity;
}

bool nn_app_node_is_subflow(const NNApplication *app, const char *node_id)
{
    const NNModel *model = nn_app_model(app);
    const NNNode *node = model ? nn_model_find_node(model, node_id) : NULL;
    return nn_app_kind_is(nn_app_find_package(app, node), "subflow");
}

const char *nn_app_output_type(const NNApplication *app, const char *node_id,
                               const char *handle_id)
{
    const NNModel *model = nn_app_model(app);
    const NNNode *node = model ? nn_model_find_node(model, node_id) : NULL;
    const NNPackage *package = nn_app_find_package(app, node);
    if (!package || !handle_id) return NULL;
    for (size_t i = 0; i < package->output_count; ++i)
        if (!strcmp(package->outputs[i].id, handle_id)) return package->outputs[i].type;
    return NULL;
}
