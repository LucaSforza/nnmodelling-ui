#include "application.h"
#include "automation.h"
#include "inference.h"
#include "yyjson.h"

#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void *__real_malloc(size_t size);
void *__real_calloc(size_t count, size_t size);
void *__real_realloc(void *pointer, size_t size);

static bool tracking;
static size_t allocation_count;
static size_t fail_index;
static bool failure_observed;

static bool should_fail(void)
{
    if (!tracking) return false;
    ++allocation_count;
    if (fail_index && allocation_count == fail_index) {
        /* Disarm immediately so cleanup can allocate if necessary. */
        tracking = false;
        fail_index = 0;
        failure_observed = true;
        return true;
    }
    return false;
}

void *__wrap_malloc(size_t size)
{
    return should_fail() ? NULL : __real_malloc(size);
}

void *__wrap_calloc(size_t count, size_t size)
{
    return should_fail() ? NULL : __real_calloc(count, size);
}

void *__wrap_realloc(void *pointer, size_t size)
{
    return should_fail() ? NULL : __real_realloc(pointer, size);
}

static void count_begin(void)
{
    allocation_count = 0;
    failure_observed = false;
    fail_index = 0;
    tracking = true;
}

static size_t count_end(void)
{
    tracking = false;
    return allocation_count;
}

static void fail_begin(size_t index)
{
    allocation_count = 0;
    failure_observed = false;
    fail_index = index;
    tracking = true;
}

static void assert_complete_report(const NNInferenceReport *report,
                                   size_t expected_count)
{
    assert(report && nn_inference_count(report) == expected_count);
    for (size_t i = 0; i < expected_count; ++i) {
        const NNInferenceResult *result = nn_inference_at(report, i);
        assert(result && result->node_id);
        if (result->status == NN_INFERENCE_SUCCESS) {
            assert(result->dtype && result->dtype[0]);
            assert(result->dimension_count > 0 && result->dimensions);
            for (size_t d = 0; d < result->dimension_count; ++d)
                assert(result->dimensions[d] && result->dimensions[d][0]);
        }
    }
}

static void make_analysis_cold(NNApplication *app)
{
    char error[512] = "";
    /* Even the unchanged binding goes through the semantic mutation path, which
       invalidates the cached report and keeps every sweep iteration identical. */
    assert(nn_app_set_parameter_text(app, "image", "binding", "image",
                                     error, sizeof(error)));
    assert(error[0] == '\0');
}

static void sweep_analysis(NNApplication *app)
{
    make_analysis_cold(app);
    count_begin();
    const NNInferenceReport *baseline = nn_app_analysis(app, NULL, 0);
    size_t count = count_end();
    size_t nodes = nn_model_node_count(nn_app_model(app));
    assert_complete_report(baseline, nodes);
    assert(count > 0);
    for (size_t i = 0; i < nodes; ++i)
        assert(nn_inference_at(baseline, i)->status == NN_INFERENCE_SUCCESS);
    printf("successful VAE inference baseline allocations: %zu\n", count);

    size_t null_reports = 0, complete_reports = 0;
    for (size_t i = 1; i <= count; ++i) {
        make_analysis_cold(app);
        char error[512] = "";
        fail_begin(i);
        const NNInferenceReport *report = nn_app_analysis(app, error, sizeof(error));
        (void)count_end();
        assert(failure_observed);
        if (!report) {
            ++null_reports;
            assert(error[0]);
        } else {
            ++complete_reports;
            /* Lua allocator faults may produce a complete fault diagnostic. */
            assert_complete_report(report, nodes);
        }
    }
    printf("analysis allocation fail-index sweep: %zu wrapped calls, %zu NULL, %zu reports\n",
           count, null_reports, complete_reports);
}

static const NNNode *first_node(const NNModel *model)
{
    assert(model && nn_model_node_count(model));
    return nn_model_node_at(model, 0);
}

static void prepare_dirty_mlp(NNApplication *app)
{
    char error[512] = "";
    assert(nn_app_open(app, "examples/mnist-mlp", error, sizeof(error)));
    const NNNode *node = first_node(nn_app_model(app));
    assert(nn_app_move_node(app, node->id, node->x + 1, node->y,
                            error, sizeof(error)));
    assert(nn_project_dirty(nn_app_project(app)));
    const NNInferenceReport *report = nn_app_analysis(app, error, sizeof(error));
    assert(report);
}

static void sweep_project_open(NNApplication *app)
{
    char error[512] = "";
    prepare_dirty_mlp(app);
    count_begin();
    assert(nn_app_open(app, "examples/mnist-vae", error, sizeof(error)));
    size_t count = count_end();
    assert(count > 0);

    size_t failed = 0, succeeded = 0;
    for (size_t i = 1; i <= count; ++i) {
        prepare_dirty_mlp(app);
        const NNProject *old_project = nn_app_project(app);
        const NNModel *old_model = nn_app_model(app);
        const NNInferenceReport *old_report = nn_app_analysis(app, error, sizeof(error));
        assert(old_project && old_model && old_report && nn_project_dirty(old_project));
        size_t old_nodes = nn_model_node_count(old_model);
        error[0] = '\0';

        fail_begin(i);
        bool okay = nn_app_open(app, "examples/mnist-vae", error, sizeof(error));
        (void)count_end();
        assert(failure_observed);
        if (!okay) {
            ++failed;
            assert(error[0]);
            assert(nn_app_project(app) == old_project);
            assert(nn_app_model(app) == old_model);
            assert(nn_project_dirty(nn_app_project(app)));
            assert(nn_model_node_count(nn_app_model(app)) == old_nodes);
            assert(nn_app_analysis(app, NULL, 0) == old_report);
        } else {
            ++succeeded;
            assert(!nn_project_dirty(nn_app_project(app)));
            assert(!strcmp(nn_project_id(nn_app_project(app)), "mnist-vae"));
            const NNInferenceReport *report = nn_app_analysis(app, error, sizeof(error));
            assert_complete_report(report, nn_model_node_count(nn_app_model(app)));
        }
    }
    printf("project-open allocation fail-index sweep: %zu wrapped calls, %zu failed, %zu succeeded\n",
           count, failed, succeeded);
}

static void sweep_add_node(NNApplication *app)
{
    char error[512] = "";
    assert(nn_app_open(app, "examples/mnist-vae", error, sizeof(error)));
    count_begin();
    assert(nn_app_add_node(app, "baseline-allocation-probe", "core.relu", "0.1.0",
                           "", 10, 10, error, sizeof(error)));
    size_t count = count_end();
    assert(count > 0);

    size_t failed = 0, succeeded = 0;
    for (size_t i = 1; i <= count; ++i) {
        assert(nn_app_open(app, "examples/mnist-vae", error, sizeof(error)));
        const NNModel *old_model = nn_app_model(app);
        const NNInferenceReport *old_report = nn_app_analysis(app, error, sizeof(error));
        assert(old_model && old_report && !nn_project_dirty(nn_app_project(app)));
        size_t old_nodes = nn_model_node_count(old_model);
        error[0] = '\0';

        fail_begin(i);
        bool okay = nn_app_add_node(app, "allocation-probe", "core.relu", "0.1.0",
                                    "", 10, 10, error, sizeof(error));
        (void)count_end();
        assert(failure_observed);
        if (!okay) {
            ++failed;
            assert(error[0]);
            assert(nn_app_model(app) == old_model);
            assert(nn_model_node_count(nn_app_model(app)) == old_nodes);
            assert(!nn_project_dirty(nn_app_project(app)));
            assert(nn_app_analysis(app, NULL, 0) == old_report);
        } else {
            ++succeeded;
            assert(error[0] == '\0');
            assert(nn_model_node_count(nn_app_model(app)) == old_nodes + 1);
            assert(nn_model_find_node(nn_app_model(app), "allocation-probe"));
            assert(nn_project_dirty(nn_app_project(app)));
            const NNInferenceReport *report = nn_app_analysis(app, error, sizeof(error));
            assert_complete_report(report, nn_model_node_count(nn_app_model(app)));
        }
    }
    printf("add-node allocation fail-index sweep: %zu wrapped calls, %zu failed, %zu succeeded\n",
           count, failed, succeeded);
}

static void assert_valid_response(const char *response, const char *operation)
{
    assert(response);
    yyjson_doc *doc = yyjson_read(response, strlen(response), 0);
    assert(doc);
    yyjson_val *root = yyjson_doc_get_root(doc);
    yyjson_val *ok = yyjson_obj_get(root, "ok");
    assert(yyjson_is_obj(root) && yyjson_is_bool(ok));
    if (yyjson_get_bool(ok)) {
        yyjson_val *result = yyjson_obj_get(root, "result");
        assert(result);
        if (!strcmp(operation, "analysis.diagnostics")) {
            assert(yyjson_is_bool(yyjson_obj_get(result, "available")));
            assert(yyjson_is_bool(yyjson_obj_get(result, "complete")));
            assert(yyjson_is_arr(yyjson_obj_get(result, "problems")));
            assert(yyjson_is_arr(yyjson_obj_get(result, "tensors")));
        } else assert(yyjson_is_obj(result));
    } else assert(yyjson_is_str(yyjson_obj_get(root, "error")));
    yyjson_doc_free(doc);
}

static size_t serializer_baseline(NNApplication *app, const char *operation)
{
    char request[128];
    snprintf(request, sizeof(request), "{\"operation\":\"%s\",\"args\":{}}", operation);
    count_begin();
    char *response = nn_automation_dispatch(app, request, NULL, NULL);
    size_t count = count_end();
    assert_valid_response(response, operation);
    free(response);
    assert(count > 0);
    return count;
}

static void sweep_serializer(NNApplication *app, const char *operation)
{
    char request[128];
    snprintf(request, sizeof(request), "{\"operation\":\"%s\",\"args\":{}}", operation);
    size_t count = serializer_baseline(app, operation);
    const NNProject *project = nn_app_project(app);
    const NNModel *model = nn_app_model(app);
    size_t nodes = nn_model_node_count(model);
    bool dirty = nn_project_dirty(project);
    size_t failures = 0, valid_responses = 0;
    for (size_t i = 1; i <= count; ++i) {
        fail_begin(i);
        char *response = nn_automation_dispatch(app, request, NULL, NULL);
        (void)count_end();
        assert(failure_observed);
        if (response) {
            ++valid_responses;
            assert_valid_response(response, operation);
            free(response);
        } else ++failures;
    }
    assert(nn_app_project(app) == project && nn_app_model(app) == model);
    assert(nn_model_node_count(nn_app_model(app)) == nodes);
    assert(nn_project_dirty(nn_app_project(app)) == dirty);
    printf("%s allocation fail-index sweep: %zu wrapped calls, %zu valid responses, %zu OOM returns\n",
           operation, count, valid_responses, failures);
}

int main(void)
{
    NNApplication *app = nn_app_new("stereotype-packages/core");
    assert(app);
    char error[512] = "";
    assert(nn_app_open(app, "examples/mnist-vae", error, sizeof(error)));

    sweep_analysis(app);
    sweep_project_open(app);
    sweep_add_node(app);

    assert(nn_app_open(app, "examples/mnist-vae", error, sizeof(error)));
    const NNInferenceReport *report = nn_app_analysis(app, error, sizeof(error));
    assert_complete_report(report, nn_model_node_count(nn_app_model(app)));
    bool dirty = nn_project_dirty(nn_app_project(app));
    sweep_serializer(app, "analysis.diagnostics");
    sweep_serializer(app, "project.snapshot");
    assert(nn_project_dirty(nn_app_project(app)) == dirty);

    nn_app_free(app);
    return 0;
}
