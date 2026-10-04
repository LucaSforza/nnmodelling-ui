#define _XOPEN_SOURCE 700
#include "application/application.h"
#include "automation/automation.h"
#include "inference/inference.h"
#include "yyjson.h"

#include <assert.h>
#include <dirent.h>
#include <ftw.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

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
            assert(result->dimensions);
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
    for (size_t i = 0; i < nodes; ++i) {
        const NNInferenceResult *result = nn_inference_at(baseline, i);
        assert(result->status == NN_INFERENCE_SUCCESS);
    }
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

static const NNNode *find_mapped_terminal(const NNModel *model, const char *owner,
                                         const char *handle)
{
    for (size_t i = 0; i < nn_model_node_count(model); ++i) {
        const NNNode *node = nn_model_node_at(model, i);
        if (node->scope_id && !strcmp(node->scope_id, owner) &&
            node->boundary_handle_id && !strcmp(node->boundary_handle_id, handle))
            return node;
    }
    return NULL;
}

static const NNNode *find_scoped_package(const NNModel *model, const char *owner,
                                         const char *package_id)
{
    for (size_t i = 0; i < nn_model_node_count(model); ++i) {
        const NNNode *node = nn_model_node_at(model, i);
        if (!strcmp(node->scope_id, owner) && !strcmp(node->package_id, package_id))
            return node;
    }
    return NULL;
}

static void assert_typed_tensors(const NNInferenceReport *report)
{
    for (size_t i = 0; i < nn_inference_count(report); ++i) {
        const NNInferenceResult *result = nn_inference_at(report, i);
        assert(result && result->node_id);
        if (result->status != NN_INFERENCE_SUCCESS) {
            assert(result->output_count == 0);
            continue;
        }
        if (!result->output_count) continue; /* Consumed terminal tensor. */
        assert(result->outputs);
        for (size_t j = 0; j < result->output_count; ++j) {
            const NNInferenceTensor *output = &result->outputs[j];
            assert(output->handle_id && output->handle_id[0]);
            assert(output->type && output->type[0]);
            assert(output->dtype && output->dtype[0]);
            assert(output->dimensions);
            for (size_t d = 0; d < output->dimension_count; ++d)
                assert(output->dimensions[d] && output->dimensions[d][0]);
        }
    }
}

static int remove_temp_path(const char *path, const struct stat *info,
                            int type, struct FTW *state)
{
    (void)info;
    (void)state;
    return type == FTW_DP ? rmdir(path) : unlink(path);
}

static void sweep_multi_output_analysis(void)
{
    char temporary[] = "/tmp/opencode/nn-multi-inference-XXXXXX";
    char *parent = mkdtemp(temporary);
    assert(parent);
    NNApplication *app = nn_app_new("stereotype-packages/core");
    assert(app);
    char error[512] = "";
    assert(nn_app_create(app, parent, "typed-output-sweep", "Typed Output Sweep",
                         false, error, sizeof(error)));
    assert(nn_app_create_dataset(app, "local.sweep-data", "0.1.0",
        "{\"name\":\"Sweep data\",\"batch\":{\"inputs\":{"
        "\"image\":{\"dtype\":\"float32\",\"shape\":[\"B\",8]}},"
        "\"targets\":{}}}", true, error, sizeof(error)));
    assert(nn_app_create_stereotype(app, "local.sweep-two", "0.1.0",
        "{\"name\":\"Two output sweep\",\"kind\":\"layer\","
        "\"outputs\":[{\"id\":\"objective\",\"type\":\"loss\"},"
        "{\"id\":\"prediction\",\"type\":\"output\"}],"
        "\"view\":{\"color\":\"#444444\",\"width\":200,\"height\":100},"
        "\"parameters\":{}}",
        "return function() return {status='success', outputs={"
        "objective=tensor.create({'B',2},'float32'),"
        "prediction=tensor.create({'B',3},'float32')}} end",
        "{}", error, sizeof(error)));
    assert(nn_app_create_stereotype(app, "local.sweep-subflow", "0.1.0",
        "{\"name\":\"Two output subflow\",\"kind\":\"subflow\","
        "\"outputs\":[{\"id\":\"objective\",\"type\":\"loss\"},"
        "{\"id\":\"prediction\",\"type\":\"output\"}],"
        "\"view\":{\"color\":\"#444444\",\"width\":200,\"height\":100},"
        "\"parameters\":{}}",
        "return function(context, parameters, services) "
        "return services.infer_subflow(context.inputs[1]) end",
        "{}", error, sizeof(error)));
    assert(nn_app_add_node(app, "input", "core.input", "0.1.0", "", 0, 0,
                           error, sizeof(error)));
    assert(nn_app_set_parameter_text(app, "input", "binding", "image",
                                     error, sizeof(error)));
    assert(nn_app_add_node(app, "owner", "local.sweep-subflow", "0.1.0", "", 0, 0,
                           error, sizeof(error)));
    assert(nn_app_connect(app, "root-input-owner", "input", "out", "owner", "in",
                          error, sizeof(error)));
    const NNNode *inner_input = find_scoped_package(nn_app_model(app), "owner", "core.input");
    assert(inner_input);
    size_t immediate_children = 0, immediate_inputs = 0;
    for (size_t i = 0; i < nn_model_node_count(nn_app_model(app)); ++i) {
        const NNNode *node = nn_model_node_at(nn_app_model(app), i);
        if (strcmp(node->scope_id, "owner")) continue;
        ++immediate_children;
        if (!strcmp(node->package_id, "core.input")) ++immediate_inputs;
    }
    assert(immediate_children == 3 && immediate_inputs == 1);
    char inner_input_id[256];
    snprintf(inner_input_id, sizeof(inner_input_id), "%s", inner_input->id);
    assert(nn_app_add_node(app, "inner-two", "local.sweep-two", "0.1.0", "owner", 0, 0,
                           error, sizeof(error)));
    assert(nn_app_connect(app, "inner-input-two", inner_input_id, "out", "inner-two", "in",
                          error, sizeof(error)));
    const NNNode *objective = find_mapped_terminal(nn_app_model(app), "owner", "objective");
    const NNNode *prediction = find_mapped_terminal(nn_app_model(app), "owner", "prediction");
    assert(objective && prediction);
    assert(objective->x == 0 && objective->y == 240);
    assert(prediction->x == 0 && prediction->y == 360);
    char objective_id[128], prediction_id[128];
    snprintf(objective_id, sizeof(objective_id), "%s", objective->id);
    snprintf(prediction_id, sizeof(prediction_id), "%s", prediction->id);
    assert(nn_app_connect(app, "objective-terminal", "inner-two", "objective",
                          objective_id, "in", error, sizeof(error)));
    assert(nn_app_connect(app, "prediction-terminal", "inner-two", "prediction",
                          prediction_id, "in", error, sizeof(error)));

    const NNInferenceReport *baseline = nn_app_analysis(app, error, sizeof(error));
    assert(baseline);
    assert_typed_tensors(baseline);
    const NNInferenceResult *owner_result = NULL;
    for (size_t i = 0; i < nn_inference_count(baseline); ++i) {
        const NNInferenceResult *result = nn_inference_at(baseline, i);
        if (!strcmp(result->node_id, "owner")) owner_result = result;
    }
    assert(owner_result && owner_result->status == NN_INFERENCE_SUCCESS &&
           owner_result->output_count == 2);

    assert(nn_app_set_parameter_text(app, "input", "binding", "image",
                                     error, sizeof(error)));
    count_begin();
    baseline = nn_app_analysis(app, error, sizeof(error));
    size_t allocations = count_end();
    assert(baseline && allocations > 0);
    assert_typed_tensors(baseline);
    size_t null_reports = 0, full_reports = 0;
    for (size_t i = 1; i <= allocations; ++i) {
        assert(nn_app_set_parameter_text(app, "input", "binding", "image",
                                         error, sizeof(error)));
        error[0] = '\0';
        fail_begin(i);
        const NNInferenceReport *report = nn_app_analysis(app, error, sizeof(error));
        (void)count_end();
        assert(failure_observed);
        if (!report) {
            ++null_reports;
            assert(error[0]);
        } else {
            ++full_reports;
            assert_typed_tensors(report);
        }
    }
    printf("multi-output recursive inference allocation sweep: %zu calls, %zu NULL, %zu reports\n",
           allocations, null_reports, full_reports);
    nn_app_free(app);
    assert(nftw(parent, remove_temp_path, 32, FTW_DEPTH | FTW_PHYS) == 0);
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
    assert(nn_app_move_node(app, node->id, node->x + NN_MODEL_GRID_SPACING, node->y,
                            error, sizeof(error)));
    assert(nn_project_dirty(nn_app_project(app)));
    const NNInferenceReport *report = nn_app_analysis(app, error, sizeof(error));
    assert(report);
}

static void sweep_history_snapshot(NNApplication *app)
{
    char error[512] = "";
    assert(nn_app_open(app, "examples/mnist-mlp", error, sizeof(error)));
    const NNNode *node = first_node(nn_app_model(app));
    char id[256];
    assert(strlen(node->id) < sizeof(id));
    strcpy(id, node->id);
    const int32_t x = node->x, y = node->y;
    assert(nn_app_move_node(app, id, x + NN_MODEL_GRID_SPACING, y, error, sizeof(error)));
    assert(nn_app_undo(app, error, sizeof(error)));
    NNModel *baseline = nn_model_copy(nn_app_model(app));
    assert(baseline);
    const NNInferenceReport *report = nn_app_analysis(app, error, sizeof(error));
    assert(report);
    count_begin();
    assert(nn_app_move_node(app, id, x + 2 * NN_MODEL_GRID_SPACING, y, error, sizeof(error)));
    const size_t count = count_end();
    assert(count > 0);
    assert(nn_app_undo(app, error, sizeof(error)));
    report = nn_app_analysis(app, error, sizeof(error));
    for (size_t i = 1; i <= count; ++i) {
        fail_begin(i);
        const bool okay = nn_app_move_node(app, id, x + 2 * NN_MODEL_GRID_SPACING, y,
                                           error, sizeof(error));
        (void)count_end();
        assert(failure_observed && !okay && error[0]);
        assert(nn_model_equal(baseline, nn_app_model(app)));
        assert(!nn_app_can_undo(app) && nn_app_can_redo(app));
        assert(!nn_project_dirty(nn_app_project(app)));
        assert(nn_app_analysis(app, NULL, 0) == report);
    }
    nn_model_free(baseline);
    printf("history snapshot allocation fail-index sweep: %zu failures preserved graph and redo\n", count);
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

static void sweep_add_subflow(NNApplication *app)
{
    char error[512] = "";
    assert(nn_app_open(app, "examples/mnist-vae", error, sizeof(error)));
    count_begin();
    assert(nn_app_add_node(app, "spawn-probe", "core.repeat", "0.1.0", "",
                           100, 100, error, sizeof(error)));
    size_t count = count_end();
    assert(count > 0);

    size_t failed = 0, succeeded = 0;
    for (size_t i = 1; i <= count; ++i) {
        assert(nn_app_open(app, "examples/mnist-vae", error, sizeof(error)));
        const NNModel *old_model = nn_app_model(app);
        const NNInferenceReport *old_report = nn_app_analysis(app, error, sizeof(error));
        size_t old_nodes = nn_model_node_count(old_model);
        assert(old_model && old_report && !nn_project_dirty(nn_app_project(app)));

        error[0] = '\0';
        fail_begin(i);
        bool okay = nn_app_add_node(app, "spawn-probe", "core.repeat", "0.1.0", "",
                                    100, 100, error, sizeof(error));
        (void)count_end();
        assert(failure_observed);
        if (!okay) {
            ++failed;
            assert(error[0]);
            assert(nn_app_model(app) == old_model);
            assert(nn_model_node_count(nn_app_model(app)) == old_nodes);
            assert(!nn_model_find_node(nn_app_model(app), "spawn-probe"));
            assert(!nn_model_find_node(nn_app_model(app), "spawn-probe-input"));
            assert(!nn_model_find_node(nn_app_model(app), "spawn-probe-boundary-out"));
            assert(!nn_project_dirty(nn_app_project(app)));
            assert(nn_app_analysis(app, NULL, 0) == old_report);
        } else {
            ++succeeded;
            assert(error[0] == '\0');
            assert(nn_model_node_count(nn_app_model(app)) == old_nodes + 3);
            const NNNode *input = nn_model_find_node(nn_app_model(app), "spawn-probe-input");
            const NNNode *terminal = nn_model_find_node(nn_app_model(app),
                                                        "spawn-probe-boundary-out");
            assert(nn_model_find_node(nn_app_model(app), "spawn-probe"));
            assert(input && !strcmp(input->package_id, "core.input") &&
                   !strcmp(input->scope_id, "spawn-probe") && input->x == 0 && input->y == 0);
            assert(terminal && terminal->x == 0 && terminal->y == 240);
            assert(nn_project_dirty(nn_app_project(app)));
        }
    }
    printf("subflow spawn allocation fail-index sweep: %zu wrapped calls, %zu failed, %zu succeeded\n",
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
    sweep_add_subflow(app);
    sweep_history_snapshot(app);

    assert(nn_app_open(app, "examples/mnist-vae", error, sizeof(error)));
    const NNInferenceReport *report = nn_app_analysis(app, error, sizeof(error));
    assert_complete_report(report, nn_model_node_count(nn_app_model(app)));
    bool dirty = nn_project_dirty(nn_app_project(app));
    sweep_serializer(app, "analysis.diagnostics");
    sweep_serializer(app, "project.snapshot");
    assert(nn_project_dirty(nn_app_project(app)) == dirty);

    nn_app_free(app);
    sweep_multi_output_analysis();
    return 0;
}
