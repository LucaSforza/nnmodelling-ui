#define _XOPEN_SOURCE 700
#include "inference/inference.h"
#include "model/model.h"
#include "project/project.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static const NNInferenceResult *find_result(const NNInferenceReport *report,
                                             const char *id)
{
    for (size_t i = 0; i < nn_inference_count(report); ++i) {
        const NNInferenceResult *result = nn_inference_at(report, i);
        if (!strcmp(result->node_id, id)) return result;
    }
    return NULL;
}

static NNProject *open_example(void)
{
    char error[512] = {0};
    NNProject *project = nn_project_open("examples/mnist-mlp", "stereotype-packages/core",
                                         error, sizeof(error));
    if (!project) fprintf(stderr, "project open failed: %s\n", error);
    return project;
}

static NNProject *open_vae(void)
{
    char error[512] = {0};
    NNProject *project = nn_project_open("examples/mnist-vae", "stereotype-packages/core",
                                         error, sizeof(error));
    if (!project) fprintf(stderr, "VAE project open failed: %s\n", error);
    return project;
}

static void assert_shape(const NNInferenceResult *result, const char *dtype,
                         const char *const *dimensions, size_t count)
{
    assert(result && result->status == NN_INFERENCE_SUCCESS);
    assert(!strcmp(result->dtype, dtype));
    assert(result->dimension_count == count);
    for (size_t i = 0; i < count; ++i) assert(!strcmp(result->dimensions[i], dimensions[i]));
}

int main(void)
{
    assert(!strcmp(nn_inference_category(NN_INFERENCE_COMPILATION_ERROR), "lua-compilation"));
    assert(!strcmp(nn_inference_severity(NN_INFERENCE_COMPILATION_ERROR), "error"));
    assert(!strcmp(nn_inference_category(NN_INFERENCE_UNRESOLVED), "incomplete"));
    assert(!strcmp(nn_inference_severity(NN_INFERENCE_UNRESOLVED), "warning"));
    assert(nn_inference_error_line("[string \"rule\"]:27: unexpected symbol") == 27);
    assert(nn_inference_error_line("no source location") == 0);
    char validation_error[256] = {0};
    assert(nn_inference_validate_source("return function() end", validation_error,
                                        sizeof(validation_error)));
    assert(!nn_inference_validate_source("return 42", validation_error,
                                         sizeof(validation_error)));
    assert(validation_error[0]);
    assert(!nn_inference_validate_source("return function(", validation_error,
                                         sizeof(validation_error)));
    assert(!nn_inference_validate_source("while true do end", validation_error,
                                         sizeof(validation_error)));
    char *oversized_source = malloc(1024 * 1024 + 2);
    assert(oversized_source);
    memset(oversized_source, ' ', 1024 * 1024 + 1);
    oversized_source[1024 * 1024 + 1] = '\0';
    assert(!nn_inference_validate_source(oversized_source, validation_error,
                                         sizeof(validation_error)));
    free(oversized_source);

    char error[256] = {0};
    NNProject *project = open_example();
    assert(project);
    NNInferenceReport *report = nn_infer_project(project);
    assert(report && nn_inference_count(report) == 10);
    const NNInferenceResult *output = find_result(report, "output");
    assert(output && output->status == NN_INFERENCE_SUCCESS);
    assert(!strcmp(output->dtype, "float32"));
    assert(output->dimension_count == 2);
    assert(!strcmp(output->dimensions[0], "B"));
    assert(!strcmp(output->dimensions[1], "10"));
    const NNInferenceResult *objective = find_result(report, "cross-entropy");
    assert(objective && objective->status == NN_INFERENCE_SUCCESS &&
           objective->output_count == 1);
    assert(!strcmp(objective->outputs[0].handle_id, "loss"));
    assert(!strcmp(objective->outputs[0].type, "loss"));
    assert(output->output_count == 0);
    assert(nn_inference_root_status(report) == NN_INFERENCE_SUCCESS);
    assert(!nn_inference_root_message(report));
    nn_inference_free(report);

    assert(nn_model_remove_node(nn_project_model(project), "loss-output", error,
                                sizeof(error)));
    report = nn_infer_project(project);
    assert(report && nn_inference_root_status(report) == NN_INFERENCE_UNRESOLVED);
    assert(nn_inference_root_message(report));
    assert(find_result(report, "output")->status == NN_INFERENCE_SUCCESS);
    nn_inference_free(report);
    while (nn_model_node_count(nn_project_model(project))) {
        const NNNode *node = nn_model_node_at(nn_project_model(project), 0);
        char id[128];
        snprintf(id, sizeof(id), "%s", node->id);
        assert(nn_model_remove_node(nn_project_model(project), id, error, sizeof(error)));
    }
    report = nn_infer_project(project);
    assert(report && nn_inference_count(report) == 0);
    assert(nn_inference_root_status(report) == NN_INFERENCE_UNRESOLVED);
    assert(nn_inference_root_message(report));
    nn_inference_free(report);
    nn_project_close(project);

    project = open_example();
    assert(project);
    NNValue bad_features = { .type = NN_VALUE_INT, .as.integer = 800 };
    assert(nn_model_set_parameter(nn_project_model(project), "dense1", "in_features",
                                  &bad_features, error, sizeof(error)));
    report = nn_infer_project(project);
    assert(report);
    const NNInferenceResult *broken = find_result(report, "dense1");
    assert(broken && broken->status == NN_INFERENCE_SEMANTIC_ERROR);
    assert(find_result(report, "output")->status == NN_INFERENCE_UNRESOLVED);
    nn_inference_free(report);
    nn_project_close(project);

    project = open_example();
    assert(project);
    assert(nn_model_add_node(nn_project_model(project), "second-input", "second input",
                             "core.input", "0.1.0", "", 0, 0,
                             error, sizeof(error)));
    report = nn_infer_project(project);
    assert(report && find_result(report, "output")->status == NN_INFERENCE_SUCCESS);
    assert(find_result(report, "second-input")->status == NN_INFERENCE_UNRESOLVED);
    nn_inference_free(report);
    nn_project_close(project);

    project = open_example();
    assert(project);
    NNValue binding = { .type = NN_VALUE_STRING, .as.string = "missing" };
    assert(nn_model_set_parameter(nn_project_model(project), "input", "binding",
                                  &binding, error, sizeof(error)));
    report = nn_infer_project(project);
    assert(report);
    const NNInferenceResult *input = find_result(report, "input");
    assert(input && input->status == NN_INFERENCE_UNRESOLVED);
    assert(find_result(report, "output")->status == NN_INFERENCE_UNRESOLVED);
    nn_inference_free(report);
    nn_project_close(project);

    project = open_vae();
    assert(project);
    report = nn_infer_project(project);
    assert(report && nn_inference_count(report) == 23);
    const char *packed[] = { "B", "2", "32" };
    const char *latent[] = { "B", "32" };
    const char *pixels[] = { "B", "784" };
    const char *kl_shape[] = { "B" };
    assert_shape(find_result(report, "encoder"), "float32", packed, 3);
    assert_shape(find_result(report, "encoder-input"), "float32", pixels, 2);
    assert_shape(find_result(report, "decoder-input"), "float32", latent, 2);
    assert_shape(find_result(report, "reconstruction"), "float32", pixels, 2);
    assert_shape(find_result(report, "kl"), "float32", kl_shape, 1);
    assert_shape(find_result(report, "mean"), "float32", latent, 2);
    assert_shape(find_result(report, "log-variance"), "float32", latent, 2);
    nn_inference_free(report);

    assert(nn_model_add_node(nn_project_model(project), "inner-proxy", "nested proxy",
                             "core.subflow-proxy", "0.1.0", "encoder", 0, 0,
                             error, sizeof(error)));
    assert(nn_model_add_node(nn_project_model(project), "inner-input", "nested input",
                             "core.input", "0.1.0", "inner-proxy", 0, 0,
                             error, sizeof(error)));
    assert(nn_model_add_node(nn_project_model(project), "inner-output", "nested output",
                             "core.output", "0.1.0", "inner-proxy", 0, 0,
                             error, sizeof(error)));
    assert(nn_model_set_boundary_handle(nn_project_model(project), "inner-output", "out",
                                        error, sizeof(error)));
    NNValue empty_binding = { .type = NN_VALUE_STRING, .as.string = "" };
    assert(nn_model_set_parameter(nn_project_model(project), "inner-input", "binding",
                                  &empty_binding, error, sizeof(error)));
    assert(nn_model_connect(nn_project_model(project), "inner-parent-edge", "encoder-relu", "out",
                            "inner-proxy", "in", error, sizeof(error)));
    assert(nn_model_connect(nn_project_model(project), "inner-edge", "inner-input", "out",
                            "inner-output", "in", error, sizeof(error)));
    report = nn_infer_project(project);
    assert(report && nn_inference_count(report) == 26);
    const char *hidden_shape[] = { "B", "128" };
    assert_shape(find_result(report, "inner-proxy"), "float32", hidden_shape, 2);
    assert_shape(find_result(report, "inner-input"), "float32", hidden_shape, 2);
    nn_inference_free(report);
    nn_project_close(project);

    project = open_vae();
    assert(project);
    NNValue wrong_features = { .type = NN_VALUE_INT, .as.integer = 31 };
    assert(nn_model_set_parameter(nn_project_model(project), "mean", "out_features",
                                  &wrong_features, error, sizeof(error)));
    report = nn_infer_project(project);
    assert(report);
    assert(find_result(report, "gaussian")->status == NN_INFERENCE_SEMANTIC_ERROR);
    assert(find_result(report, "encoder")->status == NN_INFERENCE_SEMANTIC_ERROR);
    assert(find_result(report, "sample")->status == NN_INFERENCE_UNRESOLVED);
    assert(!strcmp(find_result(report, "sample")->code, "model.blocked"));
    assert(find_result(report, "sample")->cause_node_id);
    assert(!strcmp(find_result(report, "decoder-input")->code, "model.blocked"));
    assert(!strcmp(find_result(report, "decoder-input")->cause_node_id, "gaussian"));
    nn_inference_free(report);
    nn_project_close(project);

    project = open_vae();
    assert(project);
    assert(nn_model_disconnect(nn_project_model(project), "flatten-encoder",
                               error, sizeof(error)));
    report = nn_infer_project(project);
    assert(report && nn_inference_count(report) == 23);
    assert(find_result(report, "flatten")->status == NN_INFERENCE_SUCCESS);
    assert(find_result(report, "encoder")->status == NN_INFERENCE_UNRESOLVED);
    assert(find_result(report, "encoder-input")->status == NN_INFERENCE_UNRESOLVED);
    nn_inference_free(report);
    nn_project_close(project);

    project = open_vae();
    assert(project);
    assert(nn_model_remove_node(nn_project_model(project), "encoder-input",
                                error, sizeof(error)));
    report = nn_infer_project(project);
    assert(report);
    assert(find_result(report, "encoder")->status == NN_INFERENCE_UNRESOLVED);
    assert(find_result(report, "encoder-hidden")->status == NN_INFERENCE_UNRESOLVED);
    nn_inference_free(report);
    nn_project_close(project);

    project = open_vae();
    assert(project);
    assert(nn_model_remove_node(nn_project_model(project), "decoder-output", error, sizeof(error)));
    report = nn_infer_project(project);
    assert(report);
    assert(find_result(report, "decoder")->status == NN_INFERENCE_UNRESOLVED);
    assert(find_result(report, "reconstruction")->status == NN_INFERENCE_UNRESOLVED);
    nn_inference_free(report);
    nn_project_close(project);

    project = open_vae();
    assert(project);
    assert(nn_model_add_node(nn_project_model(project), "extra-input", "extra input",
                             "core.input", "0.1.0", "encoder", 0, 0,
                             error, sizeof(error)));
    report = nn_infer_project(project);
    assert(report);
    assert(find_result(report, "encoder")->status == NN_INFERENCE_SEMANTIC_ERROR);
    nn_inference_free(report);
    assert(nn_model_add_node(nn_project_model(project), "orphan", "orphan",
                             "core.relu", "0.1.0", "missing-owner", 0, 0,
                             error, sizeof(error)));
    report = nn_infer_project(project);
    assert(report);
    assert(find_result(report, "orphan")->status == NN_INFERENCE_UNRESOLVED);
    nn_inference_free(report);
    nn_project_close(project);

    project = open_vae();
    assert(project);
    assert(nn_model_add_node(nn_project_model(project), "repeat-twice", "repeat twice",
                             "core.repeat", "0.1.0", "", 0, 0,
                             error, sizeof(error)));
    assert(nn_model_add_node(nn_project_model(project), "repeat-input", "repeat input",
                             "core.input", "0.1.0", "repeat-twice", 0, 0,
                             error, sizeof(error)));
    assert(nn_model_add_node(nn_project_model(project), "repeat-output", "repeat output",
                             "core.output", "0.1.0", "repeat-twice", 0, 0,
                             error, sizeof(error)));
    assert(nn_model_set_boundary_handle(nn_project_model(project), "repeat-output", "out",
                                        error, sizeof(error)));
    NNValue twice = { .type = NN_VALUE_INT, .as.integer = 2 };
    assert(nn_model_set_parameter(nn_project_model(project), "repeat-twice", "times",
                                  &twice, error, sizeof(error)));
    assert(nn_model_set_parameter(nn_project_model(project), "repeat-input", "binding",
                                  &empty_binding, error, sizeof(error)));
    assert(nn_model_connect(nn_project_model(project), "repeat-root-edge", "flatten", "out",
                            "repeat-twice", "in", error, sizeof(error)));
    assert(nn_model_connect(nn_project_model(project), "repeat-child-edge", "repeat-input", "out",
                            "repeat-output", "in", error, sizeof(error)));
    report = nn_infer_project(project);
    assert(report && nn_inference_count(report) == 26);
    assert_shape(find_result(report, "repeat-twice"), "float32", pixels, 2);
    assert_shape(find_result(report, "repeat-input"), "float32", pixels, 2);
    assert_shape(find_result(report, "repeat-output"), "float32", pixels, 2);
    nn_inference_free(report);
    nn_project_close(project);

    project = open_vae();
    assert(project);
    assert(nn_model_add_node(nn_project_model(project), "repeat-incomplete", "repeat incomplete",
                             "core.repeat", "0.1.0", "", 0, 0,
                             error, sizeof(error)));
    assert(nn_model_add_node(nn_project_model(project), "repeat-only-input", "repeat input",
                             "core.input", "0.1.0", "repeat-incomplete", 0, 0,
                             error, sizeof(error)));
    assert(nn_model_set_parameter(nn_project_model(project), "repeat-incomplete", "times",
                                  &twice, error, sizeof(error)));
    assert(nn_model_set_parameter(nn_project_model(project), "repeat-only-input", "binding",
                                  &empty_binding, error, sizeof(error)));
    assert(nn_model_connect(nn_project_model(project), "repeat-incomplete-root-edge", "flatten", "out",
                            "repeat-incomplete", "in", error, sizeof(error)));
    report = nn_infer_project(project);
    assert(report);
    assert(find_result(report, "repeat-incomplete")->status == NN_INFERENCE_SEMANTIC_ERROR);
    assert(find_result(report, "repeat-only-input")->status == NN_INFERENCE_UNRESOLVED);
    nn_inference_free(report);
    nn_project_close(project);

    project = open_vae();
    assert(project);
    assert(nn_model_add_node(nn_project_model(project), "budget-repeat", "budget repeat",
                             "core.repeat", "0.1.0", "", 0, 0,
                             error, sizeof(error)));
    assert(nn_model_add_node(nn_project_model(project), "budget-input", "budget input",
                             "core.input", "0.1.0", "budget-repeat", 0, 0,
                             error, sizeof(error)));
    assert(nn_model_add_node(nn_project_model(project), "budget-output", "budget output",
                             "core.output", "0.1.0", "budget-repeat", 0, 0,
                             error, sizeof(error)));
    assert(nn_model_set_boundary_handle(nn_project_model(project), "budget-output", "out",
                                        error, sizeof(error)));
    NNValue repeat_count = { .type = NN_VALUE_INT, .as.integer = 300 };
    assert(nn_model_set_parameter(nn_project_model(project), "budget-repeat", "times",
                                  &repeat_count, error, sizeof(error)));
    assert(nn_model_set_parameter(nn_project_model(project), "budget-input", "binding",
                                  &empty_binding, error, sizeof(error)));
    assert(nn_model_connect(nn_project_model(project), "budget-root-edge", "flatten", "out",
                            "budget-repeat", "in", error, sizeof(error)));
    assert(nn_model_connect(nn_project_model(project), "budget-child-edge", "budget-input", "out",
                            "budget-output", "in", error, sizeof(error)));
    report = nn_infer_project(project);
    assert(report);
    const NNInferenceResult *budget = find_result(report, "budget-repeat");
    assert(budget && budget->status == NN_INFERENCE_RUNTIME_FAULT);
    assert(strstr(budget->message, "limit"));
    nn_inference_free(report);
    nn_project_close(project);

    project = open_vae();
    assert(project);
    for (int level = 0; level <= 32; ++level) {
        char owner[48], input_id[48], output_id[48], parent_scope[48] = "";
        snprintf(owner, sizeof(owner), "depth-owner-%d", level);
        snprintf(input_id, sizeof(input_id), "depth-input-%d", level);
        snprintf(output_id, sizeof(output_id), "depth-output-%d", level);
        if (level) snprintf(parent_scope, sizeof(parent_scope), "depth-owner-%d", level - 1);
        assert(nn_model_add_node(nn_project_model(project), owner, "depth proxy",
                                 "core.subflow-proxy", "0.1.0", parent_scope, 0, 0,
                                 error, sizeof(error)));
        assert(nn_model_add_node(nn_project_model(project), input_id, "depth input",
                                 "core.input", "0.1.0", owner, 0, 0,
                                 error, sizeof(error)));
        assert(nn_model_add_node(nn_project_model(project), output_id, "depth output",
                                 "core.output", "0.1.0", owner, 0, 0,
                                 error, sizeof(error)));
        assert(nn_model_set_boundary_handle(nn_project_model(project), output_id, "out",
                                            error, sizeof(error)));
        assert(nn_model_set_parameter(nn_project_model(project), input_id, "binding",
                                      &empty_binding, error, sizeof(error)));
    }
    assert(nn_model_connect(nn_project_model(project), "depth-root-edge", "flatten", "out",
                            "depth-owner-0", "in", error, sizeof(error)));
    for (int level = 0; level <= 32; ++level) {
        char input_id[48], output_id[48], edge[48];
        snprintf(input_id, sizeof(input_id), "depth-input-%d", level);
        snprintf(output_id, sizeof(output_id), "depth-output-%d", level);
        if (level < 32) {
            char next_owner[48];
            snprintf(next_owner, sizeof(next_owner), "depth-owner-%d", level + 1);
            snprintf(edge, sizeof(edge), "depth-in-edge-%d", level);
            assert(nn_model_connect(nn_project_model(project), edge, input_id, "out",
                                    next_owner, "in", error, sizeof(error)));
            snprintf(edge, sizeof(edge), "depth-out-edge-%d", level);
            assert(nn_model_connect(nn_project_model(project), edge, next_owner, "out",
                                    output_id, "in", error, sizeof(error)));
        } else {
            snprintf(edge, sizeof(edge), "depth-in-edge-%d", level);
            assert(nn_model_connect(nn_project_model(project), edge, input_id, "out",
                                    output_id, "in", error, sizeof(error)));
        }
    }
    report = nn_infer_project(project);
    assert(report);
    const NNInferenceResult *depth = find_result(report, "depth-owner-0");
    assert(depth && depth->status == NN_INFERENCE_RUNTIME_FAULT);
    assert(strstr(depth->message, "limit"));
    nn_inference_free(report);
    nn_project_close(project);

    char temporary[] = "/tmp/opencode/nn-inference-diagnostics-XXXXXX";
    char *root = mkdtemp(temporary);
    assert(root);
    NNProject *diagnostic_project = nn_project_create(root, "diagnostics", "Diagnostics",
        false, "stereotype-packages/core", error, sizeof(error));
    assert(diagnostic_project);
    assert(nn_project_create_dataset(diagnostic_project, "local.data", "0.1.0",
        "{\"name\":\"Data\",\"batch\":{\"inputs\":{\"image\":{\"dtype\":\"float32\",\"shape\":[\"B\",8]}},\"targets\":{}}}",
        true, error, sizeof(error)));
    assert(nn_project_create_stereotype(diagnostic_project, "local.broken", "0.1.0",
        "{\"name\":\"Broken\",\"kind\":\"layer\",\"view\":{\"color\":\"#444444\",\"width\":200,\"height\":100},\"parameters\":{}}",
        "return function(", "{}", error, sizeof(error)));
    assert(nn_project_create_stereotype(diagnostic_project, "local.adversarial", "0.1.0",
        "{\"name\":\"Adversarial\",\"kind\":\"layer\",\"view\":{\"color\":\"#444444\",\"width\":200,\"height\":100},\"parameters\":{}}",
        "return function() return setmetatable({}, {__index=function(_, key) "
        "if key == 'status' then return 'success' end; "
        "if key == 'output' then return {dtype='float32', shape={1}} end end}) end",
        "{}", error, sizeof(error)));
    assert(nn_model_add_node(nn_project_model(diagnostic_project), "diagnostic-input", "Input",
        "core.input", "0.1.0", "", 0, 0, error, sizeof(error)));
    NNValue binding_value = { .type = NN_VALUE_STRING, .as.string = "image" };
    assert(nn_model_set_parameter(nn_project_model(diagnostic_project), "diagnostic-input",
        "binding", &binding_value, error, sizeof(error)));
    assert(nn_model_add_node(nn_project_model(diagnostic_project), "broken-rule", "Broken",
        "local.broken", "0.1.0", "", 0, 0, error, sizeof(error)));
    assert(nn_model_connect(nn_project_model(diagnostic_project), "diagnostic-edge",
        "diagnostic-input", "out", "broken-rule", "in", error, sizeof(error)));
    assert(nn_model_add_node(nn_project_model(diagnostic_project), "blocked-rule", "Blocked",
        "core.relu", "0.1.0", "", 0, 0, error, sizeof(error)));
    assert(nn_model_connect(nn_project_model(diagnostic_project), "blocked-edge",
        "broken-rule", "out", "blocked-rule", "in", error, sizeof(error)));
    assert(nn_model_add_node(nn_project_model(diagnostic_project), "adversarial-rule", "Adversarial",
        "local.adversarial", "0.1.0", "", 0, 0, error, sizeof(error)));
    assert(nn_model_connect(nn_project_model(diagnostic_project), "adversarial-edge",
        "diagnostic-input", "out", "adversarial-rule", "in", error, sizeof(error)));
    report = nn_infer_project(diagnostic_project);
    assert(report);
    const NNInferenceResult *compile = find_result(report, "broken-rule");
    assert(compile && compile->status == NN_INFERENCE_COMPILATION_ERROR);
    assert(!strcmp(compile->code, "lua.compile"));
    assert(compile->source_file && strstr(compile->source_file, "inference.lua"));
    assert(compile->source_line == 1);
    const NNInferenceResult *blocked = find_result(report, "blocked-rule");
    assert(blocked && blocked->status == NN_INFERENCE_UNRESOLVED);
    assert(!strcmp(blocked->code, "model.blocked"));
    assert(!strcmp(blocked->cause_node_id, "broken-rule"));
    const NNInferenceResult *adversarial = find_result(report, "adversarial-rule");
    assert(adversarial && adversarial->status == NN_INFERENCE_RUNTIME_FAULT);
    assert(!strcmp(adversarial->code, "analysis.internal"));
    nn_inference_free(report);

    assert(nn_project_create_stereotype(diagnostic_project, "local.two-outputs", "0.1.0",
        "{\"name\":\"Two outputs\",\"kind\":\"layer\","
        "\"outputs\":[{\"id\":\"prediction\",\"type\":\"output\"},"
        "{\"id\":\"objective\",\"type\":\"loss\"}],"
        "\"view\":{\"color\":\"#444444\",\"width\":200,\"height\":100},"
        "\"parameters\":{}}",
        "return function() return {status='success', outputs={"
        "prediction=tensor.create({'B',3},'float32'),"
        "objective=tensor.create({'B',2},'float32')}} end",
        "{}", error, sizeof(error)));
    assert(nn_model_add_node(nn_project_model(diagnostic_project), "two-output", "Two outputs",
        "local.two-outputs", "0.1.0", "", 0, 0, error, sizeof(error)));
    assert(nn_model_connect(nn_project_model(diagnostic_project), "two-output-edge",
        "diagnostic-input", "out", "two-output", "in", error, sizeof(error)));
    report = nn_infer_project(diagnostic_project);
    assert(report);
    const NNInferenceResult *two = find_result(report, "two-output");
    assert(two && two->status == NN_INFERENCE_SUCCESS && two->output_count == 2);
    assert(!strcmp(two->outputs[0].handle_id, "prediction"));
    assert(!strcmp(two->outputs[0].type, "output"));
    assert(two->outputs[0].dimension_count == 2 &&
           !strcmp(two->outputs[0].dimensions[1], "3"));
    assert(!strcmp(two->outputs[1].handle_id, "objective"));
    assert(!strcmp(two->outputs[1].type, "loss"));
    assert(two->outputs[1].dimension_count == 2 &&
           !strcmp(two->outputs[1].dimensions[1], "2"));
    nn_inference_free(report);
    assert(nn_project_create_stereotype(diagnostic_project, "local.pass", "0.1.0",
        "{\"name\":\"Pass\",\"kind\":\"layer\","
        "\"outputs\":[{\"id\":\"passed-loss\",\"type\":\"loss\"}],"
        "\"view\":{\"color\":\"#444444\",\"width\":200,\"height\":100},"
        "\"parameters\":{}}",
        "return function(context) return {status='success', outputs={"
        "['passed-loss']=context.inputs[1]}} end",
        "{}", error, sizeof(error)));
    assert(nn_model_add_node(nn_project_model(diagnostic_project), "prediction-pass", "Prediction pass",
        "local.pass", "0.1.0", "", 0, 0, error, sizeof(error)));
    assert(nn_model_add_node(nn_project_model(diagnostic_project), "objective-pass", "Objective pass",
        "local.pass", "0.1.0", "", 0, 0, error, sizeof(error)));
    assert(nn_model_connect(nn_project_model(diagnostic_project), "prediction-pass-edge",
        "two-output", "prediction", "prediction-pass", "in", error, sizeof(error)));
    assert(nn_model_connect(nn_project_model(diagnostic_project), "objective-pass-edge",
        "two-output", "objective", "objective-pass", "in", error, sizeof(error)));
    report = nn_infer_project(diagnostic_project);
    assert(report);
    assert(find_result(report, "prediction-pass")->dimension_count == 2);
    assert(!strcmp(find_result(report, "prediction-pass")->dimensions[1], "3"));
    assert(find_result(report, "prediction-pass")->output_count == 1);
    assert(!strcmp(find_result(report, "prediction-pass")->outputs[0].handle_id, "passed-loss"));
    assert(!strcmp(find_result(report, "prediction-pass")->outputs[0].type, "loss"));
    assert(find_result(report, "objective-pass")->dimension_count == 2);
    assert(!strcmp(find_result(report, "objective-pass")->dimensions[1], "2"));
    nn_inference_free(report);
    assert(nn_project_create_stereotype(diagnostic_project, "local.missing-output", "0.1.0",
        "{\"name\":\"Missing output\",\"kind\":\"layer\","
        "\"outputs\":[{\"id\":\"prediction\",\"type\":\"output\"},"
        "{\"id\":\"objective\",\"type\":\"loss\"}],"
        "\"view\":{\"color\":\"#444444\",\"width\":200,\"height\":100},"
        "\"parameters\":{}}",
        "return function() return {status='success', outputs={"
        "prediction=tensor.create({'B',3},'float32')}} end",
        "{}", error, sizeof(error)));
    assert(nn_model_add_node(nn_project_model(diagnostic_project), "missing-output",
        "Missing output", "local.missing-output", "0.1.0", "", 0, 0,
        error, sizeof(error)));
    assert(nn_model_connect(nn_project_model(diagnostic_project), "missing-output-edge",
        "diagnostic-input", "out", "missing-output", "in", error, sizeof(error)));
    report = nn_infer_project(diagnostic_project);
    assert(report);
    assert(find_result(report, "missing-output")->status == NN_INFERENCE_SEMANTIC_ERROR);
    assert(find_result(report, "missing-output")->output_count == 0);
    nn_inference_free(report);
    assert(nn_project_create_stereotype(diagnostic_project, "local.extra-output", "0.1.0",
        "{\"name\":\"Extra output\",\"kind\":\"layer\","
        "\"outputs\":[{\"id\":\"prediction\",\"type\":\"output\"},"
        "{\"id\":\"objective\",\"type\":\"loss\"}],"
        "\"view\":{\"color\":\"#444444\",\"width\":200,\"height\":100},"
        "\"parameters\":{}}",
        "return function() return {status='success', outputs={"
        "prediction=tensor.create({'B',3},'float32'),"
        "objective=tensor.create({'B',2},'float32'),"
        "['prediction\\0evil']=tensor.create({'B',1},'float32')}} end",
        "{}", error, sizeof(error)));
    assert(nn_model_add_node(nn_project_model(diagnostic_project), "extra-output",
        "Extra output", "local.extra-output", "0.1.0", "", 0, 0,
        error, sizeof(error)));
    assert(nn_model_connect(nn_project_model(diagnostic_project), "extra-output-edge",
        "diagnostic-input", "out", "extra-output", "in", error, sizeof(error)));
    report = nn_infer_project(diagnostic_project);
    assert(report);
    assert(find_result(report, "extra-output")->status == NN_INFERENCE_SEMANTIC_ERROR);
    assert(find_result(report, "extra-output")->output_count == 0);
    nn_inference_free(report);
    assert(nn_project_create_stereotype(diagnostic_project, "local.both-output", "0.1.0",
        "{\"name\":\"Both outputs\",\"kind\":\"layer\","
        "\"outputs\":[{\"id\":\"arbitrary\",\"type\":\"output\"}],"
        "\"view\":{\"color\":\"#444444\",\"width\":200,\"height\":100},"
        "\"parameters\":{}}",
        "return function() local t=tensor.create({'B',1},'float32'); "
        "return {status='success', output=t, outputs={arbitrary=t}} end",
        "{}", error, sizeof(error)));
    assert(nn_model_add_node(nn_project_model(diagnostic_project), "both-output",
        "Both outputs", "local.both-output", "0.1.0", "", 0, 0,
        error, sizeof(error)));
    assert(nn_model_connect(nn_project_model(diagnostic_project), "both-output-edge",
        "diagnostic-input", "out", "both-output", "in", error, sizeof(error)));
    report = nn_infer_project(diagnostic_project);
    assert(report);
    assert(find_result(report, "both-output")->status == NN_INFERENCE_SEMANTIC_ERROR);
    assert(find_result(report, "both-output")->output_count == 0);
    nn_inference_free(report);
    assert(nn_project_create_stereotype(diagnostic_project, "local.both-two", "0.1.0",
        "{\"name\":\"Both two outputs\",\"kind\":\"layer\","
        "\"outputs\":[{\"id\":\"prediction\",\"type\":\"output\"},"
        "{\"id\":\"objective\",\"type\":\"loss\"}],"
        "\"view\":{\"color\":\"#444444\",\"width\":200,\"height\":100},"
        "\"parameters\":{}}",
        "return function() local p=tensor.create({'B',3},'float32'); "
        "local o=tensor.create({'B',2},'float32'); return {status='success', "
        "output=p, outputs={prediction=p, objective=o}} end",
        "{}", error, sizeof(error)));
    assert(nn_model_add_node(nn_project_model(diagnostic_project), "both-two",
        "Both two outputs", "local.both-two", "0.1.0", "", 0, 0,
        error, sizeof(error)));
    assert(nn_model_connect(nn_project_model(diagnostic_project), "both-two-edge",
        "diagnostic-input", "out", "both-two", "in", error, sizeof(error)));
    report = nn_infer_project(diagnostic_project);
    assert(report);
    assert(find_result(report, "both-two")->status == NN_INFERENCE_SEMANTIC_ERROR);
    assert(find_result(report, "both-two")->output_count == 0);
    nn_inference_free(report);
    assert(nn_project_create_stereotype(diagnostic_project, "local.terminal-map", "0.1.0",
        "{\"name\":\"Terminal map\",\"kind\":\"output\",\"outputs\":[],"
        "\"view\":{\"color\":\"#444444\",\"width\":200,\"height\":100},"
        "\"parameters\":{}}",
        "return function() return {status='success', outputs={"
        "arbitrary=tensor.create({'B',1},'float32')}} end",
        "{}", error, sizeof(error)));
    assert(nn_model_add_node(nn_project_model(diagnostic_project), "terminal-map",
        "Terminal map", "local.terminal-map", "0.1.0", "", 0, 0,
        error, sizeof(error)));
    assert(nn_model_connect(nn_project_model(diagnostic_project), "terminal-map-edge",
        "diagnostic-input", "out", "terminal-map", "in", error, sizeof(error)));
    report = nn_infer_project(diagnostic_project);
    assert(report);
    assert(find_result(report, "terminal-map")->status == NN_INFERENCE_SEMANTIC_ERROR);
    assert(find_result(report, "terminal-map")->output_count == 0);
    nn_inference_free(report);
    assert(nn_project_create_stereotype(diagnostic_project, "local.multi-subflow", "0.1.0",
        "{\"name\":\"Multi subflow\",\"kind\":\"subflow\","
        "\"outputs\":[{\"id\":\"objective\",\"type\":\"loss\"},"
        "{\"id\":\"prediction\",\"type\":\"output\"}],"
        "\"view\":{\"color\":\"#444444\",\"width\":200,\"height\":100},"
        "\"parameters\":{}}",
        "return function(context, parameters, services) return services.infer_subflow(context.inputs[1]) end",
        "{}", error, sizeof(error)));
    assert(nn_model_add_node(nn_project_model(diagnostic_project), "nested-two", "Nested two",
        "local.multi-subflow", "0.1.0", "", 0, 0, error, sizeof(error)));
    assert(nn_model_connect(nn_project_model(diagnostic_project), "nested-two-parent-edge",
        "diagnostic-input", "out", "nested-two", "in", error, sizeof(error)));
    assert(nn_model_add_node(nn_project_model(diagnostic_project), "nested-two-input", "Nested input",
        "core.input", "0.1.0", "nested-two", 0, 0, error, sizeof(error)));
    NNValue nested_binding = { .type = NN_VALUE_STRING, .as.string = "" };
    assert(nn_model_set_parameter(nn_project_model(diagnostic_project), "nested-two-input", "binding",
        &nested_binding, error, sizeof(error)));
    assert(nn_model_add_node(nn_project_model(diagnostic_project), "nested-two-rule", "Nested outputs",
        "local.two-outputs", "0.1.0", "nested-two", 0, 0, error, sizeof(error)));
    assert(nn_model_add_node(nn_project_model(diagnostic_project), "nested-two-output", "Prediction boundary",
        "core.output", "0.1.0", "nested-two", 0, 0, error, sizeof(error)));
    assert(nn_model_set_boundary_handle(nn_project_model(diagnostic_project), "nested-two-output",
        "prediction", error, sizeof(error)));
    assert(nn_model_add_node(nn_project_model(diagnostic_project), "nested-two-loss", "Loss boundary",
        "core.loss-output", "0.1.0", "nested-two", 0, 0, error, sizeof(error)));
    assert(nn_model_set_boundary_handle(nn_project_model(diagnostic_project), "nested-two-loss",
        "objective", error, sizeof(error)));
    assert(nn_model_connect(nn_project_model(diagnostic_project), "nested-two-input-edge",
        "nested-two-input", "out", "nested-two-rule", "in", error, sizeof(error)));
    assert(nn_model_connect(nn_project_model(diagnostic_project), "nested-two-output-edge",
        "nested-two-rule", "prediction", "nested-two-output", "in", error, sizeof(error)));
    assert(nn_model_connect(nn_project_model(diagnostic_project), "nested-two-loss-edge",
        "nested-two-rule", "objective", "nested-two-loss", "in", error, sizeof(error)));
    report = nn_infer_project(diagnostic_project);
    assert(report);
    two = find_result(report, "nested-two");
    assert(two && two->status == NN_INFERENCE_SUCCESS && two->output_count == 2);
    assert(!strcmp(two->outputs[0].handle_id, "objective"));
    assert(!strcmp(two->outputs[1].handle_id, "prediction"));
    assert(!strcmp(two->outputs[0].type, "loss"));
    assert(!strcmp(two->outputs[1].type, "output"));
    assert(!strcmp(two->outputs[0].dimensions[1], "2"));
    assert(!strcmp(two->outputs[1].dimensions[1], "3"));
    nn_inference_free(report);
    assert(nn_model_set_boundary_handle(nn_project_model(diagnostic_project),
        "nested-two-rule", "prediction", error, sizeof(error)));
    report = nn_infer_project(diagnostic_project);
    assert(report);
    assert(find_result(report, "nested-two")->status == NN_INFERENCE_SEMANTIC_ERROR);
    nn_inference_free(report);
    nn_project_close(diagnostic_project);
    return 0;
}
