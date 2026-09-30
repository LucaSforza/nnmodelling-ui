#include "inference.h"
#include "model.h"
#include "project.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

    NNProject *project = open_example();
    assert(project);
    NNInferenceReport *report = nn_infer_project(project);
    assert(report && nn_inference_count(report) == 8);
    const NNInferenceResult *output = find_result(report, "output");
    assert(output && output->status == NN_INFERENCE_SUCCESS);
    assert(!strcmp(output->dtype, "float32"));
    assert(output->dimension_count == 2);
    assert(!strcmp(output->dimensions[0], "B"));
    assert(!strcmp(output->dimensions[1], "10"));
    nn_inference_free(report);

    NNValue bad_features = { .type = NN_VALUE_INT, .as.integer = 800 };
    char error[256] = {0};
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
    assert(report && nn_inference_count(report) == 21);
    const char *packed[] = { "B", "2", "32" };
    const char *latent[] = { "B", "32" };
    const char *pixels[] = { "B", "784" };
    const char *kl_shape[] = { "B" };
    assert_shape(find_result(report, "encoder"), "float32", packed, 3);
    assert_shape(find_result(report, "encoder-input"), "float32", pixels, 2);
    assert_shape(find_result(report, "decoder-input"), "float32", latent, 2);
    assert_shape(find_result(report, "reconstruction"), "float32", pixels, 2);
    assert_shape(find_result(report, "kl-output"), "float32", kl_shape, 1);
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
    NNValue empty_binding = { .type = NN_VALUE_STRING, .as.string = "" };
    assert(nn_model_set_parameter(nn_project_model(project), "inner-input", "binding",
                                  &empty_binding, error, sizeof(error)));
    assert(nn_model_connect(nn_project_model(project), "inner-parent-edge", "encoder-relu", "out",
                            "inner-proxy", "in", error, sizeof(error)));
    assert(nn_model_connect(nn_project_model(project), "inner-edge", "inner-input", "out",
                            "inner-output", "in", error, sizeof(error)));
    report = nn_infer_project(project);
    assert(report && nn_inference_count(report) == 24);
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
    nn_inference_free(report);
    nn_project_close(project);

    project = open_vae();
    assert(project);
    assert(nn_model_disconnect(nn_project_model(project), "flatten-encoder",
                               error, sizeof(error)));
    report = nn_infer_project(project);
    assert(report && nn_inference_count(report) == 21);
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
    assert(report && nn_inference_count(report) == 24);
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
    assert(find_result(report, "repeat-incomplete")->status == NN_INFERENCE_UNRESOLVED);
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
    return 0;
}
