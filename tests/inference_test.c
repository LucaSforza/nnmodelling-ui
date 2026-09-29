#include "inference.h"
#include "model.h"
#include "project.h"

#include <assert.h>
#include <stdio.h>
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

int main(void)
{
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
    return 0;
}
