#define _XOPEN_SOURCE 700
#include "application/application.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static const NNInferenceResult *result(NNApplication *app)
{
    char error[512];
    const NNInferenceReport *report = nn_app_analysis(app, error, sizeof(error));
    assert(report);
    for (size_t i = 0; i < nn_inference_count(report); ++i) {
        const NNInferenceResult *item = nn_inference_at(report, i);
        if (!strcmp(item->node_id, "parallel")) return item;
    }
    assert(0); return NULL;
}

int main(void)
{
    char directory[] = "/tmp/nn-reference-XXXXXX";
    assert(mkdtemp(directory));
    NNApplication *app = nn_app_new("stereotype-packages/core");
    char error[512] = "";
    assert(app && nn_app_create(app, directory, "test", "Test", false, error, sizeof(error)));
    assert(nn_app_create_dataset(app, "test.features", "1.0.0",
        "{\"name\":\"Features\",\"batch\":{\"inputs\":{\"x\":{\"dtype\":\"float32\",\"shape\":[\"B\",4]}},\"targets\":{}}}",
        true, error, sizeof(error)));
    assert(nn_app_add_node(app, "input", "core.input", "0.1.0", "", 0, 0, error, sizeof(error)));
    assert(nn_app_add_node(app, "parallel", "core.horizontal-repeat", "0.1.0", "", 0, 0, error, sizeof(error)));
    assert(nn_app_connect(app, "feed", "input", "out", "parallel", "in", error, sizeof(error)));
    assert(nn_app_connect(app, "body", "parallel-input", "out", "parallel-boundary-out", "in", error, sizeof(error)));
    const NNInferenceResult *item = result(app);
    if (item->status != NN_INFERENCE_SUCCESS) fprintf(stderr, "%s\n", item->message);
    assert(item->status == NN_INFERENCE_SUCCESS && !strcmp(item->dimensions[1], "8"));
    char *before = nn_app_parameter_text(app, "parallel", "join");
    assert(before && strstr(before, "core.concat"));
    const char *invalid[] = {
        "{}", "{\"id\":\"core.relu\",\"version\":\"0.1.0\",\"parameters\":{}}",
        "{\"id\":\"core.concat\",\"version\":\"9.0.0\",\"parameters\":{}}",
        "{\"id\":\"core.concat\",\"version\":\"0.1.0\",\"parameters\":{\"dim\":\"wrong\"}}",
        "{\"id\":\"core.concat\",\"version\":\"0.1.0\",\"parameters\":{\"unknown\":1}}",
        "{\"id\":\"core.concat\",\"id\":\"core.add\",\"version\":\"0.1.0\",\"parameters\":{}}"
    };
    for (size_t i = 0; i < sizeof(invalid) / sizeof(*invalid); ++i) {
        assert(!nn_app_set_parameter_text(app, "parallel", "join", invalid[i], error, sizeof(error)));
        char *after = nn_app_parameter_text(app, "parallel", "join");
        assert(after && !strcmp(before, after)); nn_app_free_text(after);
    }
    nn_app_free_text(before);
    assert(nn_app_set_parameter_text(app, "parallel", "join",
        "{\"id\":\"core.add\",\"version\":\"^0.1.0\",\"parameters\":{}}", error, sizeof(error)));
    assert(result(app)->status == NN_INFERENCE_SUCCESS && !strcmp(result(app)->dimensions[1], "4"));
    assert(nn_app_undo(app, error, sizeof(error)));
    assert(!strcmp(result(app)->dimensions[1], "8"));
    assert(nn_app_redo(app, error, sizeof(error)));
    assert(!strcmp(result(app)->dimensions[1], "4"));
    assert(nn_app_save(app, error, sizeof(error)));
    char *path = strdup(nn_project_directory(nn_app_project(app)));
    assert(path && nn_app_close(app, false, error, sizeof(error)));
    assert(nn_app_open(app, path, error, sizeof(error)));
    assert(!strcmp(result(app)->dimensions[1], "4"));
    free(path); nn_app_free(app);
    puts("stereotype references passed");
    return 0;
}
