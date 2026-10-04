#include "project/project.h"
#include "inference/inference.h"
#include "visualization/visualization.h"
#include <assert.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>

static const NNInferenceResult *result(const NNInferenceReport *report, const char *id)
{
    for (size_t i = 0; i < nn_inference_count(report); ++i) {
        const NNInferenceResult *r = nn_inference_at(report, i);
        if (!strcmp(r->node_id, id)) return r;
    }
    return NULL;
}
static const char *node_id(const NNModel *model, const char *label)
{
    for (size_t i = 0; i < nn_model_node_count(model); ++i) {
        const NNNode *n = nn_model_node_at(model, i);
        if (!strcmp(n->label, label)) return n->id;
    }
    return NULL;
}
int main(void)
{
    char error[512] = {0};
    NNProject *project = nn_project_open("examples/tiny-decoder-llm", "stereotype-packages/core", error, sizeof(error));
    if (!project) fprintf(stderr, "%s\n", error);
    assert(project);
    NNModel *before = nn_model_copy(nn_project_model(project));
    assert(before);
    NNInferenceReport *report = nn_infer_project(project);
    assert(report && nn_inference_root_status(report) == NN_INFERENCE_SUCCESS);
    for (size_t i = 0; i < nn_inference_count(report); ++i) {
        const NNInferenceResult *r = nn_inference_at(report, i);
        if (r->status != NN_INFERENCE_SUCCESS)
            fprintf(stderr, "%s: %s\n", r->node_id, r->message);
        assert(r->status == NN_INFERENCE_SUCCESS);
    }
    const NNInferenceResult *logits = result(report, "output");
    assert(logits && !strcmp(logits->dtype, "float32") && logits->dimension_count == 3);
    assert(!strcmp(logits->dimensions[0], "B") && !strcmp(logits->dimensions[1], "128") && !strcmp(logits->dimensions[2], "32000"));
    const NNInferenceResult *loss = result(report, "loss-output");
    assert(loss && !strcmp(loss->dtype, "float32") && loss->dimension_count == 0);
    NN3DScene *scene = nn_3d_build(project, error, sizeof(error));
    if (!scene) fprintf(stderr, "3D: %s\n", error);
    assert(scene);
    const NNModel *model = before;
    const char *query_id = node_id(model, "Query Q");
    const char *key_id = node_id(model, "Key K");
    const char *value_id = node_id(model, "Value V");
    const char *mask_id = node_id(model, "Causal mask");
    assert(query_id && key_id && value_id && mask_id);
    size_t queries = 0, keys = 0, values = 0, masks = 0;
    for (size_t i = 0; i < nn_3d_node_count(scene); ++i) {
        const NN3DNode *n = nn_3d_node_at(scene, i);
        assert(n && n->path && n->label && n->source_id);
        if (!strcmp(n->source_id, query_id)) ++queries;
        if (!strcmp(n->source_id, key_id)) ++keys;
        if (!strcmp(n->source_id, value_id)) ++values;
        if (!strcmp(n->source_id, mask_id)) ++masks;
        for (size_t j = 0; j < i; ++j)
            assert(strcmp(n->path, nn_3d_node_at(scene, j)->path));
    }
    assert(queries == 48 && keys == 48 && values == 48 && masks == 48);
    assert(nn_3d_group_count(scene) >= 54);
    NN3DCamera camera = {0};
    nn_3d_camera_fit(scene, &camera, 16.0/9.0);
    NN3DFrame frame = {0};
    assert(nn_3d_frame(scene, &camera, 1600, 900, &frame) && frame.count);
    nn_3d_frame_dispose(&frame);
    nn_3d_camera_home(scene, &camera, 16.0/9.0);
    assert(nn_3d_frame(scene, &camera, 1600, 900, &frame) && frame.count);
    nn_3d_frame_dispose(&frame);
    assert(nn_model_equal(before, nn_project_model(project)));
    assert(!nn_project_dirty(project));
    printf("LLM 3D: %zu nodes, %zu edges, %zu groups; 48 Q/K/V heads; complete inference\n", nn_3d_node_count(scene), nn_3d_edge_count(scene), nn_3d_group_count(scene));
    nn_3d_free(scene);
    /* Forty compact blocks must expand independently of inference's invocation budget. */
    NNValue forty = {.type = NN_VALUE_INT, .as.integer = 40};
    assert(nn_model_set_parameter(nn_project_model(project),
        "node-467fa101-ec5a-4e0a-82e5-56594d350337", "times", &forty, error, sizeof(error)));
    scene = nn_3d_build(project, error, sizeof(error));
    if (!scene) fprintf(stderr, "40-block 3D: %s\n", error);
    assert(scene);
    queries = 0;
    for (size_t i = 0; i < nn_3d_node_count(scene); ++i)
        if (!strcmp(nn_3d_node_at(scene, i)->source_id, query_id)) ++queries;
    assert(queries == 320);
    nn_3d_free(scene); nn_inference_free(report); nn_model_free(before); nn_project_close(project);
    return 0;
}
