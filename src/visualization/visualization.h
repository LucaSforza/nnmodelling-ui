#ifndef NN_VISUALIZATION_H
#define NN_VISUALIZATION_H
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct NNProject NNProject;
typedef struct NN3DScene NN3DScene;
typedef struct { double x, y, z; } NN3DVec;
typedef struct {
    const char *path, *source_id, *label, *parameters, *package_id, *color;
    size_t group;
    NN3DVec center, size;
} NN3DNode;
typedef struct {
    size_t source, target;
    const char *source_handle, *target_handle;
    bool loss;
} NN3DEdge;
typedef struct {
    const char *path, *source_id, *label;
    size_t parent;
    NN3DVec minimum, maximum;
} NN3DGroup;
typedef struct { NN3DVec position; double yaw, pitch; } NN3DCamera;
typedef enum { NN_3D_FACE, NN_3D_LINE, NN_3D_LABEL } NN3DPrimitiveKind;
typedef struct {
    NN3DPrimitiveKind kind;
    double x[4], y[4], depth;
    uint32_t rgba;
    size_t node; /* SIZE_MAX for non-node primitives. */
    const char *text; /* borrowed from scene; null except labels. */
} NN3DPrimitive;
typedef struct { NN3DPrimitive *items; size_t count; } NN3DFrame;
NN3DScene *nn_3d_build(const NNProject *project, char *error, size_t capacity);
void nn_3d_free(NN3DScene *scene);
size_t nn_3d_node_count(const NN3DScene *scene);
size_t nn_3d_edge_count(const NN3DScene *scene);
size_t nn_3d_group_count(const NN3DScene *scene);
const NN3DNode *nn_3d_node_at(const NN3DScene *scene, size_t index);
const NN3DEdge *nn_3d_edge_at(const NN3DScene *scene, size_t index);
const NN3DGroup *nn_3d_group_at(const NN3DScene *scene, size_t index);
void nn_3d_camera_fit(const NN3DScene *scene, NN3DCamera *camera, double aspect);
void nn_3d_camera_home(const NN3DScene *scene, NN3DCamera *camera, double aspect);
void nn_3d_camera_look(NN3DCamera *camera, double yaw_delta, double pitch_delta);
void nn_3d_camera_move(NN3DCamera *camera, double right, double up, double forward);
bool nn_3d_frame(const NN3DScene *scene, const NN3DCamera *camera,
                 double width, double height, NN3DFrame *frame);
void nn_3d_frame_dispose(NN3DFrame *frame);
size_t nn_3d_pick(const NN3DFrame *frame, double x, double y);
#ifdef __cplusplus
}
#endif
#endif
