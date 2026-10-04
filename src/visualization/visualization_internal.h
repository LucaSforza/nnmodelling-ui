#ifndef NN_VISUALIZATION_INTERNAL_H
#define NN_VISUALIZATION_INTERNAL_H

#include "catalog/catalog.h"
#include "model/model.h"
#include "visualization/visualization.h"

struct NN3DScene {
  NN3DNode *nodes;
  size_t node_count;
  NN3DEdge *edges;
  size_t edge_count;
  NN3DGroup *groups;
  size_t group_count;
  size_t body_instances;
};

typedef struct {
  char *id, *label, *package_id, *version;
  bool body;
  NNParameter *parameters;
  size_t parameter_count;
  const NNPackage *package;
} NN3DPlanNode;
typedef struct {
  char *source, *source_handle, *target, *target_handle;
} NN3DPlanEdge;
typedef struct {
  char *id, *node, *handle;
} NN3DPlanOutput;
typedef struct {
  NN3DPlanNode *nodes;
  size_t node_count;
  NN3DPlanEdge *edges;
  size_t edge_count;
  NN3DPlanOutput *outputs;
  size_t output_count;
} NN3DPlan;
typedef struct {
  size_t input;
  size_t outputs[2];
  size_t output_count;
} NN3DScopeBoundary;

double nn_3d_clamp(double value, double minimum, double maximum);
bool nn_3d_plan_load(const NNCatalog *catalog, const NNPackage *owner,
                     const NNNode *node, NN3DPlan *plan, char *error,
                     size_t cap);
void nn_3d_plan_dispose(NN3DPlan *plan);

#endif
