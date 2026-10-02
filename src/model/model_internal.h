#ifndef NN_MODEL_INTERNAL_H
#define NN_MODEL_INTERNAL_H

#include "model.h"

typedef struct { NNNode view; } NodeRecord;
typedef struct { NNEdge view; } EdgeRecord;

struct NNModel {
    NodeRecord *nodes;
    size_t node_count, node_capacity;
    EdgeRecord *edges;
    size_t edge_count, edge_capacity;
};

void nn_model_node_dispose(NNNode *node);
void nn_model_edge_dispose(NNEdge *edge);

#endif
