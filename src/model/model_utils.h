#ifndef NN_MODEL_UTILS_H
#define NN_MODEL_UTILS_H

#include "model.h"

bool nn_model_reserve(void **items, size_t *capacity, size_t count, size_t item_size);
size_t nn_model_node_index(const NNModel *model, const char *id);
size_t nn_model_edge_index(const NNModel *model, const char *id);
bool nn_model_valid_id(const char *id);

#endif
