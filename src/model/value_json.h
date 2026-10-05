#ifndef NN_VALUE_JSON_H
#define NN_VALUE_JSON_H
#include "model/model.h"
#include "yyjson.h"
#ifdef __cplusplus
extern "C" {
#endif
bool nn_value_from_json(yyjson_val *source, NNValue *target);
#ifdef __cplusplus
}
#endif
#endif
