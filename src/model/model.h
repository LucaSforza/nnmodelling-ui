#ifndef NN_MODEL_H
#define NN_MODEL_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef struct NNModel NNModel;

enum { NN_MODEL_GRID_SPACING = 20 };

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    NN_VALUE_BOOL,
    NN_VALUE_INT,
    NN_VALUE_REAL,
    NN_VALUE_STRING,
    NN_VALUE_ARRAY
} NNValueType;

typedef struct NNValue NNValue;
struct NNValue {
    NNValueType type;
    union {
        bool boolean;
        long long integer;
        double real;
        char *string;
        struct { NNValue *items; size_t count; } array;
    } as;
};

typedef struct {
    char *key;
    NNValue value;
} NNParameter;

/* Snapshot fields and referenced strings/values are borrowed until mutation. */
typedef struct {
    const char *id;
    const char *label;
    const char *package_id;
    const char *package_version;
    const char *scope_id;
    const char *boundary_handle_id;
    int32_t x, y;
    const NNParameter *parameters;
    size_t parameter_count;
} NNNode;

typedef struct {
    const char *id;
    const char *source_id;
    const char *source_handle_id;
    const char *target_id;
    const char *target_handle_id;
    const char *scope_id;
} NNEdge;

bool nn_value_copy(NNValue *destination, const NNValue *source);
void nn_value_dispose(NNValue *value);

NNModel *nn_model_new(void);
void nn_model_free(NNModel *model);

bool nn_model_add_node(NNModel *model, const char *id, const char *label,
                       const char *package_id, const char *package_version,
                       const char *scope_id, double x, double y,
                       char *error, size_t error_capacity);
bool nn_model_remove_node(NNModel *model, const char *id,
                          char *error, size_t error_capacity);
bool nn_model_move_node(NNModel *model, const char *id, double x, double y,
                        char *error, size_t error_capacity);
bool nn_model_rename_node(NNModel *model, const char *id, const char *label,
                          char *error, size_t error_capacity);
bool nn_model_connect(NNModel *model, const char *id,
                      const char *source_id, const char *source_handle_id,
                      const char *target_id, const char *target_handle_id,
                      char *error, size_t error_capacity);
bool nn_model_disconnect(NNModel *model, const char *id,
                         char *error, size_t error_capacity);
bool nn_model_set_parameter(NNModel *model, const char *node_id,
                            const char *key, const NNValue *value,
                            char *error, size_t error_capacity);
bool nn_model_set_boundary_handle(NNModel *model, const char *node_id,
                                  const char *handle_id,
                                  char *error, size_t error_capacity);

size_t nn_model_node_count(const NNModel *model);
size_t nn_model_edge_count(const NNModel *model);
const NNNode *nn_model_node_at(const NNModel *model, size_t index);
const NNEdge *nn_model_edge_at(const NNModel *model, size_t index);
const NNNode *nn_model_find_node(const NNModel *model, const char *id);

#ifdef __cplusplus
}
#endif

#endif
