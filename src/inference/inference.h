#ifndef NN_INFERENCE_H
#define NN_INFERENCE_H

#include <stddef.h>
#include <stdbool.h>

typedef struct NNProject NNProject;
typedef struct NNInferenceReport NNInferenceReport;

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    NN_INFERENCE_SUCCESS,
    NN_INFERENCE_SEMANTIC_ERROR,
    NN_INFERENCE_UNRESOLVED,
    NN_INFERENCE_RUNTIME_FAULT,
    NN_INFERENCE_COMPILATION_ERROR
} NNInferenceStatus;

typedef struct {
    const char *handle_id;
    const char *type;
    const char *dtype;
    const char *const *dimensions;
    size_t dimension_count;
} NNInferenceTensor;

typedef struct {
    const char *node_id;
    NNInferenceStatus status;
    const char *message;
    const char *code;
    const char *cause_node_id;
    const char *source_file;
    size_t source_line;
    const char *dtype;
    const char *const *dimensions;
    size_t dimension_count;
    const NNInferenceTensor *outputs;
    size_t output_count;
} NNInferenceResult;

/* The report owns its results and strings; project state is never changed. */
NNInferenceReport *nn_infer_project(const NNProject *project);
bool nn_inference_validate_source(const char *source, char *error,
                                 size_t error_capacity);
void nn_inference_free(NNInferenceReport *report);
size_t nn_inference_count(const NNInferenceReport *report);
const NNInferenceResult *nn_inference_at(const NNInferenceReport *report, size_t index);
NNInferenceStatus nn_inference_root_status(const NNInferenceReport *report);
const char *nn_inference_root_message(const NNInferenceReport *report);
const char *nn_inference_category(NNInferenceStatus status);
const char *nn_inference_severity(NNInferenceStatus status);
size_t nn_inference_error_line(const char *message);

#ifdef __cplusplus
}
#endif

#endif
