#ifndef NN_INFERENCE_H
#define NN_INFERENCE_H

#include <stddef.h>

typedef struct NNProject NNProject;
typedef struct NNInferenceReport NNInferenceReport;

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    NN_INFERENCE_SUCCESS,
    NN_INFERENCE_SEMANTIC_ERROR,
    NN_INFERENCE_UNRESOLVED,
    NN_INFERENCE_RUNTIME_FAULT
} NNInferenceStatus;

typedef struct {
    const char *node_id;
    NNInferenceStatus status;
    const char *message;
    const char *dtype;
    const char *const *dimensions;
    size_t dimension_count;
} NNInferenceResult;

/* The report owns its results and strings; project state is never changed. */
NNInferenceReport *nn_infer_project(const NNProject *project);
void nn_inference_free(NNInferenceReport *report);
size_t nn_inference_count(const NNInferenceReport *report);
const NNInferenceResult *nn_inference_at(const NNInferenceReport *report, size_t index);

#ifdef __cplusplus
}
#endif

#endif
