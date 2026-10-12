#ifndef NN_PARAMETER_COUNTS_H
#define NN_PARAMETER_COUNTS_H

#include <stddef.h>

typedef struct NNProject NNProject;
typedef struct NNInferenceReport NNInferenceReport;

/* Returns owned JSON text. Unknown package rules preserve known subtotals and
   set complete=false. The caller releases the text with free(). */
char *nn_parameter_counts_json(const NNProject *project,
                               const NNInferenceReport *analysis,
                               char *error, size_t capacity);

#endif
