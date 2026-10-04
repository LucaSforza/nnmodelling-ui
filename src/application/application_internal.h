#ifndef NN_APPLICATION_INTERNAL_H
#define NN_APPLICATION_INTERNAL_H
#include "application/application.h"
#include "application/history_internal.h"
#include "yyjson.h"
struct NNApplication {
    char *core_root;
    NNProject *project;
    NNInferenceReport *analysis;
    NNEditHistory history;
};
void nn_app_invalidate_analysis(NNApplication *app);
const NNPackage *nn_app_find_package(const NNApplication *app, const NNNode *node);
bool nn_app_parse_json_value(const yyjson_val *source, NNValue *value, unsigned depth);
bool nn_app_add_default(NNApplication *app, NNModel *model, const char *node_id,
                        const NNPackage *package, const NNParameterDef *definition,
                        char *error, size_t capacity);
bool nn_app_history_prepare(NNApplication *app, char *error, size_t capacity);
void nn_app_history_finish(NNApplication *app, bool success);
void nn_app_history_reset(NNApplication *app);
void nn_app_history_barrier(NNApplication *app, bool saved);
void nn_app_history_mark_saved(NNApplication *app);
bool nn_app_history_fail(NNApplication *app, char *error, size_t capacity,
                         const char *message);
#endif
