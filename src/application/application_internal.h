#ifndef NN_APPLICATION_INTERNAL_H
#define NN_APPLICATION_INTERNAL_H
#include "application/application.h"
#include "yyjson.h"
struct NNApplication { char *core_root; NNProject *project; NNInferenceReport *analysis; };
void nn_app_invalidate_analysis(NNApplication *app);
bool nn_app_kind_is(const NNPackage *package, const char *kind);
const NNPackage *nn_app_find_package(const NNApplication *app, const NNNode *node);
bool nn_app_valid_output_handle(const NNPackage *package, const char *handle);
bool nn_app_valid_input_handle(const NNPackage *package, const char *handle);
bool nn_app_parse_json_value(const yyjson_val *source, NNValue *value, unsigned depth);
bool nn_app_add_default(NNApplication *app, NNModel *model, const char *node_id,
                        const NNPackage *package, const NNParameterDef *definition,
                        char *error, size_t capacity);
#endif
