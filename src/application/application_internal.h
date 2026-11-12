#ifndef NN_APPLICATION_INTERNAL_H
#define NN_APPLICATION_INTERNAL_H
#include "application/application.h"
#include "application/history_internal.h"
#include "activity/activity.h"
#include "yyjson.h"
struct NNApplication {
    char *core_root;
    NNProject *project;
    NNInferenceReport *analysis;
    NNEditHistory history;
    NNActivityLog *activity; /* borrowed */
    const char *activity_source;
    bool snapshot_pending;
    char *analysis_signature;
};

/* One activity event around a public operation; a no-op while logging is off. */
typedef struct {
    NNApplication *app;
    char *caller;
    size_t caller_cap;
    char local[512];
    char *error; /* buffer to pass to the operation */
    size_t cap;
    yyjson_mut_doc *doc;
    yyjson_mut_val *data;
} NNAppEvent;
void nn_app_event_begin(NNAppEvent *event, NNApplication *app, char *error, size_t cap);
void nn_app_event_str(NNAppEvent *event, const char *key, const char *text);
void nn_app_event_path(NNAppEvent *event, const char *key, const char *path);
void nn_app_event_num(NNAppEvent *event, const char *key, double number);
void nn_app_event_bool(NNAppEvent *event, const char *key, bool value);
/* Emits the event and any due model.snapshot; returns ok unchanged. */
bool nn_app_event_end(NNAppEvent *event, bool ok, const char *name);
void nn_app_event_analysis(NNApplication *app, const NNInferenceReport *report);
void nn_app_activity_reset(NNApplication *app);
yyjson_mut_val *nn_app_value_json(yyjson_mut_doc *doc, const NNValue *value);
void nn_app_invalidate_analysis(NNApplication *app);
bool nn_app_kind_is(const NNPackage *package, const char *kind);
const NNPackage *nn_app_find_package(const NNApplication *app, const NNNode *node);
bool nn_app_valid_output_handle(const NNPackage *package, const char *handle);
bool nn_app_valid_input_handle(const NNPackage *package, const char *handle);
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
