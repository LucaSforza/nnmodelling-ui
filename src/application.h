#ifndef NN_APPLICATION_H
#define NN_APPLICATION_H

#include "project.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct NNApplication NNApplication;

NNApplication *nn_app_new(const char *core_root);
void nn_app_free(NNApplication *app);
bool nn_app_open(NNApplication *app, const char *directory, char *error, size_t cap);
bool nn_app_create(NNApplication *app, const char *parent, const char *id,
                   const char *name, bool mnist, char *error, size_t cap);
bool nn_app_create_stereotype(NNApplication *app, const char *id, const char *version,
                              const char *definition_json, const char *inference_lua,
                              const char *dependencies_json, char *error, size_t cap);
bool nn_app_create_dataset(NNApplication *app, const char *id, const char *version,
                           const char *definition_json, bool select,
                           char *error, size_t cap);
bool nn_app_select_dataset(NNApplication *app, const char *id, const char *version,
                           char *error, size_t cap);
bool nn_app_create_vae(NNApplication *app, const char *parent, const char *id,
                      const char *name, char *error, size_t cap);
bool nn_app_save(NNApplication *app, char *error, size_t cap);
bool nn_app_close(NNApplication *app, bool discard, char *error, size_t cap);
/* Borrowed read-only state; do not retain pointers across mutations. */
const NNProject *nn_app_project(const NNApplication *app);
const NNModel *nn_app_model(const NNApplication *app);
bool nn_app_add_node(NNApplication *app, const char *id, const char *package_id,
                     const char *version, const char *scope, double x, double y,
                     char *error, size_t cap);
bool nn_app_remove_node(NNApplication *app, const char *id, char *error, size_t cap);
bool nn_app_move_node(NNApplication *app, const char *id, double x, double y,
                      char *error, size_t cap);
bool nn_app_rename_node(NNApplication *app, const char *id, const char *label,
                        char *error, size_t cap);
bool nn_app_connect(NNApplication *app, const char *id, const char *source,
                    const char *source_handle, const char *target,
                    const char *target_handle, char *error, size_t cap);
bool nn_app_disconnect(NNApplication *app, const char *id, char *error, size_t cap);
bool nn_app_set_parameter(NNApplication *app, const char *node, const char *key,
                          const NNValue *value, char *error, size_t cap);
bool nn_app_set_parameter_text(NNApplication *app, const char *node, const char *key,
                               const char *text, char *error, size_t cap);
/* Returned UTF-8 text is owned; release with nn_app_free_text. */
char *nn_app_parameter_text(const NNApplication *app, const char *node, const char *key);
void nn_app_free_text(char *text);
size_t nn_app_port_count(const NNApplication *app, const char *node, bool output);
bool nn_app_port_id(const NNApplication *app, const char *node, bool output,
                    size_t index, char *buffer, size_t capacity);
bool nn_app_node_is_subflow(const NNApplication *app, const char *node);

#ifdef __cplusplus
}
#endif
#endif
