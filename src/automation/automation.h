#ifndef NN_AUTOMATION_H
#define NN_AUTOMATION_H

#include "application/application.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct NNAutomation NNAutomation;
/* Return malloc-owned JSON result or NULL and caller-owned error. */
typedef char *(*NNAutomationUiCallback)(void *user, const char *operation,
                                      const char *args_json, char *error, size_t cap);
NNAutomation *nn_automation_start(NNApplication *app, const char *path,
                                NNAutomationUiCallback callback, void *user,
                                char *error, size_t cap);
/* Nonblocking. Returns true if a successful command may have changed UI/model. */
bool nn_automation_poll(NNAutomation *service);
void nn_automation_stop(NNAutomation *service);
/* Pure synchronous dispatcher, owned JSON response. Useful to C tests. */
char *nn_automation_dispatch(NNApplication *app, const char *request,
                             NNAutomationUiCallback callback, void *user);

#ifdef __cplusplus
}
#endif
#endif
