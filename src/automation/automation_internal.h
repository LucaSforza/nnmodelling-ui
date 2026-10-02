#ifndef NN_AUTOMATION_INTERNAL_H
#define NN_AUTOMATION_INTERNAL_H

#include "automation.h"
#include "yyjson.h"

#define NN_AUTOMATION_REQUEST_LIMIT (2u * 1024u * 1024u)

typedef struct { yyjson_val *args; char error[512]; } NNAutomationQuery;

bool nn_automation_execute(NNApplication *app, const char *operation, NNAutomationQuery *query);
yyjson_mut_val *nn_automation_snapshot(yyjson_mut_doc *doc, NNApplication *app);
yyjson_mut_val *nn_automation_diagnostics(yyjson_mut_doc *doc, NNApplication *app,
                                         NNAutomationQuery *query);

#endif
