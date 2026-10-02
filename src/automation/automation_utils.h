#ifndef NN_AUTOMATION_UTILS_H
#define NN_AUTOMATION_UTILS_H

#include "yyjson.h"

bool nn_automation_json_string(yyjson_mut_doc *doc, yyjson_mut_val *object,
                                const char *key, const char *value);
yyjson_mut_val *nn_automation_identity(yyjson_mut_doc *doc, const char *id,
                                      const char *version);

#endif
