#include "automation_utils.h"

bool nn_automation_json_string(yyjson_mut_doc *doc, yyjson_mut_val *obj,
                                const char *key, const char *value)
{
    return yyjson_mut_obj_add_strcpy(doc, obj, key, value ? value : "");
}

yyjson_mut_val *nn_automation_identity(yyjson_mut_doc *doc, const char *id, const char *version)
{
    yyjson_mut_val *value = yyjson_mut_obj(doc);
    if (!value || !nn_automation_json_string(doc, value, "id", id) ||
        !nn_automation_json_string(doc, value, "version", version)) return NULL;
    return value;
}
