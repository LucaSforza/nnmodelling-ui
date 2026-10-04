#include "catalog/catalog.h"
#include "utils/utils.h"

#include <string.h>

bool nn_catalog_package_is_kind(const NNPackage *package, const char *kind)
{
    return package && package->kind && kind && !strcmp(package->kind, kind);
}

const NNParameterDef *nn_catalog_package_parameter(const NNPackage *package,
                                                   const char *key)
{
    if (!package || !key || !package->parameters)
        return NULL;
    for (size_t i = 0; i < package->parameter_count; i++)
        if (package->parameters[i].key &&
            !strcmp(package->parameters[i].key, key))
            return &package->parameters[i];
    return NULL;
}

const NNOutputDef *nn_catalog_package_output(const NNPackage *package,
                                             const char *handle)
{
    if (!package || !handle || !package->outputs)
        return NULL;
    for (size_t i = 0; i < package->output_count; i++)
        if (package->outputs[i].id && !strcmp(package->outputs[i].id, handle))
            return &package->outputs[i];
    return NULL;
}

bool nn_catalog_package_input_handle_valid(const NNPackage *package,
                                           const char *handle)
{
    if (!package || !handle || nn_catalog_package_is_kind(package, "input"))
        return false;
    if (nn_catalog_package_is_kind(package, "join")) {
        size_t order = 0;
        return nn_join_handle_order(handle, &order);
    }
    return !strcmp(handle, "in");
}
