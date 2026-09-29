#ifndef NN_CATALOG_H
#define NN_CATALOG_H

#include <stdbool.h>
#include <stddef.h>

typedef struct NNCatalog NNCatalog;

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const char *id;
    const char *version;
    const char *path;
} NNResourceRef;

typedef enum {
    NN_PARAMETER_NONE,
    NN_PARAMETER_BOOLEAN,
    NN_PARAMETER_INTEGER,
    NN_PARAMETER_NUMBER,
    NN_PARAMETER_STRING,
    NN_PARAMETER_JSON
} NNParameterValueType;

typedef struct {
    NNParameterValueType type;
    union { bool boolean; long long integer; double number; const char *string; } as;
} NNParameterValue;

typedef struct {
    const char *key;
    const char *type;
    bool has_minimum;
    double minimum;
    bool has_default;
    NNParameterValue default_value;
    const char *const *choices;
    size_t choice_count;
} NNParameterDef;

typedef struct {
    const char *id;
    const char *version_constraint;
} NNPackageDependency;

typedef struct NNPackage {
    const char *directory;
    const char *id;
    const char *version;
    const char *name;
    const char *description;
    const char *kind;
    const char *color;
    double width;
    double height;
    const NNParameterDef *parameters;
    size_t parameter_count;
    const NNPackageDependency *dependencies;
    size_t dependency_count;
    const char *inference_file;
} NNPackage;

NNCatalog *nn_catalog_load(const char *core_root, const char *project_root,
                           const NNResourceRef *custom, size_t custom_count,
                           char *err, size_t cap);
void nn_catalog_free(NNCatalog *catalog);
size_t nn_catalog_count(const NNCatalog *catalog);
const NNPackage *nn_catalog_at(const NNCatalog *catalog, size_t index);
const NNPackage *nn_catalog_find(const NNCatalog *catalog, const char *id,
                                 const char *version);

#ifdef __cplusplus
}
#endif

#endif
