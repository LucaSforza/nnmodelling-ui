#ifndef NN_CATALOG_INTERNAL_H
#define NN_CATALOG_INTERNAL_H

#include "catalog/catalog.h"
#include "yyjson.h"

#define CATALOG_FILE_LIMIT (4u * 1024u * 1024u)
#define CATALOG_ITEM_LIMIT 4096u

typedef struct {
  NNPackage pub;
  NNParameterDef *parameters;
  char ***choice_storage;
  NNPackageDependency *dependencies;
  NNOutputDef *outputs;
} Package;

struct NNCatalog {
  Package *items;
  size_t count;
};

bool nn_catalog_valid_id(const char *s);
bool nn_catalog_semver(const char *s, unsigned long *a, unsigned long *b, unsigned long *c);
bool nn_catalog_safe_rel(const char *s);
bool nn_catalog_checked_path(const char *base, const char *rel, bool want_dir, char *out);
bool nn_catalog_under_path(const char *root, const char *path);
char *nn_catalog_read_file(const char *path, size_t *len);
yyjson_val *nn_catalog_get(yyjson_val *v, const char *key);
const char *nn_catalog_strval(yyjson_val *v);
bool nn_catalog_number(yyjson_val *v, double *out);
bool nn_catalog_parse_definition(Package *p, yyjson_val *root);
bool nn_catalog_parse_manifest(Package *p, yyjson_val *root, const char *directory);
bool nn_catalog_validate_dependencies(NNCatalog *cat, char *err, size_t cap);
bool nn_catalog_version_satisfies(const char *version, const char *constraint);

#endif
