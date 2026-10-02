#include "catalog_internal.h"
#include "utils/utils.h"

#include <stdlib.h>
#include <string.h>
bool nn_catalog_parse_manifest(Package *p, yyjson_val *root,
                           const char *directory) {
  yyjson_val *sv, *deps, *entry, *inf;
  const char *s;
  if (!yyjson_is_obj(root) ||
      !yyjson_is_uint(sv = nn_catalog_get(root, "schemaVersion")) ||
      yyjson_get_uint(sv) != 1)
    return false;
  s = nn_catalog_strval(nn_catalog_get(root, "id"));
  if (!nn_catalog_valid_id(s) || !(p->pub.id = nn_text_copy(s)))
    return false;
  s = nn_catalog_strval(nn_catalog_get(root, "version"));
  {
    unsigned long a, b, c;
    if (!nn_catalog_semver(s, &a, &b, &c) || !(p->pub.version = nn_text_copy(s)))
      return false;
  }
  p->pub.directory = nn_text_copy(directory);
  if (!p->pub.directory)
    return false;
  deps = nn_catalog_get(root, "dependencies");
  if (!yyjson_is_obj(deps))
    return false;
  p->pub.dependency_count = yyjson_obj_size(deps);
  if (p->pub.dependency_count > CATALOG_ITEM_LIMIT)
    return false;
  p->dependencies =
      calloc(p->pub.dependency_count ? p->pub.dependency_count : 1,
             sizeof(*p->dependencies));
  if (!p->dependencies)
    return false;
  p->pub.dependencies = p->dependencies;
  {
    yyjson_obj_iter it = yyjson_obj_iter_with(deps);
    size_t i = 0;
    yyjson_val *key;
    while ((key = yyjson_obj_iter_next(&it))) {
      const char *id = yyjson_get_str(key),
                 *constraint = nn_catalog_strval(yyjson_obj_iter_get_val(key));
      unsigned long a, b, c;
      if (!nn_catalog_valid_id(id) || !constraint || !*constraint ||
          !(constraint[0] == '^' ? nn_catalog_semver(constraint + 1, &a, &b, &c)
                                 : nn_catalog_semver(constraint, &a, &b, &c)))
        return false;
      p->dependencies[i].id = nn_text_copy(id);
      p->dependencies[i].version_constraint = nn_text_copy(constraint);
      if (!p->dependencies[i].id || !p->dependencies[i].version_constraint)
        return false;
      ++i;
    }
  }
  entry = nn_catalog_get(root, "entrypoints");
  if (!yyjson_is_obj(entry))
    return false;
  s = nn_catalog_strval(nn_catalog_get(entry, "definition"));
  if (!s || !nn_catalog_safe_rel(s))
    return false;
  inf = nn_catalog_get(entry, "inference");
  if (!yyjson_is_obj(inf) ||
      strcmp(nn_catalog_strval(nn_catalog_get(inf, "language")) ? nn_catalog_strval(nn_catalog_get(inf, "language")) : "",
             "lua") != 0)
    return false;
  s = nn_catalog_strval(nn_catalog_get(inf, "file"));
  if (!s || !nn_catalog_safe_rel(s) || !(p->pub.inference_file = nn_text_copy(s)))
    return false;
  return true;
}
