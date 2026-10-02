#include "catalog_internal.h"
#include "utils/utils.h"

#include <stdlib.h>
#include <string.h>
static bool version_satisfies(const char *version, const char *constraint) {
  unsigned long a, b, c, x, y, z;
  const char *wanted = constraint;
  bool caret = *wanted == '^';
  if (caret)
    ++wanted;
  if (!nn_catalog_semver(version, &a, &b, &c) ||
      !nn_catalog_semver(wanted, &x, &y, &z))
    return false;
  if (!caret)
    return a == x && b == y && c == z;
  if (a != x)
    return false;
  if (a)
    return b > y || (b == y && c >= z);
  return b == y && c >= z;
}
bool nn_catalog_validate_dependencies(NNCatalog *cat, char *err, size_t cap) {
  size_t i, j;
  unsigned char *marks = calloc(cat->count ? cat->count : 1, 1);
  if (!marks)
    return nn_fail(err, cap, "out of memory validating dependencies");
  for (i = 0; i < cat->count; ++i)
    for (j = i + 1; j < cat->count; ++j)
      if (!strcmp(cat->items[i].pub.id, cat->items[j].pub.id)) {
        nn_errorf(err, cap, "duplicate package identity: %s",
                  cat->items[i].pub.id);
        free(marks);
        return false;
      }
  for (i = 0; i < cat->count; ++i)
    for (j = 0; j < cat->items[i].pub.dependency_count; ++j) {
      NNPackageDependency *dep = &cat->items[i].dependencies[j];
      size_t k, matches = 0;
      for (k = 0; k < cat->count; ++k)
        if (!strcmp(cat->items[k].pub.id, dep->id) &&
            version_satisfies(cat->items[k].pub.version,
                              dep->version_constraint))
          ++matches;
      if (matches != 1) {
        nn_errorf(err, cap, "dependency has no unique provider: %s", dep->id);
        free(marks);
        return false;
      }
    }
  /* Three-colour DFS, package count is bounded above. */
  for (i = 0; i < cat->count; ++i) {
    size_t stack[CATALOG_ITEM_LIMIT], edge[CATALOG_ITEM_LIMIT], depth = 0;
    if (marks[i])
      continue;
    stack[depth] = i;
    edge[depth++] = 0;
    marks[i] = 1;
    while (depth) {
      size_t at = stack[depth - 1];
      Package *p = &cat->items[at];
      if (edge[depth - 1] == p->pub.dependency_count) {
        marks[at] = 2;
        --depth;
        continue;
      }
      {
        const char *dep_id = p->dependencies[edge[depth - 1]++].id;
        size_t k;
        for (k = 0; k < cat->count && strcmp(cat->items[k].pub.id, dep_id);
             ++k) {
        }
        if (marks[k] == 1) {
          nn_errorf(err, cap, "dependency cycle at package: %s", dep_id);
          free(marks);
          return false;
        }
        if (!marks[k]) {
          marks[k] = 1;
          stack[depth] = k;
          edge[depth++] = 0;
        }
      }
    }
  }
  free(marks);
  return true;
}
