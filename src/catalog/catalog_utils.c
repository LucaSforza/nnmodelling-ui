#define _XOPEN_SOURCE 700
#include "catalog_internal.h"
#include "utils/utils.h"

#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
bool nn_catalog_valid_id(const char *s) {
  if (!s || !*s)
    return false;
  for (; *s; ++s)
    if (!((*s >= 'a' && *s <= 'z') || (*s >= 'A' && *s <= 'Z') ||
          (*s >= '0' && *s <= '9') || *s == '.' || *s == '_' || *s == '-'))
      return false;
  return true;
}

bool nn_catalog_semver(const char *s, unsigned long *a, unsigned long *b,
                   unsigned long *c) {
  unsigned long *parts[3] = {a, b, c};
  size_t i;
  if (!s || !*s)
    return false;
  for (i = 0; i < 3; ++i) {
    const char *start = s;
    unsigned long value = 0;
    if (*s < '0' || *s > '9')
      return false;
    if (*s == '0' && s[1] >= '0' && s[1] <= '9')
      return false;
    while (*s >= '0' && *s <= '9') {
      unsigned digit = (unsigned)(*s++ - '0');
      if (value > (ULONG_MAX - digit) / 10)
        return false;
      value = value * 10 + digit;
    }
    if (s == start)
      return false;
    *parts[i] = value;
    if (i < 2) {
      if (*s++ != '.')
        return false;
    } else if (*s)
      return false;
  }
  return true;
}

bool nn_catalog_safe_rel(const char *s) {
  const char *p;
  if (!s || !*s || *s == '/' || strchr(s, '\\'))
    return false;
  p = s;
  while (*p) {
    const char *start = p;
    while (*p && *p != '/')
      ++p;
    if (p == start || (p - start == 1 && start[0] == '.') ||
        (p - start == 2 && start[0] == '.' && start[1] == '.'))
      return false;
    if (*p)
      ++p;
  }
  return true;
}

bool nn_catalog_checked_path(const char *base, const char *rel, bool want_dir,
                             char *out) {
  char current[PATH_MAX];
  const char *p;
  struct stat st;
  if (!base || !nn_catalog_safe_rel(rel) ||
      strlen(base) + strlen(rel) + 2 >= sizeof(current))
    return false;
  strcpy(current, base);
  p = rel;
  while (*p) {
    char *end = current + strlen(current);
    const char *start = p;
    size_t n;
    while (*p && *p != '/')
      ++p;
    n = (size_t)(p - start);
    *end++ = '/';
    memcpy(end, start, n);
    end[n] = '\0';
    if (lstat(current, &st) != 0 || S_ISLNK(st.st_mode))
      return false;
    if (*p && !S_ISDIR(st.st_mode))
      return false;
    if (*p)
      ++p;
  }
  if ((want_dir && !S_ISDIR(st.st_mode)) || (!want_dir && !S_ISREG(st.st_mode)))
    return false;
  if (!realpath(current, out))
    return false;
  return true;
}

bool nn_catalog_under_path(const char *root, const char *path) {
  size_t n = strlen(root);
  return strncmp(root, path, n) == 0 && path[n] == '/';
}

char *nn_catalog_read_file(const char *path, size_t *len) {
  FILE *f = fopen(path, "rb");
  long size;
  char *buf;
  if (!f)
    return NULL;
  if (fseek(f, 0, SEEK_END) || (size = ftell(f)) < 0 ||
      (unsigned long)size > CATALOG_FILE_LIMIT || fseek(f, 0, SEEK_SET)) {
    fclose(f);
    return NULL;
  }
  buf = malloc((size_t)size + 1);
  if (!buf) {
    fclose(f);
    return NULL;
  }
  if (fread(buf, 1, (size_t)size, f) != (size_t)size) {
    free(buf);
    fclose(f);
    return NULL;
  }
  buf[size] = '\0';
  fclose(f);
  *len = (size_t)size;
  return buf;
}

yyjson_val *nn_catalog_get(yyjson_val *v, const char *key) {
  return yyjson_obj_get(v, key);
}

const char *nn_catalog_strval(yyjson_val *v) {
  return yyjson_is_str(v) ? yyjson_get_str(v) : NULL;
}

bool nn_catalog_number(yyjson_val *v, double *out) {
  if (yyjson_is_num(v)) {
    *out = yyjson_get_num(v);
    return isfinite(*out);
  }
  return false;
}
