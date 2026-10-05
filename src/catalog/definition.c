#include "catalog_internal.h"
#include "utils/utils.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
bool nn_catalog_parse_definition(Package *p, yyjson_val *root) {
  yyjson_val *view, *params, *v, *outputs;
  const char *s;
  double n;
  if (!yyjson_is_obj(root) || !(s = nn_catalog_strval(nn_catalog_get(root, "name"))) || !*s ||
      !(p->pub.name = nn_text_copy(s)) || !(s = nn_catalog_strval(nn_catalog_get(root, "kind"))) ||
      !*s || !(p->pub.kind = nn_text_copy(s)))
    return false;
  outputs = nn_catalog_get(root, "outputs");
  bool terminal = !strcmp(p->pub.kind, "output") ||
                  !strcmp(p->pub.kind, "loss-output");
  if (outputs && !yyjson_is_arr(outputs))
    return false;
  if (terminal && outputs && yyjson_arr_size(outputs) != 0)
    return false;
  if (!terminal) {
    size_t count = outputs ? yyjson_arr_size(outputs) : 1;
    if (!count || count > 2)
      return false;
    p->outputs = calloc(count, sizeof(*p->outputs));
    if (!p->outputs)
      return false;
    p->pub.output_count = count;
    p->pub.outputs = p->outputs;
    for (size_t i = 0; i < count; ++i) {
      const char *id, *type;
      if (outputs) {
        yyjson_val *item = yyjson_arr_get(outputs, i);
        bool have_id = false, have_type = false;
        size_t id_length = 0, type_length = 0;
        if (!yyjson_is_obj(item) || yyjson_obj_size(item) != 2)
          return false;
        yyjson_obj_iter fields = yyjson_obj_iter_with(item);
        yyjson_val *key;
        id = type = NULL;
        while ((key = yyjson_obj_iter_next(&fields))) {
          const char *field = yyjson_get_str(key);
          size_t field_length = yyjson_get_len(key);
          yyjson_val *value = yyjson_obj_iter_get_val(key);
          if (!field || strlen(field) != field_length)
            return false;
          if (!strcmp(field, "id") && !have_id) {
            have_id = true;
            id = nn_catalog_strval(value);
            if (id) id_length = yyjson_get_len(value);
          } else if (!strcmp(field, "type") && !have_type) {
            have_type = true;
            type = nn_catalog_strval(value);
            if (type) type_length = yyjson_get_len(value);
          } else return false;
        }
        if (!have_id || !have_type)
          return false;
        if (!id || !type || strlen(id) != id_length || strlen(type) != type_length)
          return false;
      } else {
        id = !strcmp(p->pub.kind, "loss") ? "loss" : "out";
        type = !strcmp(p->pub.kind, "loss") ? "loss" : "output";
      }
      if (!nn_catalog_valid_id(id) || !type || (strcmp(type, "output") && strcmp(type, "loss")))
        return false;
      for (size_t j = 0; j < i; ++j)
        if (!strcmp(p->outputs[j].id, id) || !strcmp(p->outputs[j].type, type))
          return false;
      p->outputs[i].id = nn_text_copy(id);
      p->outputs[i].type = nn_text_copy(type);
      if (!p->outputs[i].id || !p->outputs[i].type)
        return false;
    }
  }
  s = nn_catalog_strval(nn_catalog_get(root, "description"));
  if (s && !(p->pub.description = nn_text_copy(s)))
    return false;
  view = nn_catalog_get(root, "view");
  if (!yyjson_is_obj(view))
    return false;
  s = nn_catalog_strval(nn_catalog_get(view, "color"));
  if (!s || !*s || !(p->pub.color = nn_text_copy(s)))
    return false;
  if (!(v = nn_catalog_get(view, "width")) || !nn_catalog_number(v, &n) || n <= 0)
    return false;
  p->pub.width = n;
  if (!(v = nn_catalog_get(view, "height")) || !nn_catalog_number(v, &n) || n <= 0)
    return false;
  p->pub.height = n;
  params = nn_catalog_get(root, "parameters");
  if (!yyjson_is_obj(params))
    return false;
  p->pub.parameter_count = yyjson_obj_size(params);
  if (p->pub.parameter_count > CATALOG_ITEM_LIMIT)
    return false;
  p->parameters = calloc(p->pub.parameter_count ? p->pub.parameter_count : 1,
                         sizeof(*p->parameters));
  p->choice_storage = calloc(
      p->pub.parameter_count ? p->pub.parameter_count : 1, sizeof(char **));
  if (!p->parameters || !p->choice_storage)
    return false;
  p->pub.parameters = p->parameters;
  {
    yyjson_obj_iter it = yyjson_obj_iter_with(params);
    size_t i = 0;
    while ((v = yyjson_obj_iter_next(&it))) {
      yyjson_val *val = yyjson_obj_iter_get_val(v);
      NNParameterDef *d = &p->parameters[i];
      const char *key = yyjson_get_str(v);
      d->key = nn_text_copy(key);
      d->type = nn_text_copy(nn_catalog_strval(nn_catalog_get(val, "type")));
      if (!d->key || !*d->key || !d->type ||
           (strcmp(d->type, "boolean") && strcmp(d->type, "integer") &&
            strcmp(d->type, "number") && strcmp(d->type, "string") &&
           strcmp(d->type, "dtype") && strcmp(d->type, "json") &&
           strcmp(d->type, "stereotype")))
        return false;
      for (size_t previous = 0; previous < i; ++previous)
        if (!strcmp(p->parameters[previous].key, d->key))
          return false;
      {
        const char *position = nn_catalog_strval(nn_catalog_get(val, "position"));
        if (nn_catalog_get(val, "position")) {
          if (!position ||
              (strcmp(position, "top") && strcmp(position, "bottom")))
            return false;
          d->position = nn_text_copy(position);
          if (!d->position)
            return false;
        }
      }
      if (nn_catalog_get(val, "kind")) {
        const char *kind = nn_catalog_strval(nn_catalog_get(val, "kind"));
        if (!kind || !*kind || strcmp(d->type, "stereotype") ||
            !(d->kind = nn_text_copy(kind))) return false;
      }
      if (nn_catalog_get(val, "minimum")) {
        if (!nn_catalog_number(nn_catalog_get(val, "minimum"), &d->minimum) ||
            (strcmp(d->type, "integer") && strcmp(d->type, "number")))
          return false;
        d->has_minimum = true;
      }
      {
        yyjson_val *dv = nn_catalog_get(val, "default");
        if (dv) {
          d->has_default = true;
          if (yyjson_is_bool(dv)) {
            d->default_value.type = NN_PARAMETER_BOOLEAN;
            d->default_value.as.boolean = yyjson_get_bool(dv);
          } else if (!strcmp(d->type, "number") && yyjson_is_num(dv)) {
            d->default_value.type = NN_PARAMETER_NUMBER;
            d->default_value.as.number = yyjson_get_num(dv);
          } else if (!strcmp(d->type, "integer") && yyjson_is_int(dv)) {
            d->default_value.type = NN_PARAMETER_INTEGER;
            d->default_value.as.integer = yyjson_get_sint(dv);
          } else if (yyjson_is_num(dv)) {
            d->default_value.type = NN_PARAMETER_NUMBER;
            d->default_value.as.number = yyjson_get_num(dv);
          } else if (yyjson_is_str(dv)) {
            d->default_value.type = NN_PARAMETER_STRING;
            d->default_value.as.string = nn_text_copy(yyjson_get_str(dv));
          } else if (yyjson_is_arr(dv) || yyjson_is_obj(dv)) {
            d->default_value.type = NN_PARAMETER_JSON;
            d->default_value.as.string =
                yyjson_val_write_opts(dv, 0, NULL, NULL, NULL);
          } else
            return false;
          if ((d->default_value.type == NN_PARAMETER_STRING ||
               d->default_value.type == NN_PARAMETER_JSON) &&
              !d->default_value.as.string)
            return false;
        }
      }
      {
        yyjson_val *choices = nn_catalog_get(val, "choices");
        if (choices) {
          size_t j, count = yyjson_arr_size(choices);
          if (!yyjson_is_arr(choices) || count > CATALOG_ITEM_LIMIT ||
              (strcmp(d->type, "string") && strcmp(d->type, "dtype")))
            return false;
          p->choice_storage[i] = calloc(count + 1, sizeof(char *));
          if (!p->choice_storage[i])
            return false;
          for (j = 0; j < count; ++j) {
            const char *choice = nn_catalog_strval(yyjson_arr_get(choices, j));
            if (!choice || !(p->choice_storage[i][j] = nn_text_copy(choice)))
              return false;
          }
          d->choices = (const char *const *)p->choice_storage[i];
          d->choice_count = count;
        }
      }
      ++i;
    }
  }
  return true;
}
