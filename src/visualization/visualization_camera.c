#include "visualization_internal.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static NN3DVec add(NN3DVec a, NN3DVec b) {
  return (NN3DVec){a.x + b.x, a.y + b.y, a.z + b.z};
}
static NN3DVec sub(NN3DVec a, NN3DVec b) {
  return (NN3DVec){a.x - b.x, a.y - b.y, a.z - b.z};
}
static NN3DVec scale(NN3DVec a, double n) {
  return (NN3DVec){a.x * n, a.y * n, a.z * n};
}
static double dot(NN3DVec a, NN3DVec b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
static NN3DVec cross(NN3DVec a, NN3DVec b) {
  return (NN3DVec){a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
                   a.x * b.y - a.y * b.x};
}
double nn_3d_clamp(double value, double minimum, double maximum) {
  return value < minimum ? minimum : value > maximum ? maximum : value;
}

static NN3DVec forward(const NN3DCamera *camera) {
  double cp = cos(camera->pitch);
  return (NN3DVec){sin(camera->yaw) * cp, sin(camera->pitch),
                   cos(camera->yaw) * cp};
}
typedef struct {
  double x, y, z;
} Projected;
static Projected project(const NN3DCamera *camera, NN3DVec point, double width,
                         double height);

void nn_3d_camera_fit(const NN3DScene *scene, NN3DCamera *camera,
                      double aspect) {
  if (!scene || !camera)
    return;
  if (!isfinite(aspect) || aspect <= 0.0)
    aspect = 1.0;
  NN3DVec low = {INFINITY, INFINITY, INFINITY},
          high = {-INFINITY, -INFINITY, -INFINITY};
  bool have_bounds = false;
  if (scene->node_count || scene->group_count) {
    if (!scene->node_count) {
      low = (NN3DVec){0, 0, 0};
      high = low;
    }
    const NN3DNode *n = &scene->nodes[0];
    if (scene->node_count) {
      low = sub(n->center, scale(n->size, 0.5));
      high = add(n->center, scale(n->size, 0.5));
      have_bounds = true;
    }
    for (size_t i = 1; i < scene->node_count; ++i) {
      n = &scene->nodes[i];
      NN3DVec a = sub(n->center, scale(n->size, 0.5));
      NN3DVec b = add(n->center, scale(n->size, 0.5));
      low.x = fmin(low.x, a.x);
      low.y = fmin(low.y, a.y);
      low.z = fmin(low.z, a.z);
      high.x = fmax(high.x, b.x);
      high.y = fmax(high.y, b.y);
      high.z = fmax(high.z, b.z);
    }
    for (size_t i = 0; i < scene->group_count; i++) {
      const NN3DGroup *g = &scene->groups[i];
      if (g->maximum.x <= g->minimum.x || g->maximum.y <= g->minimum.y ||
          g->maximum.z <= g->minimum.z)
        continue;
      low.x = fmin(low.x, g->minimum.x);
      low.y = fmin(low.y, g->minimum.y);
      low.z = fmin(low.z, g->minimum.z);
      high.x = fmax(high.x, g->maximum.x);
      high.y = fmax(high.y, g->maximum.y);
      high.z = fmax(high.z, g->maximum.z);
      have_bounds = true;
    }
  }
  if (!have_bounds) {
    camera->position = (NN3DVec){0, 0, -500};
    camera->yaw = 0;
    camera->pitch = 0;
    return;
  }
  NN3DVec center = scale(add(low, high), 0.5);
  camera->yaw = 0.72;
  camera->pitch = 0.42;
  double width = fmax(320.0, aspect * 600.0), height = 600.0;
  double margin = 36.0;
  NN3DVec corners[8];
  for (size_t i = 0; i < 8; i++)
    corners[i] = (NN3DVec){(i & 1) ? high.x : low.x, (i & 2) ? high.y : low.y,
                           (i & 4) ? high.z : low.z};
  double distance = 100.0;
  NN3DVec f = forward(camera);
  for (;;) {
    camera->position = sub(center, scale(f, distance));
    bool fits = true;
    for (size_t i = 0; i < 8; i++) {
      Projected p = project(camera, corners[i], width, height);
      if (!isfinite(p.x) || p.x < margin || p.x > width - margin ||
          p.y < margin || p.y > height - margin) {
        fits = false;
        break;
      }
    }
    if (fits || distance > 1e9)
      break;
    distance *= 1.5;
  }
  double high_distance = distance, low_distance = distance / 1.5;
  for (size_t step = 0; step < 32; step++) {
    double middle = (low_distance + high_distance) * 0.5;
    camera->position = sub(center, scale(f, middle));
    bool fits = true;
    for (size_t i = 0; i < 8; i++) {
      Projected p = project(camera, corners[i], width, height);
      if (!isfinite(p.x) || p.x < margin || p.x > width - margin ||
          p.y < margin || p.y > height - margin) {
        fits = false;
        break;
      }
    }
    if (fits)
      high_distance = middle;
    else
      low_distance = middle;
  }
  camera->position = sub(center, scale(f, high_distance));
}

void nn_3d_camera_home(const NN3DScene *scene, NN3DCamera *camera,
                       double aspect) {
  if (!scene || !camera)
    return;
  NN3DVec center = {0, 0, 0};
  if (scene->node_count) {
    NN3DVec low = {INFINITY, INFINITY, INFINITY},
            high = {-INFINITY, -INFINITY, -INFINITY};
    for (size_t i = 0; i < scene->node_count; i++) {
      const NN3DNode *n = &scene->nodes[i];
      NN3DVec h = scale(n->size, 0.5), a = sub(n->center, h),
              b = add(n->center, h);
      low.x = fmin(low.x, a.x);
      low.y = fmin(low.y, a.y);
      low.z = fmin(low.z, a.z);
      high.x = fmax(high.x, b.x);
      high.y = fmax(high.y, b.y);
      high.z = fmax(high.z, b.z);
    }
    center = scale(add(low, high), 0.5);
    for (size_t i = 0; i < scene->node_count; i++)
      if (scene->nodes[i].group == SIZE_MAX) {
        center = scene->nodes[i].center;
        break;
      }
  }
  camera->yaw = 1.15;
  camera->pitch = 0.24;
  double span =
      fmax(320.0, scene->node_count ? scene->nodes[0].size.x * 3.0 : 320.0);
  camera->position = sub(center, scale(forward(camera), span));
  (void)aspect;
}

void nn_3d_camera_look(NN3DCamera *camera, double yaw_delta,
                       double pitch_delta) {
  if (!camera || !isfinite(yaw_delta) || !isfinite(pitch_delta))
    return;
  camera->yaw = remainder(camera->yaw + yaw_delta, 6.28318530717958647692);
  camera->pitch = nn_3d_clamp(camera->pitch + pitch_delta, -1.52, 1.52);
}

void nn_3d_camera_move(NN3DCamera *camera, double right, double up,
                       double forward_delta) {
  if (!camera)
    return;
  if (!isfinite(right) || !isfinite(up) || !isfinite(forward_delta))
    return;
  NN3DVec f = forward(camera);
  NN3DVec world_up = {0, 1, 0};
  NN3DVec r = cross(f, world_up);
  double length = sqrt(dot(r, r));
  if (length > 1e-9)
    r = scale(r, 1.0 / length);
  camera->position =
      add(camera->position, add(add(scale(r, right), scale(world_up, up)),
                                scale(f, forward_delta)));
}

static Projected project(const NN3DCamera *camera, NN3DVec point, double width,
                         double height) {
  NN3DVec f = forward(camera), world_up = {0, 1, 0};
  NN3DVec right = cross(f, world_up);
  double rl = sqrt(dot(right, right));
  if (rl < 1e-9)
    right = (NN3DVec){1, 0, 0};
  else
    right = scale(right, 1.0 / rl);
  NN3DVec up = cross(right, f);
  NN3DVec rel = sub(point, camera->position);
  double z = dot(rel, f);
  double focal = fmax(1.0, fmin(width, height) * 0.92);
  if (z <= 0.1)
    return (Projected){NAN, NAN, z};
  return (Projected){width * 0.5 + focal * dot(rel, right) / z,
                     height * 0.5 - focal * dot(rel, up) / z, z};
}

typedef struct {
  NN3DPrimitive *items;
  size_t count, capacity;
} Builder;
static bool push(Builder *b, NN3DPrimitive p) {
  if (b->count == b->capacity) {
    size_t cap = b->capacity ? b->capacity * 2 : 128;
    if (cap < b->capacity || cap > SIZE_MAX / sizeof(*b->items))
      return false;
    NN3DPrimitive *items = realloc(b->items, cap * sizeof(*items));
    if (!items)
      return false;
    b->items = items;
    b->capacity = cap;
  }
  b->items[b->count++] = p;
  return true;
}
static bool face(Builder *b, const NN3DCamera *camera, const NN3DVec corners[4],
                 double width, double height, uint32_t color, size_t node) {
  NN3DVec clipped[5];
  size_t count = 0;
  for (size_t i = 0; i < 4; i++) {
    NN3DVec a = corners[i], c = corners[(i + 1) % 4];
    Projected pa = project(camera, a, width, height),
              pc = project(camera, c, width, height);
    bool ina = pa.z > 0.1, inc = pc.z > 0.1;
    if (ina)
      clipped[count++] = a;
    if (ina != inc) {
      double t = (0.100001 - pa.z) / (pc.z - pa.z);
      clipped[count++] = add(a, scale(sub(c, a), t));
    }
  }
  if (count < 3)
    return true;
  for (size_t i = 1; i + 1 < count; i++) {
    NN3DVec tri[3] = {clipped[0], clipped[i], clipped[i + 1]};
    NN3DPrimitive p = {0};
    p.kind = NN_3D_FACE;
    p.rgba = color;
    p.node = node;
    for (size_t j = 0; j < 3; j++) {
      Projected q = project(camera, tri[j], width, height);
      p.x[j] = q.x;
      p.y[j] = q.y;
      p.depth += q.z / 3.0;
    }
    p.x[3] = p.x[2];
    p.y[3] = p.y[2];
    if (!push(b, p))
      return false;
  }
  return true;
}
static bool line(Builder *b, const NN3DCamera *camera, NN3DVec a, NN3DVec c,
                 double width, double height, uint32_t color, size_t node,
                 const char *text) {
  NN3DPrimitive p = {0};
  p.kind = text ? NN_3D_LABEL : NN_3D_LINE;
  p.rgba = color;
  p.node = node;
  p.text = text;
  Projected pa = project(camera, a, width, height),
            pc = project(camera, c, width, height);
  if (pa.z <= 0.1 && pc.z <= 0.1)
    return true;
  if (pa.z <= 0.1 || pc.z <= 0.1) {
    double t = (0.100001 - pa.z) / (pc.z - pa.z);
    NN3DVec clipped = add(a, scale(sub(c, a), t));
    if (pa.z <= 0.1)
      a = clipped;
    else
      c = clipped;
    pa = project(camera, a, width, height);
    pc = project(camera, c, width, height);
  }
  p.x[0] = pa.x;
  p.y[0] = pa.y;
  p.x[1] = pc.x;
  p.y[1] = pc.y;
  p.x[2] = pc.x;
  p.y[2] = pc.y;
  p.x[3] = pc.x;
  p.y[3] = pc.y;
  p.depth = (pa.z + pc.z) * 0.5;
  return push(b, p);
}
static int depth_compare(const void *a, const void *b) {
  const NN3DPrimitive *x = a, *y = b;
  return x->depth < y->depth ? 1 : x->depth > y->depth ? -1 : 0;
}
static uint32_t shade(uint32_t color, double factor) {
  uint32_t r = (uint32_t)(((color >> 16) & 255u) * factor);
  uint32_t g = (uint32_t)(((color >> 8) & 255u) * factor);
  uint32_t b = (uint32_t)((color & 255u) * factor);
  return (color & 0xff000000u) | (r << 16) | (g << 8) | b;
}

bool nn_3d_frame(const NN3DScene *scene, const NN3DCamera *camera, double width,
                 double height, NN3DFrame *frame) {
  if (!frame)
    return false;
  frame->items = NULL;
  frame->count = 0;
  if (!scene || !camera || !isfinite(width) || !isfinite(height) ||
      width <= 0 || height <= 0)
    return false;
  Builder b = {0};
  bool ok = true;
  for (size_t i = 0; i < scene->node_count && ok; i++) {
    const NN3DNode *n = &scene->nodes[i];
    NN3DVec h = scale(n->size, 0.5);
    NN3DVec p[8] = {{n->center.x - h.x, n->center.y - h.y, n->center.z - h.z},
                    {n->center.x + h.x, n->center.y - h.y, n->center.z - h.z},
                    {n->center.x + h.x, n->center.y + h.y, n->center.z - h.z},
                    {n->center.x - h.x, n->center.y + h.y, n->center.z - h.z},
                    {n->center.x - h.x, n->center.y - h.y, n->center.z + h.z},
                    {n->center.x + h.x, n->center.y - h.y, n->center.z + h.z},
                    {n->center.x + h.x, n->center.y + h.y, n->center.z + h.z},
                    {n->center.x - h.x, n->center.y + h.y, n->center.z + h.z}};
    uint32_t color = 0xff7b9fc8u;
    if (n->color && n->color[0] == '#') {
      char *end = NULL;
      unsigned long v = strtoul(n->color + 1, &end, 16);
      if (end && *end == '\0' && end != n->color + 1)
        color = 0xff000000u | (uint32_t)(v & 0xffffffu);
    }
    ok = face(&b, camera, (NN3DVec[]){p[4], p[5], p[6], p[7]}, width, height,
              shade(color, 0.75), i) &&
         face(&b, camera, (NN3DVec[]){p[0], p[1], p[5], p[4]}, width, height,
              shade(color, 0.50), i) &&
         face(&b, camera, (NN3DVec[]){p[1], p[2], p[6], p[5]}, width, height,
              shade(color, 0.58), i) &&
         face(&b, camera, (NN3DVec[]){p[2], p[3], p[7], p[6]}, width, height,
              shade(color, 0.64), i) &&
         face(&b, camera, (NN3DVec[]){p[3], p[0], p[4], p[7]}, width, height,
              shade(color, 0.7), i) &&
         face(&b, camera, (NN3DVec[]){p[0], p[1], p[2], p[3]}, width, height,
              color, i);
    if (ok) {
      Projected q = project(camera, n->center, width, height);
      Projected left = project(
          camera,
          (NN3DVec){n->center.x - n->size.x * 0.5, n->center.y, n->center.z},
          width, height);
      Projected right = project(
          camera,
          (NN3DVec){n->center.x + n->size.x * 0.5, n->center.y, n->center.z},
          width, height);
      Projected top = project(
          camera,
          (NN3DVec){n->center.x, n->center.y + n->size.y * 0.5, n->center.z},
          width, height);
      Projected bottom = project(
          camera,
          (NN3DVec){n->center.x, n->center.y - n->size.y * 0.5, n->center.z},
          width, height);
      double card_width = fabs(right.x - left.x),
             card_height = fabs(bottom.y - top.y);
      bool visible =
          q.z > 0.1 && q.x >= 0 && q.x <= width && q.y >= 0 && q.y <= height;
      if (visible && card_width >= 55.0 && card_height >= 26.0) {
        NN3DPrimitive t = {0};
        t.kind = NN_3D_LABEL;
        t.rgba = 0xff101820u;
        t.node = i;
        t.text = n->label;
        t.x[0] = q.x;
        t.y[0] = q.y;
        t.depth = q.z - 0.01;
        ok = push(&b, t);
      }
      if (ok && visible && card_width >= 130.0 && card_height >= 60.0 &&
          n->parameters && *n->parameters) {
        NN3DVec below = {n->center.x, n->center.y - n->size.y * 0.28,
                         n->center.z + n->size.z * 0.51};
        Projected qp = project(camera, below, width, height);
        if (qp.z > 0.1 && qp.x >= 0 && qp.x <= width && qp.y >= 0 &&
            qp.y <= height) {
          NN3DPrimitive params = {0};
          params.kind = NN_3D_LABEL;
          params.rgba = 0xff18212au;
          params.node = i;
          params.text = n->parameters;
          params.x[0] = qp.x;
          params.y[0] = qp.y;
          params.depth = qp.z - 0.005;
          ok = push(&b, params);
        }
      }
    }
  }
  for (size_t i = 0; i < scene->edge_count && ok; i++) {
    const NN3DEdge *e = &scene->edges[i];
    if (e->source >= scene->node_count || e->target >= scene->node_count)
      continue;
    const NN3DNode *a = &scene->nodes[e->source], *c = &scene->nodes[e->target];
    uint32_t color = e->loss ? 0xffe04d4du : 0xff252525u;
    NN3DVec start = {a->center.x, a->center.y + a->size.y * 0.5, a->center.z};
    NN3DVec end = {c->center.x, c->center.y - c->size.y * 0.5, c->center.z};
    ok = line(&b, camera, start, end, width, height, color, SIZE_MAX, NULL);
    if (ok) {
      NN3DVec d = sub(end, start);
      double len = sqrt(dot(d, d));
      if (len > 1e-9) {
        d = scale(d, 1.0 / len);
        NN3DVec side = {-d.z, 0, d.x};
        NN3DVec base = sub(end, scale(d, 12.0));
        ok = line(&b, camera, end, add(base, scale(side, 5.0)), width, height,
                  color, SIZE_MAX, NULL) &&
             line(&b, camera, end, sub(base, scale(side, 5.0)), width, height,
                  color, SIZE_MAX, NULL);
      }
    }
    if (ok && e->source_handle && *e->source_handle) {
      NN3DVec middle = scale(add(start, end), 0.5);
      Projected q = project(camera, middle, width, height);
      Projected ps = project(camera, start, width, height),
                pe = project(camera, end, width, height);
      if (q.z > 0.1 && ps.z > 0.1 && pe.z > 0.1 &&
          hypot(pe.x - ps.x, pe.y - ps.y) > 65.0) {
        NN3DPrimitive label = {0};
        label.kind = NN_3D_LABEL;
        label.rgba = color;
        label.node = SIZE_MAX;
        label.text = e->source_handle;
        label.x[0] = q.x;
        label.y[0] = q.y;
        label.depth = q.z - 0.002;
        ok = push(&b, label);
      }
    }
  }
  static const unsigned char box_edges[12][2] = {
      {0, 1}, {1, 2}, {2, 3}, {3, 0}, {4, 5}, {5, 6},
      {6, 7}, {7, 4}, {0, 4}, {1, 5}, {2, 6}, {3, 7}};
  for (size_t g = 0; g < scene->group_count && ok; g++) {
    const NN3DGroup *group = &scene->groups[g];
    NN3DVec a = group->minimum, c = group->maximum;
    if (c.x <= a.x || c.y <= a.y || c.z <= a.z)
      continue;
    NN3DVec p[8] = {{a.x, a.y, a.z}, {c.x, a.y, a.z}, {c.x, c.y, a.z},
                    {a.x, c.y, a.z}, {a.x, a.y, c.z}, {c.x, a.y, c.z},
                    {c.x, c.y, c.z}, {a.x, c.y, c.z}};
    for (size_t e = 0; e < 12 && ok; e++)
      ok = line(&b, camera, p[box_edges[e][0]], p[box_edges[e][1]], width,
                height, 0xff537ba8u, SIZE_MAX, NULL);
    if (ok) {
      NN3DVec anchor = {a.x, c.y + 16.0, a.z};
      Projected q = project(camera, anchor, width, height);
      Projected p0 = project(camera, a, width, height),
                p1 = project(camera, c, width, height);
      if (q.z > 0.1 && q.x >= 0 && q.x <= width && q.y >= 0 && q.y <= height &&
          p0.z > 0.1 && p1.z > 0.1 && fabs(p1.x - p0.x) > 90.0 &&
          fabs(p1.y - p0.y) > 35.0) {
        NN3DPrimitive label = {0};
        label.kind = NN_3D_LABEL;
        label.rgba = 0xff355f8bu;
        label.node = SIZE_MAX;
        label.text = group->label;
        label.x[0] = q.x;
        label.y[0] = q.y;
        label.depth = q.z - 0.01;
        ok = push(&b, label);
      }
    }
  }
  if (!ok) {
    free(b.items);
    return false;
  }
  qsort(b.items, b.count, sizeof(*b.items), depth_compare);
  frame->items = b.items;
  frame->count = b.count;
  return true;
}

void nn_3d_frame_dispose(NN3DFrame *frame) {
  if (!frame)
    return;
  free(frame->items);
  frame->items = NULL;
  frame->count = 0;
}

size_t nn_3d_pick(const NN3DFrame *frame, double x, double y) {
  if (!frame || !isfinite(x) || !isfinite(y))
    return SIZE_MAX;
  for (size_t i = frame->count; i > 0; i--) {
    const NN3DPrimitive *p = &frame->items[i - 1];
    if (p->kind != NN_3D_FACE || p->node == SIZE_MAX)
      continue;
    bool inside = false;
    for (size_t a = 0, b = 3; a < 4; b = a++) {
      double ax = p->x[a], ay = p->y[a], bx = p->x[b], by = p->y[b];
      if (((ay > y) != (by > y)) && (x < (bx - ax) * (y - ay) / (by - ay) + ax))
        inside = !inside;
    }
    if (inside)
      return p->node;
  }
  return SIZE_MAX;
}
