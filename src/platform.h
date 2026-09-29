#ifndef NNMODELLING_PLATFORM_H
#define NNMODELLING_PLATFORM_H

#include <stdbool.h>
#include <stdint.h>

typedef struct NNPlatform NNPlatform;

typedef enum {
    NN_PLATFORM_EVENT_OTHER,
    NN_PLATFORM_EVENT_QUIT,
    NN_PLATFORM_EVENT_MOUSE_MOVE,
    NN_PLATFORM_EVENT_MOUSE_DOWN,
    NN_PLATFORM_EVENT_MOUSE_UP,
    NN_PLATFORM_EVENT_WHEEL,
    NN_PLATFORM_EVENT_KEY_DOWN,
    NN_PLATFORM_EVENT_KEY_UP,
    NN_PLATFORM_EVENT_TEXT,
    NN_PLATFORM_EVENT_RESIZE
} NNPlatformEventType;

typedef struct {
    NNPlatformEventType type;
    float x, y;
    float delta_x, delta_y;
    int button;
    uint32_t key;
    uint16_t modifiers;
    int width, height;
    char text[64];
} NNPlatformEvent;

typedef struct { float x, y; } NNPlatformPoint;
typedef struct { float x, y, width, height; } NNPlatformRect;
typedef struct { uint8_t r, g, b, a; } NNPlatformColor;

bool platform_open(const char *title, int width, int height, NNPlatform **platform);
bool platform_wait_event(NNPlatform *platform, NNPlatformEvent *event);
bool platform_poll_event(NNPlatform *platform, NNPlatformEvent *event);
void platform_close(NNPlatform *platform);
const char *platform_error(void);

bool platform_begin_frame(NNPlatform *platform, NNPlatformColor clear_color);
void platform_end_frame(NNPlatform *platform);
bool platform_capture_bmp(NNPlatform *platform, const char *path);
void platform_fill_rect(NNPlatform *platform, NNPlatformRect rect, NNPlatformColor color);
void platform_outline_rect(NNPlatform *platform, NNPlatformRect rect, NNPlatformColor color);
void platform_line(NNPlatform *platform, NNPlatformPoint start, NNPlatformPoint end,
                   NNPlatformColor color);
void platform_circle(NNPlatform *platform, NNPlatformPoint center, float radius,
                     NNPlatformColor color, bool filled);
bool platform_push_clip(NNPlatform *platform, NNPlatformRect rect);
void platform_pop_clip(NNPlatform *platform);
void platform_draw_text(NNPlatform *platform, float x, float y, const char *utf8,
                        NNPlatformColor color);
float platform_text_width(NNPlatform *platform, const char *utf8);

#endif
