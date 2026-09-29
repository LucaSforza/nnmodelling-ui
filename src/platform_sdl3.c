#include "platform.h"

#include <SDL3/SDL.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define STB_TRUETYPE_IMPLEMENTATION
#include "../third_party/stb/stb_truetype.h"

#define FONT_PATH "third_party/fonts/DejaVuSans.ttf"
#define FONT_SIZE 18.0f
#define GLYPH_FIRST 32
#define GLYPH_COUNT 224
#define CLIP_DEPTH 16
#define ATLAS_SIZE 1024

struct NNPlatform {
    SDL_Window *window;
    SDL_Renderer *renderer;
    SDL_Texture *font_texture;
    stbtt_bakedchar glyphs[GLYPH_COUNT];
    NNPlatformRect clips[CLIP_DEPTH];
    int clip_count;
};

static char last_error[256];

static void save_error(const char *message)
{
    (void)snprintf(last_error, sizeof(last_error), "%s", message != NULL ? message : "unknown error");
}

static SDL_FRect to_sdl_rect(NNPlatformRect rect)
{
    return (SDL_FRect){rect.x, rect.y, rect.width, rect.height};
}

static SDL_Rect to_sdl_clip(NNPlatformRect rect)
{
    int left = (int)floorf(rect.x), top = (int)floorf(rect.y);
    int right = (int)ceilf(rect.x + rect.width), bottom = (int)ceilf(rect.y + rect.height);
    return (SDL_Rect){left, top, right - left, bottom - top};
}

static void set_color(NNPlatform *platform, NNPlatformColor color)
{
    (void)SDL_SetRenderDrawColor(platform->renderer, color.r, color.g, color.b, color.a);
}

static bool load_font(NNPlatform *platform)
{
    FILE *file = fopen(FONT_PATH, "rb");
    if (file == NULL) {
        save_error("could not open bundled font: " FONT_PATH);
        return false;
    }
    if (fseek(file, 0, SEEK_END) != 0) {
        fclose(file);
        save_error("could not seek bundled font");
        return false;
    }
    long length = ftell(file);
    if (length <= 0 || fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        save_error("bundled font is empty or unreadable");
        return false;
    }
    unsigned char *data = malloc((size_t)length);
    if (data == NULL || fread(data, 1, (size_t)length, file) != (size_t)length) {
        free(data);
        fclose(file);
        save_error("could not read bundled font");
        return false;
    }
    fclose(file);

    unsigned char *bitmap = calloc((size_t)ATLAS_SIZE * ATLAS_SIZE, 1);
    if (bitmap == NULL || stbtt_BakeFontBitmap(data, 0, FONT_SIZE, bitmap, ATLAS_SIZE,
                                               ATLAS_SIZE, GLYPH_FIRST, GLYPH_COUNT,
                                               platform->glyphs) <= 0) {
        free(bitmap);
        free(data);
        save_error("could not rasterize bundled font into atlas");
        return false;
    }
    free(data);

    unsigned char *rgba = malloc((size_t)ATLAS_SIZE * ATLAS_SIZE * 4);
    if (rgba == NULL) {
        free(bitmap);
        save_error("out of memory creating font atlas");
        return false;
    }
    for (size_t i = 0, n = (size_t)ATLAS_SIZE * ATLAS_SIZE; i < n; ++i) {
        rgba[i * 4] = rgba[i * 4 + 1] = rgba[i * 4 + 2] = 255;
        rgba[i * 4 + 3] = bitmap[i];
    }
    free(bitmap);

    platform->font_texture = SDL_CreateTexture(platform->renderer, SDL_PIXELFORMAT_RGBA32,
                                                SDL_TEXTUREACCESS_STATIC, ATLAS_SIZE, ATLAS_SIZE);
    bool ok = platform->font_texture != NULL &&
              SDL_UpdateTexture(platform->font_texture, NULL, rgba, ATLAS_SIZE * 4) &&
              SDL_SetTextureBlendMode(platform->font_texture, SDL_BLENDMODE_BLEND);
    free(rgba);
    if (!ok) {
        save_error(SDL_GetError());
        return false;
    }
    return true;
}

static void translate_event(const SDL_Event *native, NNPlatformEvent *event)
{
    memset(event, 0, sizeof(*event));
    switch (native->type) {
    case SDL_EVENT_QUIT: event->type = NN_PLATFORM_EVENT_QUIT; break;
    case SDL_EVENT_MOUSE_MOTION:
        event->type = NN_PLATFORM_EVENT_MOUSE_MOVE;
        event->x = native->motion.x; event->y = native->motion.y;
        break;
    case SDL_EVENT_MOUSE_BUTTON_DOWN:
    case SDL_EVENT_MOUSE_BUTTON_UP:
        event->type = native->type == SDL_EVENT_MOUSE_BUTTON_DOWN
            ? NN_PLATFORM_EVENT_MOUSE_DOWN : NN_PLATFORM_EVENT_MOUSE_UP;
        event->x = native->button.x; event->y = native->button.y;
        event->button = native->button.button;
        break;
    case SDL_EVENT_MOUSE_WHEEL:
        event->type = NN_PLATFORM_EVENT_WHEEL;
        event->delta_x = native->wheel.x; event->delta_y = native->wheel.y;
        break;
    case SDL_EVENT_KEY_DOWN:
    case SDL_EVENT_KEY_UP:
        event->type = native->type == SDL_EVENT_KEY_DOWN
            ? NN_PLATFORM_EVENT_KEY_DOWN : NN_PLATFORM_EVENT_KEY_UP;
        event->key = (uint32_t)native->key.key;
        event->modifiers = (uint16_t)native->key.mod;
        break;
    case SDL_EVENT_TEXT_INPUT:
        event->type = NN_PLATFORM_EVENT_TEXT;
        (void)snprintf(event->text, sizeof(event->text), "%s", native->text.text);
        break;
    case SDL_EVENT_WINDOW_RESIZED:
    case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED:
        event->type = NN_PLATFORM_EVENT_RESIZE;
        event->width = native->window.data1; event->height = native->window.data2;
        break;
    default: event->type = NN_PLATFORM_EVENT_OTHER; break;
    }
}

bool platform_open(const char *title, int width, int height, NNPlatform **platform)
{
    last_error[0] = '\0';
    if (platform == NULL || title == NULL || width <= 0 || height <= 0) {
        save_error("platform_open requires a title, output handle, and positive dimensions");
        return false;
    }
    *platform = NULL;
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        save_error(SDL_GetError());
        return false;
    }
    NNPlatform *opened = calloc(1, sizeof(*opened));
    if (opened == NULL) {
        save_error("out of memory");
        SDL_Quit();
        return false;
    }
    opened->window = SDL_CreateWindow(title, width, height, SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY);
    if (opened->window == NULL) {
        save_error(SDL_GetError());
        goto fail;
    }
    opened->renderer = SDL_CreateRenderer(opened->window, NULL);
    if (opened->renderer == NULL) {
        save_error(SDL_GetError());
        goto fail;
    }
    if (!SDL_SetRenderLogicalPresentation(opened->renderer, width, height,
                                          SDL_LOGICAL_PRESENTATION_STRETCH)) {
        save_error(SDL_GetError());
        goto fail;
    }
    if (!SDL_SetRenderDrawBlendMode(opened->renderer, SDL_BLENDMODE_BLEND)) {
        save_error(SDL_GetError());
        goto fail;
    }
    if (!load_font(opened))
        goto fail;
    if (!SDL_StartTextInput(opened->window)) {
        save_error(SDL_GetError());
        goto fail;
    }
    *platform = opened;
    return true;

fail:
    if (opened->font_texture != NULL) SDL_DestroyTexture(opened->font_texture);
    if (opened->renderer != NULL) SDL_DestroyRenderer(opened->renderer);
    if (opened->window != NULL) SDL_DestroyWindow(opened->window);
    free(opened);
    SDL_Quit();
    return false;
}

static bool next_event(NNPlatform *platform, NNPlatformEvent *event, bool wait)
{
    if (platform == NULL || event == NULL) {
        save_error("event polling requires a platform and event output");
        return false;
    }
    SDL_Event native;
    bool received = wait ? SDL_WaitEvent(&native) : SDL_PollEvent(&native);
    if (!received) {
        if (wait) save_error(SDL_GetError());
        return false;
    }
    translate_event(&native, event);
    if (event->type == NN_PLATFORM_EVENT_RESIZE) {
        if (!SDL_GetWindowSize(platform->window, &event->width, &event->height) ||
            !SDL_SetRenderLogicalPresentation(platform->renderer,
                                               event->width, event->height,
                                               SDL_LOGICAL_PRESENTATION_STRETCH)) {
            save_error(SDL_GetError());
            return false;
        }
    }
    return true;
}

bool platform_wait_event(NNPlatform *platform, NNPlatformEvent *event)
{
    last_error[0] = '\0';
    return next_event(platform, event, true);
}

bool platform_poll_event(NNPlatform *platform, NNPlatformEvent *event)
{
    last_error[0] = '\0';
    return next_event(platform, event, false);
}

void platform_close(NNPlatform *platform)
{
    if (platform == NULL) return;
    if (platform->font_texture != NULL) SDL_DestroyTexture(platform->font_texture);
    if (platform->renderer != NULL) SDL_DestroyRenderer(platform->renderer);
    if (platform->window != NULL) SDL_DestroyWindow(platform->window);
    free(platform);
    SDL_Quit();
}

const char *platform_error(void) { return last_error; }

bool platform_begin_frame(NNPlatform *platform, NNPlatformColor clear_color)
{
    last_error[0] = '\0';
    if (platform == NULL) { save_error("platform_begin_frame requires a platform"); return false; }
    set_color(platform, clear_color);
    if (!SDL_RenderClear(platform->renderer)) { save_error(SDL_GetError()); return false; }
    platform->clip_count = 0;
    return true;
}

void platform_end_frame(NNPlatform *platform)
{
    if (platform != NULL) SDL_RenderPresent(platform->renderer);
}

bool platform_capture_bmp(NNPlatform *platform, const char *path)
{
    if (!platform || !path) { save_error("capture requires platform and path"); return false; }
    SDL_Surface *surface = SDL_RenderReadPixels(platform->renderer, NULL);
    if (!surface) { save_error(SDL_GetError()); return false; }
    bool saved = SDL_SaveBMP(surface, path);
    if (!saved) save_error(SDL_GetError());
    SDL_DestroySurface(surface);
    return saved;
}

void platform_fill_rect(NNPlatform *platform, NNPlatformRect rect, NNPlatformColor color)
{
    if (platform == NULL) return;
    set_color(platform, color);
    SDL_FRect sdl_rect = to_sdl_rect(rect);
    if (!SDL_RenderFillRect(platform->renderer, &sdl_rect)) save_error(SDL_GetError());
}

void platform_outline_rect(NNPlatform *platform, NNPlatformRect rect, NNPlatformColor color)
{
    if (platform == NULL) return;
    set_color(platform, color);
    SDL_FRect sdl_rect = to_sdl_rect(rect);
    if (!SDL_RenderRect(platform->renderer, &sdl_rect)) save_error(SDL_GetError());
}

void platform_line(NNPlatform *platform, NNPlatformPoint start, NNPlatformPoint end,
                   NNPlatformColor color)
{
    if (platform == NULL) return;
    set_color(platform, color);
    if (!SDL_RenderLine(platform->renderer, start.x, start.y, end.x, end.y)) save_error(SDL_GetError());
}

void platform_circle(NNPlatform *platform, NNPlatformPoint center, float radius,
                     NNPlatformColor color, bool filled)
{
    if (platform == NULL || radius < 0.0f) return;
    set_color(platform, color);
    int r = (int)(radius + 0.5f);
    for (int y = -r; y <= r; ++y) {
        int x = (int)(sqrtf((float)(r * r - y * y)) + 0.5f);
        if (filled) SDL_RenderLine(platform->renderer, center.x - x, center.y + y,
                                   center.x + x, center.y + y);
        else {
            SDL_RenderPoint(platform->renderer, center.x - x, center.y + y);
            SDL_RenderPoint(platform->renderer, center.x + x, center.y + y);
        }
    }
}

static NNPlatformRect intersect_rect(NNPlatformRect a, NNPlatformRect b)
{
    float left = a.x > b.x ? a.x : b.x, top = a.y > b.y ? a.y : b.y;
    float right_a = a.x + a.width, right_b = b.x + b.width;
    float bottom_a = a.y + a.height, bottom_b = b.y + b.height;
    float right = right_a < right_b ? right_a : right_b;
    float bottom = bottom_a < bottom_b ? bottom_a : bottom_b;
    return (NNPlatformRect){left, top, right > left ? right - left : 0, bottom > top ? bottom - top : 0};
}

bool platform_push_clip(NNPlatform *platform, NNPlatformRect rect)
{
    if (platform == NULL || platform->clip_count >= CLIP_DEPTH) {
        save_error("clip stack is full or platform is null");
        return false;
    }
    NNPlatformRect effective = platform->clip_count == 0 ? rect
        : intersect_rect(platform->clips[platform->clip_count - 1], rect);
    platform->clips[platform->clip_count++] = effective;
    SDL_Rect clip = to_sdl_clip(effective);
    if (!SDL_SetRenderClipRect(platform->renderer, &clip)) {
        --platform->clip_count;
        save_error(SDL_GetError());
        return false;
    }
    return true;
}

void platform_pop_clip(NNPlatform *platform)
{
    if (platform == NULL || platform->clip_count == 0) return;
    --platform->clip_count;
    if (platform->clip_count == 0) SDL_SetRenderClipRect(platform->renderer, NULL);
    else {
        SDL_Rect clip = to_sdl_clip(platform->clips[platform->clip_count - 1]);
        SDL_SetRenderClipRect(platform->renderer, &clip);
    }
}

static float draw_one(NNPlatform *platform, float *x, float y, unsigned int codepoint,
                      NNPlatformColor color, bool draw)
{
    if (codepoint < GLYPH_FIRST || codepoint >= GLYPH_FIRST + GLYPH_COUNT) codepoint = '?';
    stbtt_bakedchar *glyph = &platform->glyphs[codepoint - GLYPH_FIRST];
    if (draw && glyph->x1 > glyph->x0 && glyph->y1 > glyph->y0) {
        SDL_FRect source = {(float)glyph->x0, (float)glyph->y0,
                            (float)(glyph->x1 - glyph->x0), (float)(glyph->y1 - glyph->y0)};
        SDL_FRect target = {*x + glyph->xoff, y + glyph->yoff,
                            source.w, source.h};
        SDL_SetTextureColorMod(platform->font_texture, color.r, color.g, color.b);
        SDL_SetTextureAlphaMod(platform->font_texture, color.a);
        if (!SDL_RenderTexture(platform->renderer, platform->font_texture, &source, &target))
            save_error(SDL_GetError());
    }
    *x += glyph->xadvance;
    return *x;
}

static unsigned int decode_utf8(const unsigned char **input)
{
    unsigned char c = *(*input)++;
    if (c < 0x80) return c;
    if ((c & 0xe0) == 0xc0) {
        unsigned int next = *(*input)++;
        return ((c & 0x1f) << 6) | (next & 0x3f);
    }
    if ((c & 0xf0) == 0xe0) {
        unsigned int second = *(*input)++, third = *(*input)++;
        return ((c & 0x0f) << 12) | ((second & 0x3f) << 6) | (third & 0x3f);
    }
    if ((c & 0xf8) == 0xf0) {
        unsigned int second = *(*input)++, third = *(*input)++, fourth = *(*input)++;
        return ((c & 0x07) << 18) | ((second & 0x3f) << 12) |
               ((third & 0x3f) << 6) | (fourth & 0x3f);
    }
    return '?';
}

static float text_run(NNPlatform *platform, float x, float y, const char *utf8,
                      NNPlatformColor color, bool draw)
{
    if (platform == NULL || utf8 == NULL) return 0.0f;
    const unsigned char *cursor = (const unsigned char *)utf8;
    float origin = x;
    while (*cursor != '\0') draw_one(platform, &x, y, decode_utf8(&cursor), color, draw);
    return x - origin;
}

void platform_draw_text(NNPlatform *platform, float x, float y, const char *utf8,
                        NNPlatformColor color)
{
    (void)text_run(platform, x, y, utf8, color, true);
}

float platform_text_width(NNPlatform *platform, const char *utf8)
{
    return text_run(platform, 0, 0, utf8, (NNPlatformColor){0, 0, 0, 0}, false);
}
