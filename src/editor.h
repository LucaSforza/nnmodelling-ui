#ifndef NN_EDITOR_H
#define NN_EDITOR_H

#include "platform.h"

/* Owns transient UI state and active project for duration of loop. */
int nn_editor_run(NNPlatform *platform, const char *core_root,
                  const char *initial_project_path);

#endif
