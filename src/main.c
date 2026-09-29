#include "editor.h"
#include "platform.h"

#include <stdio.h>

int main(int argc, char **argv)
{
    if (argc > 2) {
        fprintf(stderr, "Usage: %s [project-directory]\n", argv[0]);
        return 2;
    }
    NNPlatform *platform = NULL;
    if (!platform_open("NNModelling", 1400, 900, &platform)) {
        fprintf(stderr, "Could not open NNModelling: %s\n", platform_error());
        return 1;
    }
    int result = nn_editor_run(platform, "stereotype-packages/core",
                               argc == 2 ? argv[1] : NULL);
    platform_close(platform);
    return result;
}
