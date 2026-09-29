#include "platform.h"

#include <stdio.h>

int main(void)
{
    NNPlatform *platform = NULL;
    if (!platform_open("NNModelling smoke", 64, 64, &platform)) {
        fprintf(stderr, "Platform smoke failed: %s\n", platform_error());
        return 1;
    }

    platform_close(platform);
    return 0;
}
