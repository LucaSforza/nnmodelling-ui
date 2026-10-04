#define _GNU_SOURCE
#include <dlfcn.h>
#include <netinet/in.h>
#include <string.h>
#include <sys/socket.h>

/* Qt's VNC plugin binds Any. Confine this temporary process to loopback. */
int bind(int fd, const struct sockaddr *address, socklen_t size) {
    int (*real_bind)(int, const struct sockaddr *, socklen_t) = dlsym(RTLD_NEXT, "bind");
    if (address->sa_family == AF_INET && size == sizeof(struct sockaddr_in)) {
        struct sockaddr_in local;
        memcpy(&local, address, size);
        local.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        return real_bind(fd, (const struct sockaddr *)&local, size);
    }
    if (address->sa_family == AF_INET6 && size == sizeof(struct sockaddr_in6)) {
        struct sockaddr_in6 local;
        memcpy(&local, address, size);
        local.sin6_addr = in6addr_loopback;
        return real_bind(fd, (const struct sockaddr *)&local, size);
    }
    return real_bind(fd, address, size);
}
