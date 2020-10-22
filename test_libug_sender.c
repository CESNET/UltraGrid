#include <unistd.h>

#include <stdio.h>
#include <stdlib.h>

#include <libug.h>

#define WIDTH 1920
#define HEIGHT 1920
#define FPS 30.0

static void render_packet_received_callback(void *udata, struct RenderPacket *pkt) {
        (void) udata;
        printf("Received RenderPacket: %p\n", pkt);
}

int main(int argc, char *argv[]) {
        struct ug_sender *s = ug_sender_init(argc == 1 ? "localhost" : argv[1], 1500, render_packet_received_callback, NULL);
        if (!s) {
                return 1;
        }
        size_t len = WIDTH * HEIGHT * 4;
        char *test = malloc(len);
        while (1) {
                ug_send_frame(s, test, len, UG_RGBA, WIDTH, HEIGHT);
                usleep(1000 * 1000 / FPS);
        }

        free(test);
        ug_sender_done(s);
}

