#include <libug.h>
#include <unistd.h>
#include <stdlib.h>

#define WIDTH 1920
#define HEIGHT 1920
#define FPS 30.0

int main() {
        struct ug_sender_state *s = ug_sender_init("localhost", 1500);
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

