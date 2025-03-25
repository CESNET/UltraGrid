/**
 * Simple Duck IVF mixer for VP8/VP9/AV1.
 *
 * Can be used eg. to process exported (--record) VP9 files, that cannot be
 * simply concatenated as eg. H.264 frames. Specification:
 * <https://wiki.multimedia.cx/index.php/Duck_IVF>
 */

#include <assert.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const char *
get_fcc(const char *ext)
{
        if (strcasecmp(ext, "vp8") == 0) {
                return "VP80";
        }
        if (strcasecmp(ext, "vp9") == 0) {
                return "VP90";
        }
        if (strcasecmp(ext, "av1") == 0) {
                return "av01";
        }
        fprintf(stderr, "Unsupported extension: %s\n", ext);
        exit(EXIT_FAILURE);
}

#define OUTPUT(val) fwrite(&(val), sizeof (val), 1, stdout)
int
main(int argc, char *argv[])
{
        if (argc <= 4) {
                fprintf(
                    stderr,
                    "Usage:\n\t%s <width> <height> <fps> INFILES > out.ivf\n",
                    argv[0]);
                return EXIT_FAILURE;
        }
        argv += 1; // skip progname

        // file header
        printf("DKIF");
        const uint16_t version = 0;
        OUTPUT(version);
        const uint16_t hdr_len = 32;
        OUTPUT(hdr_len);
        printf("%s", get_fcc(strchr(argv[3], '.') + 1)); // codec FourCC
        const uint16_t width  = atoi(*argv++);
        const uint16_t height = atoi(*argv++);
        const uint32_t fps    = atoi(*argv++);
        assert(width * height * fps != 0);
        OUTPUT(width);
        OUTPUT(height);
        const uint32_t fps_den = fps;
        OUTPUT(fps_den);
        const uint32_t fps_num = 1;
        OUTPUT(fps_num);
        const uint32_t nr_frames = argc - 3;
        OUTPUT(nr_frames);
        const uint32_t unused = 0;
        OUTPUT(unused);

        uint64_t pts = 0;
        while (*argv != NULL) { // iterate over files
                FILE *f = fopen(*argv++, "rb");
                assert(f != NULL);

                fseek(f, 0, SEEK_END);
                const uint32_t file_len = ftell(f);
                fseek(f, 0, SEEK_SET);

                // frame header
                OUTPUT(file_len);
                OUTPUT(pts);
                pts += 1;

                // copy the content of the file to output
                int c = 0;
                while ((c = fgetc(f)) != EOF) {
                        putc(c, stdout);
                }
                fclose(f);
        }
        // we do not do much error checking so at least return errno
        return errno;
}
