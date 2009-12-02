#ifndef __v_codec_h

#define __v_codec_h

typedef enum {
        RGBA,
        UYVY,
        Vuy2,
        DVS8,
        R10k,
        v210,
        DVS10,
} codec_t;

struct codec_info_t {
        int codec;
        const char *name;
        unsigned int fcc;
        int h_align;
        double bpp;
        unsigned rgb:1;
} codec_info_t;

extern const struct codec_info_t codec_info[];

void show_codec_help(void);

#endif
