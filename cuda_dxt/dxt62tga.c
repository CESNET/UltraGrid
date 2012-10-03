#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

/* size of TARGA header */
#define TGA_HEADER_SIZE 18 


typedef uint64_t u64;
typedef uint32_t u32;
typedef uint16_t u16;
typedef uint8_t u8;

static u8 clamp(const double s) {
    const int is = (int)(s + 0.5);
    if(is > 255)
        return 255;
    if(is < 0)
        return 0;
    return is;
}


static int ycocg_to_bgr(int x, int y, u8 * out, int out_stride, double a, double r, double g, double b) {
    const double _scale = 1.00000000E+00/(3.18750000E+01*b + 1.00000000E+00);
    const double _Co = (r - 5.01960814E-01)*_scale;
    const double _Cg = (g - 5.01960814E-01)*_scale;
    const double _R = (a + _Co) - _Cg;
    const double _G = a + _Cg;
    const double _B = (a - _Co) - _Cg;
    out += 3 * (x + y * out_stride);
    out[2] = clamp(_R * 255.0);
    out[1] = clamp(_G * 255.0);
    out[0] = clamp(_B * 255.0);
/*     printf("A: %f.\n", a); */
}

static double alpha_decode(const double a0, const double a1, int alpha_idx) {
    if(a0 > a1) {
        switch(alpha_idx) {
            case 0: return a0;
            case 1: return a1;
            case 2: return (6.0 * a0 + 1.0 * a1) / 7.0;
            case 3: return (5.0 * a0 + 2.0 * a1) / 7.0; 
            case 4: return (4.0 * a0 + 3.0 * a1) / 7.0;
            case 5: return (3.0 * a0 + 4.0 * a1) / 7.0;
            case 6: return (2.0 * a0 + 5.0 * a1) / 7.0;
            case 7: return (1.0 * a0 + 6.0 * a1) / 7.0;
        }
    } else {
        switch(alpha_idx) {
            case 0: return a0;
            case 1: return a1;
            case 2: return (4.0 * a0 + 1.0 * a1) / 5.0;
            case 3: return (3.0 * a0 + 2.0 * a1) / 5.0; 
            case 4: return (2.0 * a0 + 3.0 * a1) / 5.0;
            case 5: return (1.0 * a0 + 4.0 * a1) / 5.0;
            case 6: return 0.0;
            case 7: return 1.0;
        }
    }
}

static int dxt5ycocg_block_to_bgr(int x_offset, int y_offset, u8 * out, int out_stride, u64 alpha_code, u64 rgb_code) {
    int x, y, rgb_idx;
    double a, r[4], g[4], b[4];
    const double alpha_max = (alpha_code & 0xFF) / 255.0;
    const double alpha_min = ((alpha_code >> 8) & 0xFF) / 255.0;
    
    b[0] = (rgb_code & 0x1F) / 31.0;
    g[0] = ((rgb_code >> 5) & 0x3F) / 63.0;
    r[0] = ((rgb_code >> 11) & 0x1F) / 31.0;
    b[1] = ((rgb_code >> 16) & 0x1F) / 31.0;
    g[1] = ((rgb_code >> 21) & 0x3F) / 63.0;
    r[1] = ((rgb_code >> 27) & 0x1F) / 31.0;
    b[2] = (2.0 * b[0] + 1.0 * b[1]) / 3.0;
    g[2] = (2.0 * g[0] + 1.0 * g[1]) / 3.0;
    r[2] = (2.0 * r[0] + 1.0 * r[1]) / 3.0;
    b[3] = (1.0 * b[0] + 2.0 * b[1]) / 3.0;
    g[3] = (1.0 * g[0] + 2.0 * g[1]) / 3.0;
    r[3] = (1.0 * r[0] + 2.0 * r[1]) / 3.0;
    
    alpha_code >>= 16;
    rgb_code >>= 32;
    for(y = 0; y < 4; y++) {
        for(x = 0; x < 4; x++) {
            a = alpha_decode(alpha_max, alpha_min, alpha_code & 7);
            rgb_idx = rgb_code & 3;
            alpha_code >>= 3;
            rgb_code >>= 2;
            ycocg_to_bgr(x_offset + x, y_offset + y, out, out_stride,
                         a, r[rgb_idx], g[rgb_idx], b[rgb_idx]);
        }
    }
}


static int dxt5ycocg_to_bgr(const u64 * in, u8 * out, int size_x, int size_y) {
    int x, y;
    
    for(y = 0; y < size_y; y += 4) {
        for(x = 0; x < size_x; x += 4) {
            dxt5ycocg_block_to_bgr(x, y, out, size_x, in[0], in[1]);
            in += 2;
        }
    }
    
    /* nothing checked in this implementation */
    return 0;
}



static void usage(const char * const name, const char * const message) {
    printf("ERROR: %s\n", message);
    printf("Usage: %s width height input.yog output.tga\n", name);
}


int main(int argc, char** argv) {
    FILE *in_file, *tga_file;
    size_t in_size, tga_size;
    int size_x, size_y;
    u8 * tga_header;
    void * tga = 0, * in = 0;
    
    /* check arguments */
    if(5 != argc) {
        usage(argv[0], "Incorrect argument count.");
        return -1;
    }
    
    /* get image size */
    size_x = atoi(argv[1]);
    size_y = atoi(argv[2]);
    if(size_x <= 0 || (size_x & 3) || (size_y <= 0) || (size_y & 3)) {
        usage(argv[0], "Image sizes must be positive numbers divisible by 4.");
        return -1;
    }
    
    /* allocate buffers */
    in_size = size_x * size_y;
    tga_size = in_size * 3 + TGA_HEADER_SIZE;
    in = malloc(in_size);
    tga = malloc(tga_size);
    if(!in || !tga) {
        usage(argv[0], "Cannot allocate buffers.\n");
        return -1;
    }
    
    /* open input file */
    in_file = fopen(argv[3], "r");
    if(!in_file) {
        usage(argv[0], "Could not open input file.");
        return -1;
    }
    
    /* load data */
    if(1 != fread(in, in_size, 1, in_file)) {
        usage(argv[0], "Could not read from input file.");
        return -1;
    }
    fclose(in_file);
    
    /* decompress */
    if(dxt5ycocg_to_bgr((u64*)in, (u8*)tga + TGA_HEADER_SIZE, size_x, size_y)) {
        usage(argv[0], "DXT6 decompression error.");
        return -1;
    }
    
    /* add the TARGA header for 3 component RGB image */
    tga_header = (u8*)tga;
    *(tga_header + 0) = 0; /* no ID field */
    *(tga_header + 1) = 0; /* no color map */
    *(tga_header + 2) = 2; /* uncompressed RGB */
    *(tga_header + 3) = 0; /* ignored if no color map */
    *(tga_header + 4) = 0; /* ignored if no color map */
    *(tga_header + 5) = 0; /* ignored if no color map */
    *(tga_header + 6) = 0; /* ignored if no color map */
    *(tga_header + 7) = 0; /* ignored if no color map */
    *((u16*)(tga_header + 8)) = 0; /* image origin X */
    *((u16*)(tga_header + 10)) = 0; /* image origin Y */
    *((u16*)(tga_header + 12)) = size_x; /* image width */
    *((u16*)(tga_header + 14)) = size_y; /* image height */
    *(tga_header + 16) = 24; /* 24 bits per pixel */
    *(tga_header + 17) = 1 << 5; /* origin in top-left corner */
    
    /* open output file */
    tga_file = fopen(argv[4], "w");
    if(!tga_file) {
        usage(argv[0], "Could not open output file for writing.");
        return -1;
    }
    
    /* write output into the file */
    if(1 != fwrite(tga, tga_size, 1, tga_file)) {
        usage(argv[0], "Could not write into output file.");
        return -1;
    }
    fclose(tga_file);
    
    /* free buffers */
    free(in);
    free(tga);
    
    /* indicate success */
    return 0;
}

