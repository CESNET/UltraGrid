/**
 * @file   utils/pam.h
 * @author Martin Pulec     <pulec@cesnet.cz>
 *
 * Very simple library to read and write PAM/PPM files. Only binary formats
 * (P5, P6) for PNM are processed, P4 is also not used.
 *
 * This file is part of GPUJPEG.
 */
/*
 * Copyright (c) 2013-2022, CESNET z.s.p.o.
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef PAM_H_7E23A609_963A_45A8_88E2_ED4D3FDFF69F
#define PAM_H_7E23A609_963A_45A8_88E2_ED4D3FDFF69F

#ifdef __cplusplus
#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#else
#include <ctype.h>
#include <errno.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#endif

#ifdef __GNUC__
#define PAM_ATTRIBUTE_UNUSED __attribute__((unused))
#else
#define PAM_ATTRIBUTE_UNUSED
#endif

struct pam_metadata {
        int width;
        int height;
        int depth; // == channel count
        int maxval;
};

static inline bool pam_read(const char *filename, struct pam_metadata *info, unsigned char **data, void *(*allocator)(size_t)) PAM_ATTRIBUTE_UNUSED;
static bool pam_write(const char *filename, unsigned int width, unsigned int height, int depth, int maxval, const unsigned char *data, bool pnm) PAM_ATTRIBUTE_UNUSED;

static inline void parse_pam(FILE *file, struct pam_metadata *info) {
        char line[128];
        fgets(line, sizeof line - 1, file);
        while (!feof(file) && !ferror(file)) {
                if (strcmp(line, "ENDHDR\n") == 0) {
                        break;
                }
                char *spc = strchr(line, ' ');
                if (spc == NULL) {
                        break;
                }
                const char *key = line;
                *spc = '\0';
                const char *val = spc + 1;
                if (strcmp(key, "WIDTH") == 0) {
                        info->width = atoi(val);
                } else if (strcmp(key, "HEIGHT") == 0) {
                        info->height = atoi(val);
                } else if (strcmp(key, "DEPTH") == 0) {
                        info->depth = atoi(val);
                } else if (strcmp(key, "MAXVAL") == 0) {
                        info->maxval = atoi(val);
                } else if (strcmp(key, "TUPLTYPE") == 0) {
                        // ignored - assuming MAXVAL == 255, value of DEPTH is sufficient
                        // to determine pixel format
                } else {
                        fprintf(stderr, "unrecognized key %s in PAM header\n", key);
                }
                fgets(line, sizeof line - 1, file);
        }
}

static inline bool parse_pnm(FILE *file, char pnm_id, struct pam_metadata *info) {
        switch (pnm_id) {
                case '1':
                case '2':
                case '3':
                case '4':
                        fprintf(stderr, "Unsupported PNM P%c\n", pnm_id);
                        return false;
                case '5':
                        info->depth = 1;
                        break;
                case '6':
                        info->depth = 3;
                        break;
                default:
                        fprintf(stderr, "Wrong PNM P%c\n", pnm_id);
                        return false;
        }
        int item_nr = 0;
        while (!feof(file) && !ferror(file)) {
                int val = 0;
                if (fscanf(file, "%d", &val) == 1) {
                        switch (item_nr++) {
                                case 0:
                                        info->width = val;
                                        break;
                                case 1:
                                        info->height = val;
                                        break;
                                case 2:
                                        info->maxval = val;
                                        getc(file); // skip whitespace following header
                                        return true;
                        }
                } else {
                        if (getc(file) == '#') {
                                int ch;
                                while ((ch = getc(file)) != '\n' && ch != EOF)
                                        ;
                        } else {
                                break;
                        }
                }
        }
        fprintf(stderr, "Problem parsing PNM header, number of hdr items successfully read: %d\n", item_nr);
        return false;
}

static inline bool pam_read(const char *filename, struct pam_metadata *info, unsigned char **data, void *(*allocator)(size_t)) {
        char line[128];
        errno = 0;
        FILE *file = fopen(filename, "rb");
        if (!file) {
                fprintf(stderr, "Failed to open %s: %s", filename, strerror(errno));
                return false;
        }
        memset(info, 0, sizeof *info);
        fgets(line, 4, file);
        if (feof(file) || ferror(file)) {
                fprintf(stderr, "File '%s' read error: %s\n", filename, strerror(errno));
        }
        if (strcmp(line, "P7\n") == 0) {
                parse_pam(file, info);
        } else if (strlen(line) == 3 && line[0] == 'P' && isspace(line[2])) {
                parse_pnm(file, line[1], info);
        } else {
               fprintf(stderr, "File '%s' doesn't seem to be valid PAM or PNM.\n", filename);
               fclose(file);
               return false;
        }
        if (info->width * info->height == 0) {
                fprintf(stderr, "Unspecified size header field!");
                fclose(file);
                return false;
        }
        if (info->depth == 0) {
                fprintf(stderr, "Unspecified depth header field!");
                fclose(file);
                return false;
        }
        if (info->maxval == 0) {
                fprintf(stderr, "Unspecified maximal value field!");
                fclose(file);
                return false;
        }
        if (data == NULL || allocator == NULL) {
                fclose(file);
                return true;
        }
        size_t datalen = (size_t) info->depth * info->width * info->height * (info->maxval <= 255 ? 1 : 2);
        *data = (unsigned char *) allocator(datalen);
        if (!*data) {
                fprintf(stderr, "Unspecified depth header field!");
                fclose(file);
                return false;
        }
        fread((char *) *data, datalen, 1, file);
        if (feof(file) || ferror(file)) {
                perror("Unable to load PAM/PNM data from file");
                fclose(file);
                return false;
        }
        fclose(file);
        return true;
}

static bool pam_write(const char *filename, unsigned int width, unsigned int height, int depth, int maxval, const unsigned char *data, bool pnm) {
        errno = 0;
        FILE *file = fopen(filename, "wb");
        if (!file) {
                fprintf(stderr, "Failed to open %s for writing: %s", filename, strerror(errno));
                return false;
        }
        if (pnm) {
                if (depth != 1 && depth != 3) {
                        fprintf(stderr, "Only 1 or 3 channels supported for PNM!\n");
                        return false;
                }
                fprintf(file, "P%d\n"
                        "%u %u\n"
                        "%d\n",
                        depth == 1 ? 5 : 6,
                        width, height, maxval);
        } else {
                const char *tuple_type = "INVALID";
                switch (depth) {
                        case 4: tuple_type = "RGB_ALPHA"; break;
                        case 3: tuple_type = "RGB"; break;
                        case 2: tuple_type = "GRAYSCALE_ALPHA"; break;
                        case 1: tuple_type = "GRAYSCALE"; break;
                        default: fprintf(stderr, "Wrong depth: %d\n", depth);
                }
                fprintf(file, "P7\n"
                        "WIDTH %u\n"
                        "HEIGHT %u\n"
                        "DEPTH %d\n"
                        "MAXVAL %d\n"
                        "TUPLTYPE %s\n"
                        "ENDHDR\n",
                        width, height, depth, maxval, tuple_type);
        }
        fwrite((const char *) data, width * height * depth, maxval <= 255 ? 1 : 2, file);
        bool ret = !ferror(file);
        if (!ret) {
                perror("Unable to write PAM/PNM data");
        }
        fclose(file);
        return ret;
}

#endif // defined PAM_H_7E23A609_963A_45A8_88E2_ED4D3FDFF69F
