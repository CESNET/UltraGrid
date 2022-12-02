/**
 * @file   utils/y4m.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 *
 * This file is part of GPUJPEG.
 */
/*
 * Copyright (c) 2022, CESNET z.s.p.o.
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

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "y4m.h"

static bool y4m_process_chroma_type(char *c, struct y4m_metadata *info) {
        info->bitdepth = 8;
        if (strcmp(c, "444alpha") == 0) {
                info->subsampling = Y4M_SUBS_YUVA;
                return true;
        }
        if (strncmp(c, "mono", 4) == 0) {
                info->subsampling = Y4M_SUBS_MONO;
                sscanf(c, "mono%d", &info->bitdepth);
                return true;
        }
        int subsampling = 0;
        if (sscanf(c, "%dp%d", &subsampling, &info->bitdepth) == 0) {
                fprintf(stderr, "Y4M: unable to parse chroma type");
                return false;
        }
        info->subsampling = subsampling;
        return true;
}

static size_t y4m_get_data_len(int width, int height, enum y4m_subsampling subsampling) {
        switch (subsampling) {
                case Y4M_SUBS_MONO: return (size_t) width * height;
                case Y4M_SUBS_420: return (size_t) width * height + (size_t) 2 * ((width + 1) / 2) * ((height + 1) / 2);
                case Y4M_SUBS_422: return (size_t) width * height + (size_t) 2 * ((width + 1) / 2) * height;
                case Y4M_SUBS_444: return (size_t) width * height * 3;
                case Y4M_SUBS_YUVA: return (size_t) width * height * 4;
                default:
                          fprintf(stderr, "Unsupported subsampling '%d'\n", subsampling);
                          return 0;
        }
}

size_t y4m_read(const char *filename, struct y4m_metadata *info, unsigned char **data, void *(*allocator)(size_t)) {
        FILE *file = fopen(filename, "rb");
        if (!file) {
                fprintf(stderr, "Failed to open %s: %s", filename, strerror(errno));
                return 0;
        }
        char item[129];
        if (fscanf(file, "%128s", item) != 1 || strcmp(item, "YUV4MPEG2") != 0) {
               fprintf(stderr, "File '%s' doesn't seem to be valid Y4M.\n", filename);
               fclose(file);
               return 0;
        }
        memset(info, 0, sizeof *info);
        info->width = info->height = info->bitdepth = info->subsampling = 0;
        while (fscanf(file, " %128s", item) == 1 && strcmp(item, "FRAME") != 0) {
                switch (item[0]) {
                        case 'W': info->width = atoi(item + 1); break;
                        case 'H': info->height = atoi(item + 1); break;
                        case 'C':
                                  if (!y4m_process_chroma_type(item + 1, info)) {
                                          fclose(file);
                                          return 0;
                                  }
                                  break;
                        case 'X':
                                  if (strcmp(item, "XCOLORRANGE=LIMITED") == 0) {
                                          info->limited = true;
                                  }
                        // F, I, A currently ignored
                }
        }
        if (getc(file) != '\n') { // after FRAME
                return 0;
        }
        size_t datalen = y4m_get_data_len(info->width, info->height, info->subsampling);
        if (data == NULL || allocator == NULL) {
                fclose(file);
                return datalen;
        }
        *data = (unsigned char *) allocator(datalen);
        if (!*data) {
                fprintf(stderr, "Unspecified depth header field!");
                fclose(file);
                return 0;
        }
        fread((char *) *data, datalen, 1, file);
        if (feof(file) || ferror(file)) {
                perror("Unable to load Y4M data from file");
                fclose(file);
                return 0;
        }
        fclose(file);
        return datalen;
}

bool y4m_write(const char *filename, int width, int height, enum y4m_subsampling subsampling, int depth, bool limited, const unsigned char *data) {
        errno = 0;
        FILE *file = fopen(filename, "wb");
        if (!file) {
                fprintf(stderr, "Failed to open %s for writing: %s", filename, strerror(errno));
                return false;
        }
        char chroma_type[42];
        if (subsampling == Y4M_SUBS_MONO) {
                snprintf(chroma_type, sizeof chroma_type, "mono");
        } else if (subsampling == Y4M_SUBS_YUVA) {
                if (depth != 8) {
                        fprintf(stderr, "Only 8-bit 444alpha is supported for Y4M!");
                        return false;
                }
                snprintf(chroma_type, sizeof chroma_type, "444alpha");
        } else {
                snprintf(chroma_type, sizeof chroma_type, "%d", subsampling);
        }
        size_t len = y4m_get_data_len(width, height, subsampling);
        if (len == 0) {
                return false;
        }
        if (depth > 8) {
                len *= 2;
                snprintf(chroma_type + strlen(chroma_type), sizeof chroma_type - strlen(chroma_type), "%s%d",
                                subsampling != Y4M_SUBS_MONO ? "p" : "", depth); // 'p' in 420p10 but not mono10
        }

        fprintf(file, "YUV4MPEG2 W%d H%d F25:1 Ip A0:0 C%s XCOLORRANGE=%s\nFRAME\n",
                        width, height, chroma_type, limited ? "LIMITED" : "FULL");
        fwrite((const char *) data, len, 1, file);
        bool ret = !ferror(file);
        if (!ret) {
                perror("Unable to write Y4M data");
        }
        fclose(file);
        return ret;
}

