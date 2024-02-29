/**
 * @file   utils/y4m.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 *
 * This file is part of GPUJPEG.
 */
/*
 * Copyright (c) 2022-2024, CESNET z.s.p.o.
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
                fprintf(stderr, "Y4M: unable to parse chroma type\n");
                return false;
        }
        info->subsampling = subsampling;
        return true;
}

static size_t y4m_get_data_len(const struct y4m_metadata *info) {
        size_t ret = 0;
        switch (info->subsampling) {
                case Y4M_SUBS_MONO: ret = (size_t) info->width * info->height; break;
                case Y4M_SUBS_420: ret = (size_t) info->width * info->height + (size_t) 2 * ((info->width + 1) / 2) * ((info->height + 1) / 2); break;
                case Y4M_SUBS_422: ret = (size_t) info->width * info->height + (size_t) 2 * ((info->width + 1) / 2) * info->height; break;
                case Y4M_SUBS_444: ret = (size_t) info->width * info->height * 3; break;
                case Y4M_SUBS_YUVA: ret = (size_t) info->width * info->height * 4; break;
                default:
                          fprintf(stderr, "Unsupported subsampling '%d'\n", info->subsampling);
                          return 0;
        }
        return ret * (info->bitdepth > 8 ? 2 : 1);
}

size_t y4m_read(const char *filename, struct y4m_metadata *info, unsigned char **data, void *(*allocator)(size_t)) {
        FILE *file = fopen(filename, "rb");
        if (!file) {
                fprintf(stderr, "Failed to open %s: %s\n", filename, strerror(errno));
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
                fclose(file);
                return 0;
        }
        size_t datalen = y4m_get_data_len(info);
        if (data == NULL || allocator == NULL) {
                fclose(file);
                return datalen;
        }
        *data = (unsigned char *) allocator(datalen);
        if (!*data) {
                fprintf(stderr, "Unspecified depth header field!\n");
                fclose(file);
                return 0;
        }
        errno = 0;
        size_t bytes_read = fread((char *) *data, 1, datalen, file);
        if (bytes_read != datalen) {
                fprintf(stderr, "Unable to load %zd data bytes from Y4M file, read %zd bytes: %s\n",
                        datalen, bytes_read, feof(file) ? "EOF" : strerror(errno));
                fclose(file);
                return 0;
        }
        fclose(file);
        return datalen;
}

bool y4m_write(const char *filename, const struct y4m_metadata *info, const unsigned char *data) {
        errno = 0;
        FILE *file = fopen(filename, "wb");
        if (!file) {
                fprintf(stderr, "Failed to open %s for writing: %s\n", filename, strerror(errno));
                return false;
        }
        char chroma_type[42];
        if (info->subsampling == Y4M_SUBS_MONO) {
                snprintf(chroma_type, sizeof chroma_type, "mono");
        } else if (info->subsampling == Y4M_SUBS_YUVA) {
                if (info->bitdepth != 8) {
                        fprintf(stderr, "Only 8-bit 444alpha is supported for Y4M!\n");
                        fclose(file);
                        return false;
                }
                snprintf(chroma_type, sizeof chroma_type, "444alpha");
        } else {
                snprintf(chroma_type, sizeof chroma_type, "%d", info->subsampling);
        }
        size_t len = y4m_get_data_len(info);
        if (len == 0) {
                fclose(file);
                return false;
        }
        if (info->bitdepth > 8) {
                snprintf(chroma_type + strlen(chroma_type), sizeof chroma_type - strlen(chroma_type), "%s%d",
                                info->subsampling != Y4M_SUBS_MONO ? "p" : "", info->bitdepth); // 'p' in 420p10 but not mono10
        }

        fprintf(file, "YUV4MPEG2 W%d H%d F25:1 Ip A0:0 C%s XCOLORRANGE=%s\nFRAME\n",
                        info->width, info->height, chroma_type, info->limited ? "LIMITED" : "FULL");
        errno = 0;
        size_t bytes_written = fwrite((const char *) data, 1, len, file);
        if (bytes_written != len) {
                fprintf(stderr, "Unable to write Y4M data - length %zd, written %zd: %s\n",
                        len, bytes_written, strerror(errno));
        }
        fclose(file);
        return bytes_written == len;
}

