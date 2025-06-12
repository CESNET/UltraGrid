/**
 * @file   utils/pam.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 *
 * Very simple library to read and write PAM/PPM files. Only binary formats
 * (P5, P6) for PNM are processed, P4 is also not used.
 *
 * This file is part of GPUJPEG.
 */
/*
 * Copyright (c) 2013-2025, CESNET
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

#include <ctype.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pam.h"

static bool parse_pam(FILE *file, struct pam_metadata *info) {
        char line[128];
        while (fgets(line, sizeof line - 1, file) != NULL) {
                if (strcmp(line, "ENDHDR\n") == 0) {
                        break;
                }
                if (line[0] == '#') {
                        continue;
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
                        info->ch_count = atoi(val);
                } else if (strcmp(key, "MAXVAL") == 0) {
                        info->maxval = atoi(val);
                } else if (strcmp(key, "TUPLTYPE") == 0) {
                        // ignored, value of DEPTH is sufficient to determine pixel format
                } else {
                        fprintf(stderr, "unrecognized key %s in PAM header\n", key);
                }
        }
        return true;
}

static bool check_nl(FILE *file) {
        if (getc(file) != '\n') {
                fprintf(stderr, "PNM maximal value isn't immediately followed by <NL>\n");
                return false;
        }
        return true;
}

static bool parse_pnm(FILE *file, char pnm_id, struct pam_metadata *info) {
        switch (pnm_id) {
                case '1':
                case '2':
                case '3':
                        fprintf(stderr, "Plain (ASCII) PNM are not supported, input is P%c\n", pnm_id);
                        return false;
                case '4':
                        info->ch_count = 1;
                        info->maxval = 1;
                        info->bitmap_pbm = true;
                        break;
                case '5':
                        info->ch_count = 1;
                        break;
                case '6':
                        info->ch_count = 3;
                        break;
                default:
                        fprintf(stderr, "Wrong PNM type P%c\n", pnm_id);
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
                                        if (info->bitmap_pbm) {
                                                return check_nl(file);
                                        }
                                        break;
                                case 2:
                                        info->maxval = val;
                                        return check_nl(file);
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

#ifdef _WIN32
#include <wchar.h>
#include <windows.h>
static wchar_t*
mbs_to_wstr_helper(const char* mbstr, wchar_t* wstr_buf, size_t wstr_len)
{
    const int size_needed = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, mbstr, -1, NULL, 0);
    if ( size_needed == 0 ) {
        fprintf(stderr, "[PAM] MultiByteToWideChar error: %d (0x%x)!\n", GetLastError(), GetLastError());
        return NULL;
    }
    if ( size_needed > (int)wstr_len ) {
        fprintf(stderr, "[PAM] buffer provided to %s too short - needed %d, got %zu!\n", __func__, size_needed,
                wstr_len);
        return NULL;
    }
    MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, mbstr, -1, wstr_buf, size_needed);
    return wstr_buf;
}
#define mbs_to_wstr(tstr) mbs_to_wstr_helper(tstr, (wchar_t[1024]){0}, 1024)
#endif

bool pam_read(const char *filename, struct pam_metadata *info, unsigned char **data, void *(*allocator)(size_t)) {
        char line[128];
        errno = 0;
#ifdef _WIN32
        FILE *file = _wfopen(mbs_to_wstr(filename), L"rb");
#else
        FILE *file = fopen(filename, "rb");
#endif
        if (!file) {
                fprintf(stderr, "Failed to open %s: %s\n", filename, strerror(errno));
                return false;
        }
        memset(info, 0, sizeof *info);
        if (fgets(line, 4, file) == NULL) {
                fprintf(stderr, "File '%s' read error: %s\n", filename, strerror(errno));
        }
        bool parse_rc = false;
        if (strcmp(line, "P7\n") == 0) {
                parse_rc = parse_pam(file, info);
        } else if (strlen(line) == 3 && line[0] == 'P' && isspace(line[2])) {
                parse_rc = parse_pnm(file, line[1], info);
        } else {
               fprintf(stderr, "File '%s' doesn't seem to be valid PAM or PNM.\n", filename);
        }
        if (!parse_rc) {
               fclose(file);
               return false;
        }
        if (info->width <= 0 || info->height <= 0) {
                fprintf(stderr, "Unspecified/incorrect size %dx%d!\n", info->width, info->height);
                parse_rc = false;
        }
        if (info->ch_count <= 0) {
                fprintf(stderr, "Unspecified/incorrect channel count %d!\n", info->ch_count);
                parse_rc = false;
        }
        if (info->maxval <= 0 || info->maxval > 65535) {
                fprintf(stderr, "Unspecified/incorrect maximal value %d!\n", info->maxval);
                parse_rc = false;
        }
        if (data == NULL || allocator == NULL || !parse_rc) {
                fclose(file);
                return parse_rc;
        }
        size_t datalen = (size_t) info->ch_count * info->width * info->height;
        if (info->maxval == 1 && info->bitmap_pbm) {
                datalen = (info->width + 7) / 8 * info->height;
        } else if (info->maxval > 255) {
                datalen *= 2;
        }
        *data = (unsigned char *) allocator(datalen);
        if (!*data) {
                fprintf(stderr, "Failed to allocate data!\n");
                fclose(file);
                return false;
        }
        size_t bytes_read = fread((char *) *data, 1, datalen, file);
        if (bytes_read != datalen) {
                fprintf(stderr, "Unable to load PAM/PNM data from file - read %zu B, expected %zu B: %s\n",
                                bytes_read, datalen, strerror(errno));
                fclose(file);
                return false;
        }
        fclose(file);
        return true;
}

bool pam_write(const char *filename, unsigned int width, unsigned int height, int ch_count, int maxval, const unsigned char *data, bool pnm) {
        errno = 0;
#ifdef _WIN32
        FILE *file = _wfopen(mbs_to_wstr(filename), L"wb");
#else
        FILE *file = fopen(filename, "wb");
#endif
        if (!file) {
                fprintf(stderr, "Failed to open %s for writing: %s\n", filename, strerror(errno));
                return false;
        }
        if (pnm) {
                if (ch_count != 1 && ch_count != 3) {
                        fprintf(stderr, "Only 1 or 3 channels supported for PNM!\n");
                        fclose(file);
                        return false;
                }
                fprintf(file, "P%d\n"
                        "%u %u\n"
                        "%d\n",
                        ch_count == 1 ? 5 : 6,
                        width, height, maxval);
        } else {
                const char *tuple_type = "INVALID";
                switch (ch_count) {
                        case 4: tuple_type = "RGB_ALPHA"; break;
                        case 3: tuple_type = "RGB"; break;
                        case 2: tuple_type = "GRAYSCALE_ALPHA"; break;
                        case 1: tuple_type = "GRAYSCALE"; break;
                        default: fprintf(stderr, "Wrong channel count: %d\n", ch_count);
                }
                fprintf(file, "P7\n"
                        "WIDTH %u\n"
                        "HEIGHT %u\n"
                        "DEPTH %d\n"
                        "MAXVAL %d\n"
                        "TUPLTYPE %s\n"
                        "ENDHDR\n",
                        width, height, ch_count, maxval, tuple_type);
        }
        size_t len = (size_t) width * height * ch_count * (maxval <= 255 ? 1 : 2);
        errno = 0;
        size_t bytes_written = fwrite((const char *) data, 1, len, file);
        if (bytes_written != len) {
                fprintf(stderr, "Unable to write PAM/PNM data - length %zd, written %zd: %s\n",
                        len, bytes_written, strerror(errno));
        }
        fclose(file);
        return bytes_written == len;
}

