/*
 * FILE:    test_video_display.c
 * AUTHORS: Colin Perkins
 *
 * Test video hardware probing routines
 *
 * Copyright (c) 2003-2004 University of Southern California
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, is permitted provided that the following conditions 
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *      This product includes software developed by the University of 
 *      Southern California Information Sciences Institute
 * 4. Neither the name of the University nor of the Institute may be used
 *    to endorse or promote products derived from this software without
 *    specific prior written permission.
 * THIS SOFTWARE IS PROVIDED BY THE AUTHORS AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */

#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#include "debug.h"
#include "memory.h"
#include "video_types.h"
#include "video_display.h"
#include "test_video_display.h"

int test_video_display(void)
{
        display_type_t *dt;
        int i;
        unsigned int j;
        int argc = 0;
        char *argv[1];

        argv[0] = "run_tests";

        printf
            ("Testing video hardware detection ......................................... ");
        if (display_init_devices() != 0) {
                printf("FAIL\n");
                printf("    Cannot probe video devices\n");
                return 1;
        }
        printf("Ok\n");

        for (i = 0; i < display_get_device_count(); i++) {
                dt = display_get_device_details(i);
                printf("    \"%s\"\n", dt->name);
                printf("        description: %s\n", dt->description);
                printf("        formats    :");
                for (j = 0; j < dt->num_formats; j++) {
                        switch (dt->formats[j].size) {
                        case DS_176x144:
                                printf(" QCIF");
                                break;
                        case DS_352x288:
                                printf(" CIF");
                                break;
                        case DS_702x576:
                                printf(" SCIF");
                                break;
                        case DS_1920x1080:
                                printf(" 1080i");
                                break;
                        case DS_1280x720:
                                printf(" 720p");
                                break;
                        case DS_NONE:
                                printf(" NONE");
                                continue;
                        }
                        switch (dt->formats[j].colour_mode) {
                        case DC_YUV:
                                printf("/YUV");
                                break;
                        case DC_RGB:
                                printf("/RGB");
                                break;
                        case DC_NONE:
                                printf("/NONE");
                                break;
                        }
                        if (dt->formats[j].num_images != -1) {
                                printf("/%d", dt->formats[j].num_images);
                        }
                }
                printf("\n");
        }
        display_free_devices();
        return 0;
}
