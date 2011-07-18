/*
 * FILE:   quicktime.c
 * AUTHOR: Colin Perkins <csp@csperkins.org
 *         Alvaro Saurin <saurin@dcs.gla.ac.uk>
 *         Martin Benes     <martinbenesh@gmail.com>
 *         Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 *         Petr Holub       <hopet@ics.muni.cz>
 *         Milos Liska      <xliska@fi.muni.cz>
 *         Jiri Matela      <matela@ics.muni.cz>
 *         Dalibor Matura   <255899@mail.muni.cz>
 *         Ian Wesley-Smith <iwsmith@cct.lsu.edu>
 *
 * Copyright (c) 2005-2006 University of Glasgow
 * Copyright (c) 2005-2010 CESNET z.s.p.o.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, is permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 * 
 *      This product includes software developed by the University of Southern
 *      California Information Sciences Institute. This product also includes
 *      software developed by CESNET z.s.p.o.
 * 
 * 4. Neither the name of the University, Institute, CESNET nor the names of
 *    its contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHORS AND CONTRIBUTORS
 * ``AS IS'' AND ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING,
 * BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
 * AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * $Revision: 1.15.2.3 $
 * $Date: 2010/02/04 15:51:33 $
 *
 */

#include "host.h"
#include "config.h"
#include "config_unix.h"
#include "debug.h"
#include "tv.h"
#include "video_display.h"
#include "video_capture.h"
#include "video_capture/quicktime.h"
#include "video_display/quicktime.h"
#include "video_codec.h"
#include <math.h>

#ifdef HAVE_MACOSX
#include <Carbon/Carbon.h>
#include <QuickTime/QuickTime.h>
#include <QuickTime/QuickTimeComponents.h>

#define MAGIC_QT_GRABBER	VIDCAP_QUICKTIME_ID

struct qt_grabber_state {
        uint32_t magic;
        SeqGrabComponent grabber;
        SGChannel video_channel;
        Rect bounds;
        GWorldPtr gworld;
        ImageSequence seqID;
        int sg_idle_enough;
        int major;
        int minor;
        struct video_frame frame;
        const struct codec_info_t *c_info;
        unsigned gui:1;
        int frames;
        struct timeval t0;
        const quicktime_mode_t *qt_mode;
};

/*
 * Sequence grabber data procedure
 * The sequence grabber calls the data function whenever any of the grabbers
 * channels write digitized data to the destination movie file.
 *
 * This data function does two things, it first decompresses captured video
 * data into an offscreen GWorld, draws some status information onto the frame then
 * transfers the frame to an onscreen window.
 *
 * For more information refer to Inside Macintosh: QuickTime Components, page 5-120
 * c - the channel component that is writing the digitized data.
 * p - a pointer to the digitized data.
 * len - the number of bytes of digitized data.
 * offset - a pointer to a field that may specify where you are to write the
 * digitized data, and that is to receive a value indicating where you wrote the data.
 * chRefCon - per channel reference constant specified using SGSetChannelRefCon.
 * time    - the starting time of the data, in the channel's time scale.
 * writeType - the type of write operation being performed.
 * 	seqGrabWriteAppend  - Append new data.
 * 	seqGrabWriteReserve - Do not write data. Instead, reserve space for the amount of data
 * 	                      specified in the len parameter.
 * 	seqGrabWriteFill    - Write data into the location specified by offset. Used to
 * 	                      fill the space previously reserved with seqGrabWriteReserve. 
 *	                      The Sequence Grabber may call the DataProc several times to 
 *	                      fill a single reserved location.
 * refCon - the reference constant you specified when you assigned your data
 *          function to the sequence grabber.
 */

static pascal OSErr
qt_data_proc(SGChannel c, Ptr p, long len, long *offset, long chRefCon,
             TimeValue time, short writeType, long refCon)
{
        struct qt_grabber_state *s = (struct qt_grabber_state *)refCon;
        struct timeval t;

        UNUSED(c);
        UNUSED(offset);
        UNUSED(chRefCon);
        UNUSED(time);
        UNUSED(writeType);

        if (s == NULL) {
                debug_msg("corrupt state\n");
                return -1;
        }

        memcpy(s->frame.data, p, len);
        s->sg_idle_enough = 1;

        s->frames++;
        gettimeofday(&t, NULL);
        double seconds = tv_diff(t, s->t0);
        if (seconds >= 5) {
                float fps = s->frames / seconds;
                fprintf(stderr, "%d frames in %g seconds = %g FPS\n", s->frames,
                        seconds, fps);
                s->t0 = t;
                s->frames = 0;
        }

        return 0;
}

static Boolean
SeqGrabberModalFilterProc(DialogPtr theDialog, const EventRecord * theEvent,
                          short *itemHit, long refCon)
{
        UNUSED(theDialog);
        UNUSED(itemHit);

        // Ordinarily, if we had multiple windows we cared about, we'd handle
        // updating them in here, but since we don't, we'll just clear out
        // any update events meant for us

        Boolean handled = false;

        if ((theEvent->what == updateEvt) &&
            ((WindowPtr) theEvent->message == (WindowPtr) refCon)) {
                BeginUpdate((WindowPtr) refCon);
                EndUpdate((WindowPtr) refCon);
                handled = true;
        }
        return (handled);
}

//  SGSettingsDialog with the "Compression" panel removed
static OSErr MinimalSGSettingsDialog(SeqGrabComponent seqGrab,
                                     SGChannel sgchanVideo, WindowPtr gMonitor)
{
        OSErr err;
        Component *panelListPtr = NULL;
        UInt8 numberOfPanels = 0;

        ComponentDescription cd = { SeqGrabPanelType, VideoMediaType, 0, 0, 0 };
        Component c = 0;
        Component *cPtr = NULL;

        numberOfPanels = CountComponents(&cd);
        panelListPtr =
            (Component *) NewPtr(sizeof(Component) * (numberOfPanels + 1));

        cPtr = panelListPtr;
        numberOfPanels = 0;
        CFStringRef compressionCFSTR = CFSTR("Compression");
        do {
                ComponentDescription compInfo;
                c = FindNextComponent(c, &cd);
                if (c) {
                        Handle hName = NewHandle(0);
                        GetComponentInfo(c, &compInfo, hName, NULL, NULL);
                        CFStringRef nameCFSTR =
                            CFStringCreateWithPascalString(kCFAllocatorDefault,
                                                           (unsigned char
                                                            *)(*hName),
                                                           kCFStringEncodingASCII);
                        if (CFStringCompare
                            (nameCFSTR, compressionCFSTR,
                             kCFCompareCaseInsensitive) != kCFCompareEqualTo) {
                                *cPtr++ = c;
                                numberOfPanels++;
                        }
                        DisposeHandle(hName);
                }
        } while (c);

        if ((err =
             SGSettingsDialog(seqGrab, sgchanVideo, numberOfPanels,
                              panelListPtr, seqGrabSettingsPreviewOnly,
                              (SGModalFilterUPP)
                              NewSGModalFilterUPP(SeqGrabberModalFilterProc),
                              (long)gMonitor))) {
                return err;
        }
        return 0;
}

static void 
nprintf(unsigned char *str)
{
        char tmp[((int)str[0]) + 1];

        strncpy(tmp, (char*)(&str[1]), str[0]);
        tmp[(int)str[0]] = 0;
        fprintf(stdout, "%s", tmp);
}

static void 
shrink(unsigned char *str)
{
        int i, j;
        j=1;
        for(i=1; i <= str[0]; i++) {
           while((str[i] == '\t' ||
                 str[i] == ' ') && i < str[0])
                i++; 
           if(i <= str[0]) {
                if(str[i] >= 'a' && str[i] <= 'z')
                    str[j] = str[i] - ('a'-'A');
                else
                    str[j] = str[i];
                j++;
            }
        }
        str[0] = j-1;
}

static unsigned char * 
shrink2(unsigned char *str)
{
        int i, j;
        int len = strlen((char*)str);
        str = (unsigned char*)strdup((char*)str);
        j=0;
        for(i=0; i < len; i++) {
           while((str[i] == '\t' ||
                 str[i] == ' ') && i < str[0])
                i++; 
           if(i < len) {
                if(str[i] >= 'a' && str[i] <= 'z')
                    str[j] = str[i] - ('a'-'A');
                else
                    str[j] = str[i];
                j++;
            }
        }
        return str;
}

/* Initialize the QuickTime grabber */
static int qt_open_grabber(struct qt_grabber_state *s, char *fmt)
{
        GrafPtr savedPort;
        WindowPtr gMonitor;
        //SGModalFilterUPP      seqGrabModalFilterUPP;

        assert(s != NULL);
        assert(s->magic == MAGIC_QT_GRABBER);

        /****************************************************************************************/
        /* Step 0: Initialise the QuickTime movie toolbox.                                      */
        InitCursor();
        EnterMovies();

        /****************************************************************************************/
        /* Step 1: Create an off-screen graphics world object, into which we can capture video. */
        /* Lock it into position, to prevent QuickTime from messing with it while capturing.    */
        OSType pixelFormat;
        pixelFormat = FOUR_CHAR_CODE('BGRA');
        /****************************************************************************************/
        /* Step 2: Open and initialise the default sequence grabber.                            */
        s->grabber = OpenDefaultComponent(SeqGrabComponentType, 0);
        if (s->grabber == 0) {
                debug_msg("Unable to open grabber\n");
                return 0;
        }

        gMonitor = GetDialogWindow(GetNewDialog(1000, NULL, (WindowPtr) - 1L));

        GetPort(&savedPort);
        SetPort((GrafPtr)gMonitor);

        if (SGInitialize(s->grabber) != noErr) {
                debug_msg("Unable to init grabber\n");
                return 0;
        }

        SGSetGWorld(s->grabber, GetDialogPort((DialogPtr)gMonitor), NULL);

        /****************************************************************************************/
        /* Specify the destination data reference for a record operation tell it */
        /* we're not making a movie if the flag seqGrabDontMakeMovie is used,    */
        /* the sequence grabber still calls your data function, but does not     */
        /* write any data to the movie file writeType will always be set to      */
        /* seqGrabWriteAppend                                                    */
        if (SGSetDataRef(s->grabber, 0, 0, seqGrabDontMakeMovie) != noErr) {
                CloseComponent(s->grabber);
                debug_msg("Unable to set data ref\n");
                return 0;
        }

        if (SGSetGWorld(s->grabber, NULL, NULL) != noErr) {
                debug_msg("Unable to get gworld from grabber\n");
                return 0;
        }

        if (SGNewChannel(s->grabber, VideoMediaType, &s->video_channel) !=
            noErr) {
                debug_msg("Unable to open video channel\n");
                return 0;
        }

        /* Print available devices */
        int i;
        int j;
        SGDeviceInputList inputList;
        SGDeviceList deviceList;
        if (strcmp(fmt, "help") == 0) {
                if (SGGetChannelDeviceList
                    (s->video_channel, sgDeviceListIncludeInputs,
                     &deviceList) == noErr) {
                        fprintf(stdout, "Available capture devices:\n");
                        for (i = 0; i < (*deviceList)->count; i++) {
                                SGDeviceName *deviceEntry =
                                    &(*deviceList)->entry[i];
                                fprintf(stdout, " Device %d: ", i);
                                nprintf(deviceEntry->name);
                                if (deviceEntry->flags &
                                    sgDeviceNameFlagDeviceUnavailable) {
                                        fprintf(stdout,
                                                "  - ### NOT AVAILABLE ###");
                                }
                                if (i == (*deviceList)->selectedIndex) {
                                        fprintf(stdout, " - ### ACTIVE ###");
                                }
                                fprintf(stdout, "\n");
                                short activeInputIndex = 0;
                                inputList = deviceEntry->inputs;
                                if (inputList && (*inputList)->count >= 1) {
                                        SGGetChannelDeviceAndInputNames
                                            (s->video_channel, NULL, NULL,
                                             &activeInputIndex);
                                        for (j = 0; j < (*inputList)->count;
                                             j++) {
                                                fprintf(stdout, "\t");
                                                fprintf(stdout, "- %d. ", j);
                                                nprintf((unsigned char*)&(*inputList)->entry
                                                             [j].name);
                                                if ((i ==
                                                     (*deviceList)->selectedIndex)
                                                    && (j == activeInputIndex))
                                                        fprintf(stdout,
                                                                " - ### ACTIVE ###");
                                                fprintf(stdout, "\n");
                                        }
                                }
                        }
                        SGDisposeDeviceList(s->grabber, deviceList);
                        CodecNameSpecListPtr list;
                        GetCodecNameList(&list, 1);
                        printf("Compression types:\n");
                        for (i = 0; i < list->count; i++) {
                                int fcc = list->list[i].cType;
                                printf("\t%d) ", i);
                                nprintf(list->list[i].typeName);
                                printf(" - FCC (%c%c%c%c)",
                                       fcc >> 24,
                                       (fcc >> 16) & 0xff,
                                       (fcc >> 8) & 0xff, (fcc) & 0xff);
                                printf(" - codec id %x",
                                       (unsigned int)(list->list[i].codec));
                                printf(" - cType %x",
                                       (unsigned int)list->list[i].cType);
                                printf("\n");
                        }
                }
                return 0;
        }

        if (SGSetChannelUsage
            (s->video_channel,
             seqGrabRecord | seqGrabPreview | seqGrabAlwaysUseTimeBase |
             seqGrabLowLatencyCapture) != noErr) {
                debug_msg("Unable to set channel usage\n");
                return 0;
        }

        if (SGSetChannelPlayFlags(s->video_channel, channelPlayAllData) !=
            noErr) {
                debug_msg("Unable to set channel flags\n");
                return 0;
        }

        SGStartPreview(s->grabber);

        /* Select the device */
        if (strcmp(fmt, "gui") == 0) {  //Use gui to select input
                MinimalSGSettingsDialog(s->grabber, s->video_channel, gMonitor);
        } else {                // Use input specified on cmd
                if (SGGetChannelDeviceList
                    (s->video_channel, sgDeviceListIncludeInputs,
                     &deviceList) != noErr) {
                        debug_msg("Unable to get list of quicktime devices\n");
                        return 0;
                }

                char *tmp;

                tmp = strtok(fmt, ":");
                if (!tmp) {
                        fprintf(stderr, "Wrong config %s\n", fmt);
                        return 0;
                }
                s->major = atoi(tmp);
                tmp = strtok(NULL, ":");
                if (!tmp) {
                        fprintf(stderr, "Wrong config %s\n", fmt);
                        return 0;
                }
                s->minor = atoi(tmp);
                tmp = strtok(NULL, ":");
                if (!tmp) {
                        fprintf(stderr, "Wrong config %s\n", fmt);
                        return 0;
                }
                s->frame.color_spec = atoi(tmp);

                SGDeviceName *deviceEntry = &(*deviceList)->entry[s->major];
                printf("Quicktime: Setting device: ");
                nprintf(deviceEntry->name);
                printf("\n");
                if (SGSetChannelDevice(s->video_channel, deviceEntry->name) !=
                    noErr) {
                        debug_msg("Setting up the selected device failed\n");
                        return 0;
                }

                /* Select input */
                inputList = deviceEntry->inputs;
                printf("Quicktime: Setting input: ");
                nprintf((unsigned char *)(&(*inputList)->entry[s->minor].name));
                printf("\n");
                if (SGSetChannelDeviceInput(s->video_channel, s->minor) !=
                    noErr) {
                        debug_msg
                            ("Setting up input on selected device failed\n");
                        return 0;
                }
        }

        /* Set video size according to selected video format */
        Rect gActiveVideoRect;
        SGGetSrcVideoBounds(s->video_channel, &gActiveVideoRect);

        s->frame.width = s->bounds.right =
            gActiveVideoRect.right - gActiveVideoRect.left;
        s->frame.height = s->bounds.bottom =
            gActiveVideoRect.bottom - gActiveVideoRect.top;

        unsigned char *deviceName;
        unsigned char *inputName;
        short  inputNumber;

        if ((SGGetChannelDeviceList(s->video_channel, sgDeviceListIncludeInputs, &deviceList) != noErr) ||
            (SGGetChannelDeviceAndInputNames(s->video_channel, NULL, NULL, &inputNumber)!=noErr)) {
                debug_msg("Unable to query channel settings\n");
                return 0;
        }

        SGDeviceName *deviceEntry = &(*deviceList)->entry[(*deviceList)->selectedIndex];
        inputList = deviceEntry->inputs;

        deviceName = deviceEntry->name;
        inputName = (unsigned char*)&(*inputList)->entry[inputNumber].name;

        shrink(deviceName);
        shrink(inputName);

        for(i=0; quicktime_modes[i].device != NULL; i++) {
                unsigned char *device = shrink2(quicktime_modes[i].device);
                unsigned char *input = shrink2(quicktime_modes[i].input);

                if((strncmp((char*)device, (char*)&deviceName[1], deviceName[0])) == 0 &&
                   (strncmp((char*)input, (char*)&inputName[1], inputName[0])) == 0) {
                        s->qt_mode = &quicktime_modes[i];
                        printf("Quicktime: mode should be: %dx%d@%0.2ffps, flags: 0x%x\n",
                                        s->qt_mode->width,
                                        s->qt_mode->height,
                                        s->qt_mode->fps,
                                        s->qt_mode->aux);
                        s->frame.fps = s->qt_mode->fps;
                        s->frame.aux = s->qt_mode->aux & (AUX_INTERLACED|AUX_PROGRESSIVE|AUX_SF);
                        free(device);
                        free(input);
                        break;
                }
                free(device);
                free(input);
        }
        if (s->qt_mode == NULL) {
                fprintf(stdout, "\n\nQuicktime WARNING: device ");
                nprintf(deviceName);
                fprintf(stdout, " \n\twith input ");
                nprintf(inputName);
                fprintf(stdout, " was not found in mode table.\n"
                                "\tPlease report it to xhejtman@ics.muni.cz\n\n");
        }

        printf("Quicktime: Video size: %dx%d\n", s->bounds.right,
               s->bounds.bottom);

        if (SGSetChannelBounds(s->video_channel, &(s->bounds)) != noErr) {
                debug_msg("Unable to set channel bounds\n");
                return 0;
        }

        /* Set selected fmt->codec and get pixel format of that codec */
        int pixfmt;
        if (s->frame.color_spec != 0xffffffff) {
                CodecNameSpecListPtr list;
                GetCodecNameList(&list, 1);
                pixfmt = list->list[s->frame.color_spec].cType;
                printf("Quicktime: SetCompression: %s\n",
                       (int)SGSetVideoCompressor(s->video_channel, 0,
                                                 list->list[s->frame.color_spec].codec, 0,
                                                 0, 0)==0?"OK":"Failed!");
        } else {
                CompressorComponent  codec;
                SGGetVideoCompressor(s->video_channel, NULL, &codec, NULL, NULL,
                                     NULL);
                CodecNameSpecListPtr list;
                GetCodecNameList(&list, 1);
                for (i = 0; i < list->count; i++) {
                        if (codec == list->list[i].codec) {
                                pixfmt = list->list[i].cType;
                                break;
                        }
                }
        }

        for (i = 0; codec_info[i].name != NULL; i++) {
                if ((unsigned)pixfmt == codec_info[i].fcc) {
                        s->c_info = &codec_info[i];
                        s->frame.color_spec = s->c_info->codec;
                        break;
                }
        }

        printf("Quicktime: Selected pixel format: %c%c%c%c\n",
               pixfmt >> 24, (pixfmt >> 16) & 0xff, (pixfmt >> 8) & 0xff,
               (pixfmt) & 0xff);

        int h_align = s->c_info->h_align;
        int aligned_x=s->frame.width;
        if (h_align) {
                aligned_x = ((s->frame.width + h_align - 1) / h_align) * h_align;
                printf
                    ("Quicktime: Pixel format 'v210' was selected -> Setting width to %d\n",
                     aligned_x);
        }

        s->frame.data_len = aligned_x * s->frame.height * s->c_info->bpp;
        s->frame.data = malloc(s->frame.data_len);

        SetPort(savedPort);

        /****************************************************************************************/
        /* Step ?: Set the data procedure, which processes the frames as they're captured.      */
        SGSetDataProc(s->grabber, NewSGDataUPP(qt_data_proc), (long)s);

        /****************************************************************************************/
        /* Step ?: Start capturing video...                                                     */
        if (SGPrepare(s->grabber, FALSE, TRUE) != noErr) {
                debug_msg("Unable to prepare capture\n");
                return 0;
        }

        if (SGStartRecord(s->grabber) != noErr) {
                debug_msg("Unable to start recording\n");
                return 0;
        }

        return 1;
}

/*******************************************************************************
 * Public API
 ******************************************************************************/
struct vidcap_type *vidcap_quicktime_probe(void)
{
        struct vidcap_type *vt;

        vt = (struct vidcap_type *)malloc(sizeof(struct vidcap_type));
        if (vt != NULL) {
                vt->id = VIDCAP_QUICKTIME_ID;
                vt->name = "quicktime";
                vt->description = "QuickTime capture device";
        }

        return vt;
}

/* Initialize the QuickTime grabbing system */
void *vidcap_quicktime_init(char *fmt)
{
        struct qt_grabber_state *s;

        s = (struct qt_grabber_state *)calloc(1,sizeof(struct qt_grabber_state));
        if (s != NULL) {
                s->magic = MAGIC_QT_GRABBER;
                s->grabber = 0;
                s->video_channel = 0;
                s->seqID = 0;
                s->bounds.top = 0;
                s->bounds.left = 0;
                s->sg_idle_enough = 0;
                s->frame.color_spec = 0xffffffff;

                if (qt_open_grabber(s, fmt) == 0) {
                        free(s);
                        return NULL;
                }
        }

        return s;
}

/* Finalize the grabbing system */
void vidcap_quicktime_done(void *state)
{
        struct qt_grabber_state *s = (struct qt_grabber_state *)state;

        assert(s != NULL);

        if (s != NULL) {
                assert(s->magic == MAGIC_QT_GRABBER);
                SGStop(s->grabber);
                CloseComponent(s->grabber);
                ExitMovies();
                free(s);
        }
}

/* Grab a frame */
struct video_frame *vidcap_quicktime_grab(void *state, int *count )
{
        struct qt_grabber_state *s = (struct qt_grabber_state *)state;

        assert(s != NULL);
        assert(s->magic == MAGIC_QT_GRABBER);

        /* Run the QuickTime sequence grabber idle function, which provides */
        /* processor time to out data proc running as a callback.           */

        /* The while loop done in this way is also sort of nice bussy waiting */
        /* and synchronizes capturing and sending.                            */
        s->sg_idle_enough = 0;
        while (!s->sg_idle_enough) {
                if (SGIdle(s->grabber) != noErr) {
                        debug_msg("Error in SGIDle\n");
                        *count = 0;
                        return NULL;
                }
        }

        *count = 1;
        return &s->frame;
}

#endif                          /* HAVE_MACOSX */
