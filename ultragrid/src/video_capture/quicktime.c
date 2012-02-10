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
 */

#include "config.h"
#include "config_unix.h"
#include "debug.h"
#include "host.h"
#include "tv.h"
#include "video_display.h"
#include "video_capture.h"
#include "video_capture/quicktime.h"
#include "video_display/quicktime.h"
#include "video_codec.h"
#include "audio/audio.h"
#include <math.h>

#ifdef HAVE_MACOSX
#include <Carbon/Carbon.h>
#include <QuickTime/QuickTime.h>
#include <QuickTime/QuickTimeComponents.h>

#define MAGIC_QT_GRABBER        VIDCAP_QUICKTIME_ID

struct qt_grabber_state {
        uint32_t magic;
        SeqGrabComponent grabber;
        SGChannel video_channel;
        SGChannel audio_channel;
        Rect bounds;
        GWorldPtr gworld;
        ImageSequence seqID;
        int sg_idle_enough;
        int major;
        int minor;
        int audio_major;
        int audio_minor;
        struct video_frame *frame;
        struct tile *tile;
        struct audio_frame audio;
        char *abuffer[2], *vbuffer[2];
        int abuffer_len;
        int grab_buf_idx;
        const struct codec_info_t *c_info;
        unsigned gui:1;
        int frames;
        struct timeval t0;
        const quicktime_mode_t *qt_mode;
        unsigned int grab_audio:1;

        pthread_t thread_id;
        pthread_mutex_t lock;
        pthread_cond_t boss_cv;
        pthread_cond_t worker_cv;
        volatile int boss_waiting;
        volatile int worker_waiting;
        volatile int work_to_do;
};

void * vidcap_quicktime_thread(void *state);
void InitCursor(void);
void GetPort(GrafPtr *port);
void SetPort(GrafPtr port);

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
 *      seqGrabWriteAppend  - Append new data.
 *      seqGrabWriteReserve - Do not write data. Instead, reserve space for the amount of data
 *                            specified in the len parameter.
 *      seqGrabWriteFill    - Write data into the location specified by offset. Used to
 *                            fill the space previously reserved with seqGrabWriteReserve. 
 *                            The Sequence Grabber may call the DataProc several times to 
 *                            fill a single reserved location.
 * refCon - the reference constant you specified when you assigned your data
 *          function to the sequence grabber.
 */

static pascal OSErr
qt_data_proc(SGChannel c, Ptr p, long len, long *offset, long chRefCon,
             TimeValue time, short writeType, long refCon)
{
        struct qt_grabber_state *s = (struct qt_grabber_state *)refCon;
        struct timeval t;

        UNUSED(offset);
        UNUSED(chRefCon);
        UNUSED(time);

        if (s == NULL) {
                debug_msg("corrupt state\n");
                return -1;
        }

        if(c == s->video_channel) {
                memcpy(s->vbuffer[s->grab_buf_idx], p, len);
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
        } else if(c == s->audio_channel) {
                switch(writeType) {
                        case seqGrabWriteReserve:
                                break;
                        case seqGrabWriteFill:
                                memcpy(s->abuffer[s->grab_buf_idx] + s->abuffer_len, p, len);
                                s->abuffer_len += len;
                                break;
                        case seqGrabWriteAppend:
                                break;
                }
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
nprintf(char *str)
{
        char tmp[((int)str[0]) + 1];

        strncpy(tmp, (char*)(&str[1]), str[0]);
        tmp[(int)str[0]] = 0;
        fprintf(stdout, "%s", tmp);
}

static void 
shrink(char *str)
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

static char * 
shrink2(char *str)
{
        int i, j;
        int len = strlen((char*)str);
        str = strdup((char*)str);
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

/*
 * If there are more than one Blackmagic card, QT appends its index to its name:
 * Blackmagic 2 HD 1080
 * ....
 */ 
static char * 
delCardIndicesCard(char *str)
{
        if(strncmp((char *) str, "Blackmagic", strlen("Blackmagic")) == 0) {
                if(isdigit(str[strlen("Blackmagic") + 1])) { // Eg. Blackmagic X ...
                        int len = 
                                strlen((char *) str + strlen("Blackmagic") + 2);
                        memmove(str + strlen("Blackmagic"),
                                str + strlen("Blackmagic") + 2,
                                len);
                        str[strlen("Blackmagic") + len] = '\0';
                }
        }
        return str;
}

/*
 * ... and to its modes:
 * eg.: Blackmagic HD 1080i 50 - 8 Bit (2)
 */
static char * 
delCardIndicesMode(char *str)
{
        if(strncmp((char *) str, "Blackmagic", strlen("Blackmagic")) == 0) {
                if(str[strlen((char *)str) - 1 - 2] == '('
                                && isdigit(str[strlen((char *) str) - 1 - 1])
                                && str[strlen((char *) str) - 1] == ')') {
                        str[strlen((char *) str) - 4] = '\0'; // Eg.: Blackmagic ... (x)
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

        /* do not check for grab audio in case that we will only display usage */
        if (SGNewChannel(s->grabber, SoundMediaType, &s->audio_channel) !=
                noErr) {
                fprintf(stderr, "Warning: Creating audio channel failed. "
                                "Disabling sound output.\n");
                s->grab_audio = FALSE;
        }

        /* Print available devices */
        int i;
        int j;
        SGDeviceInputList inputList;
        SGDeviceList deviceList;
        if (strcmp(fmt, "help") == 0) {
                printf("\nUsage:\t-t quicktime:<device>:<mode>:<pixel_type>[:<audio_device>:<audio_mode>]\n\n");
                if (SGGetChannelDeviceList
                    (s->video_channel, sgDeviceListIncludeInputs,
                     &deviceList) == noErr) {
                        fprintf(stdout, "\nAvailable capture devices:\n");
                        for (i = 0; i < (*deviceList)->count; i++) {
                                SGDeviceName *deviceEntry =
                                    &(*deviceList)->entry[i];
                                fprintf(stdout, " Device %d: ", i);
                                nprintf((char *) deviceEntry->name);
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
                                                nprintf((char *) &(*inputList)->entry
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
                        printf("\nCompression types:\n");
                        for (i = 0; i < list->count; i++) {
                                int fcc = list->list[i].cType;
                                printf("\t%d) ", i);
                                nprintf((char *) list->list[i].typeName);
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
                if (SGGetChannelDeviceList
                    (s->audio_channel, sgDeviceListIncludeInputs,
                     &deviceList) == noErr) {
                        fprintf(stdout, "\nAvailable audio devices:\n");
                        for (i = 0; i < (*deviceList)->count; i++) {
                                SGDeviceName *deviceEntry =
                                    &(*deviceList)->entry[i];
                                fprintf(stdout, " Device %d: ", i);
                                nprintf((char *) deviceEntry->name);
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
                                                nprintf((char*)&(*inputList)->entry
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

        if(s->grab_audio) {
                if (SGSetChannelUsage
                    (s->audio_channel,
                     seqGrabRecord) != noErr) {
                        fprintf(stderr, "Unable to set audio channel usage\n");
                        s->grab_audio = FALSE;
                }

                if (SGSetChannelPlayFlags(s->audio_channel, channelPlayAllData) !=
                    noErr) {
                        fprintf(stderr, "Unable to set channel flags\n");
                        s->grab_audio = FALSE;
                }
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
                s->frame->color_spec = atoi(tmp);

                s->audio_major = -1;
                s->audio_minor = -1;
                tmp = strtok(NULL, ":");
                if(tmp) s->audio_major = atoi(tmp);
                tmp = strtok(NULL, ":");
                if(tmp) s->audio_minor = atoi(tmp);
                if(s->audio_major == -1 && s->audio_minor == -1)
                        s->grab_audio = FALSE;

                SGDeviceName *deviceEntry = &(*deviceList)->entry[s->major];
                printf("Quicktime: Setting device: ");
                nprintf((char *) deviceEntry->name);
                printf("\n");
                if (SGSetChannelDevice(s->video_channel, deviceEntry->name) !=
                    noErr) {
                        debug_msg("Setting up the selected device failed\n");
                        return 0;
                }

                /* Select input */
                inputList = deviceEntry->inputs;
                printf("Quicktime: Setting input: ");
                nprintf((char *)(&(*inputList)->entry[s->minor].name));
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

        s->tile->width = s->bounds.right =
            gActiveVideoRect.right - gActiveVideoRect.left;
        s->tile->height = s->bounds.bottom =
            gActiveVideoRect.bottom - gActiveVideoRect.top;

        char *deviceName;
        char *inputName;
        short  inputNumber;
        TimeScale scale;

        if ((SGGetChannelDeviceList(s->video_channel, sgDeviceListIncludeInputs, &deviceList) != noErr) ||
            (SGGetChannelDeviceAndInputNames(s->video_channel, NULL, NULL, &inputNumber)!=noErr)) {
                debug_msg("Unable to query channel settings\n");
                return 0;
        }

        SGDeviceName *deviceEntry = &(*deviceList)->entry[(*deviceList)->selectedIndex];
        inputList = deviceEntry->inputs;

        deviceName = (char *) deviceEntry->name;
        inputName = (char*)&(*inputList)->entry[inputNumber].name;

        shrink(delCardIndicesCard(deviceName));
        shrink(delCardIndicesMode(inputName));

        for(i=0; quicktime_modes[i].device != NULL; i++) {
                char *device = shrink2(quicktime_modes[i].device);
                char *input = shrink2(quicktime_modes[i].input);

                if((strncmp((char*)device, (char*)&deviceName[1], deviceName[0])) == 0 &&
                   (strncmp((char*)input, (char*)&inputName[1], inputName[0])) == 0) {
                        s->qt_mode = &quicktime_modes[i];
                        printf("Quicktime: mode should be: %dx%d@%0.2ffps, flags: 0x%x\n",
                                        s->qt_mode->width,
                                        s->qt_mode->height,
                                        s->qt_mode->fps,
                                        s->qt_mode->aux);
                        switch(s->qt_mode->aux & (AUX_INTERLACED|AUX_PROGRESSIVE|AUX_SF)) {
                                case AUX_PROGRESSIVE:
                                        s->frame->interlacing = PROGRESSIVE;
                                        break;
                                case AUX_INTERLACED:
                                        s->frame->interlacing = INTERLACED_MERGED;
                                        break;
                                case AUX_SF:
                                        s->frame->interlacing = SEGMENTED_FRAME;
                                        break;
                        }
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
        int pixfmt = -1;
        if (s->frame->color_spec != 0xffffffff) {
                CodecNameSpecListPtr list;
                GetCodecNameList(&list, 1);
                pixfmt = list->list[s->frame->color_spec].cType;
                printf("Quicktime: SetCompression: %s\n",
                       (int)SGSetVideoCompressor(s->video_channel, 0,
                                                 list->list[s->frame->color_spec].codec, 0,
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
        if(pixfmt == -1) {
                fprintf(stderr, "[QuickTime] Unknown pixelformat!\n");
                goto error;
        }

        for (i = 0; codec_info[i].name != NULL; i++) {
                if ((unsigned)pixfmt == codec_info[i].fcc) {
                        s->c_info = &codec_info[i];
                        s->frame->color_spec = s->c_info->codec;
                        break;
                }
        }

        printf("Quicktime: Selected pixel format: %c%c%c%c\n",
               pixfmt >> 24, (pixfmt >> 16) & 0xff, (pixfmt >> 8) & 0xff,
               (pixfmt) & 0xff);

        s->tile->data_len = vc_get_linesize(s->tile->width, s->frame->color_spec) * s->tile->height;
        s->vbuffer[0] = malloc(s->tile->data_len);
        s->vbuffer[1] = malloc(s->tile->data_len);

        s->grab_buf_idx = 0;

        SGDisposeDeviceList(s->grabber, deviceList);
        if(s->grab_audio) {
                OSErr ret;
                OSType compression;

                if (SGGetChannelDeviceList
                    (s->audio_channel, sgDeviceListIncludeInputs,
                     &deviceList) != noErr) {
                        fprintf(stderr, "Unable to get list of quicktime audio devices\n");
                        s->grab_audio = FALSE;
                        goto AFTER_AUDIO;
                }
                SGDeviceName *deviceEntry = &(*deviceList)->entry[s->audio_major];
                printf("Quicktime: Setting audio device: ");
                nprintf((char *) deviceEntry->name);
                printf("\n");
                if (SGSetChannelDevice(s->audio_channel, deviceEntry->name) !=
                    noErr) {
                        fprintf(stderr, "Setting up the selected audio device failed\n");
                        s->grab_audio = FALSE;
                        goto AFTER_AUDIO;
                }

                /* Select input */
                inputList = deviceEntry->inputs;
                printf("Quicktime: Setting audio input: ");
                nprintf((char *)(&(*inputList)->entry[s->audio_minor].name));
                printf("\n");
                if (SGSetChannelDeviceInput(s->audio_channel, s->audio_minor) !=
                    noErr) {
                        fprintf(stderr, "Setting up input on selected audiodevice failed\n");
                        s->grab_audio = FALSE;
                        goto AFTER_AUDIO;
                }
                short bps, ch_count;
                ret = SGGetSoundInputParameters(s->audio_channel, &bps,
                                &ch_count, &compression);
                s->audio.bps = bps;
                s->audio.ch_count = ch_count;

                if(ret != noErr) {
                        fprintf(stderr, "Quicktime: failed to get audio properties");
                        s->grab_audio = FALSE;
                        goto AFTER_AUDIO;
                }
                /* if we need to specify format explicitly, we would use it here
                 * but take care that sowt is 16-bit etc.! */
                /*ret = SGSetSoundInputParameters(s->audio_channel, s->audio.bps,
                                s->audio.ch_count, 'sowt');
                if(ret != noErr) {
                        fprintf(stderr, "Quicktime: failed to set audio properties");
                        s->grab_audio = FALSE;
                        goto AFTER_AUDIO;
                }*/
                s->audio.bps /= 8; /* bits -> bytes */
                Fixed tmp;
                tmp = SGGetSoundInputRate(s->audio_channel);
                /* next line solves common Fixed overflow (wtf QT?) */
                s->audio.sample_rate = Fix2X(UnsignedFixedMulDiv(tmp, X2Fix(1), X2Fix(2)))* 2.0;
                s->abuffer[0] = (char *) malloc(s->audio.sample_rate * s->audio.bps *
                                s->audio.ch_count);
                s->abuffer[1] = (char *) malloc(s->audio.sample_rate * s->audio.bps *
                                s->audio.ch_count);
                
                SGSetSoundRecordChunkSize(s->audio_channel, -65536/120); /* Negative argument meens
                                                                            that the value is Fixed
                                                                            (in secs). 1/120 sec
                                                                            seems to be quite decent
                                                                            value. */
        }
AFTER_AUDIO:

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

        SGGetChannelTimeScale(s->video_channel, &scale);
        s->frame->fps = scale / 100.0;

error:
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
void *vidcap_quicktime_init(char *fmt, unsigned int flags)
{
        struct qt_grabber_state *s;

        s = (struct qt_grabber_state *)calloc(1,sizeof(struct qt_grabber_state));
        
        s->frame = vf_alloc(1);
        s->tile = vf_get_tile(s->frame, 0);
        
        if (s != NULL) {
                s->magic = MAGIC_QT_GRABBER;
                s->grabber = 0;
                s->video_channel = 0;
                s->seqID = 0;
                s->bounds.top = 0;
                s->bounds.left = 0;
                s->sg_idle_enough = 0;
                s->frame->color_spec = 0xffffffff;

                if(flags & VIDCAP_FLAG_ENABLE_AUDIO) {
                        s->grab_audio = TRUE;
                } else {
                        s->grab_audio = FALSE;
                }


                if (qt_open_grabber(s, fmt) == 0) {
                        free(s);
                        return NULL;
                }

                pthread_mutex_init(&s->lock, NULL);
                pthread_cond_init(&s->boss_cv, NULL);
                pthread_cond_init(&s->worker_cv, NULL);
                s->boss_waiting = FALSE;
                s->worker_waiting = FALSE;
                s->work_to_do = TRUE;
                s->grab_buf_idx = 0;
                s->tile->data = s->vbuffer[0];
                s->audio.data = s->abuffer[0];
                pthread_create(&s->thread_id, NULL, vidcap_quicktime_thread, s);
        }

        return s;
}

/* Finalize the grabbing system */
void vidcap_quicktime_finish(void *state)
{
        struct qt_grabber_state *s = (struct qt_grabber_state *)state;

        assert(s != NULL);

        pthread_mutex_lock(&s->lock);
        if(s->work_to_do) {
                s->work_to_do = FALSE;
                pthread_cond_signal(&s->boss_cv);
        }
        pthread_mutex_unlock(&s->lock);
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
                vf_free(s->frame);
                free(s);
        }
}

void * vidcap_quicktime_thread(void *state) 
{
        struct qt_grabber_state *s = (struct qt_grabber_state *)state;

        while(!should_exit) {
                memset(s->abuffer[s->grab_buf_idx], 0, s->abuffer_len);
                s->abuffer_len = 0;
                /* Run the QuickTime sequence grabber idle function, which provides */
                /* processor time to out data proc running as a callback.           */

                /* The while loop done in this way is also sort of nice bussy waiting */
                /* and synchronizes capturing and sending.                            */
                s->sg_idle_enough = 0;
                while (!s->sg_idle_enough) {
                        if (SGIdle(s->grabber) != noErr) {
                                debug_msg("Error in SGIDle\n");
                        }
                }
                pthread_mutex_lock(&s->lock);
                while(!s->work_to_do) {
                        s->worker_waiting = TRUE;
                        pthread_cond_wait((&s->worker_cv), &s->lock);
                        s->worker_waiting = FALSE;
                }
                s->audio.data_len = s->abuffer_len;
                s->tile->data = s->vbuffer[s->grab_buf_idx];
                s->audio.data = s->abuffer[s->grab_buf_idx];

                s->grab_buf_idx = (s->grab_buf_idx + 1 ) % 2;
                s->work_to_do = FALSE;

                if(s->boss_waiting)
                        pthread_cond_signal(&s->boss_cv);
                pthread_mutex_unlock(&s->lock);
        }
        return NULL;
}

/* Grab a frame */
struct video_frame *vidcap_quicktime_grab(void *state, struct audio_frame **audio)
{
        struct qt_grabber_state *s = (struct qt_grabber_state *)state;

        assert(s != NULL);
        assert(s->magic == MAGIC_QT_GRABBER);

        pthread_mutex_lock(&s->lock);
        while(s->work_to_do) {
                s->boss_waiting = TRUE;
                pthread_cond_wait(&s->boss_cv, &s->lock);
                s->boss_waiting = FALSE;
        }

        if(s->grab_audio && s->audio.data_len > 0) {
                *audio = &s->audio;
        } else {
                *audio = NULL;
        }

        s->work_to_do = TRUE;
        
        if(s->worker_waiting)
                pthread_cond_signal(&s->worker_cv);
        pthread_mutex_unlock(&s->lock);

        return s->frame;
}

#endif                          /* HAVE_MACOSX */
