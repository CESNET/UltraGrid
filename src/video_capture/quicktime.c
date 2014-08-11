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
#include "video.h"
#include "audio/audio.h"
#include <math.h>

#ifdef HAVE_MACOSX
#include <Carbon/Carbon.h>
#include <QuickTime/QuickTime.h>
#include <QuickTime/QuickTimeComponents.h>

#define MAGIC_QT_GRABBER        VIDCAP_QUICKTIME_ID

static struct {
        uint32_t qt_fourcc;
        unsigned long ug_codec;
} codec_mapping[] = {
        {'2' << 24 | 'v' << 16 | 'u' << 8 | 'y', UYVY},
        {'y' << 24 | 'u' << 16 | 'v' << 8 | '2', UYVY},
        {'y' << 24 | 'u' << 16 | 'V' << 8 | '2', UYVY},
        {'j' << 24 | 'p' << 16 | 'e' << 8 | 'g', MJPG},
        {'a' << 24 | 'v' << 16 | 'c' << 8 | '1', H264},
        {0, 0xffffffff}
};

struct avbuffer {
	int width, height;
        char *video_data;
        int video_len;
        char *audio_data;
        int audio_len;
};

static void free_avbuffer(struct avbuffer *buf)
{
        if(!buf)
                return;
        free(buf->video_data);
        free(buf->audio_data);
        free(buf);
}

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
        struct avbuffer *buffer_captured;
        struct avbuffer *buffer_ready_to_send;
        struct avbuffer *buffer_network;
        unsigned gui:1;
        int frames;
        struct timeval t0;
        const quicktime_mode_t *qt_mode;
        unsigned int grab_audio:1;

        pthread_t thread_id;
        pthread_mutex_t lock;
        pthread_cond_t boss_cv;

	volatile bool should_exit;

        bool translate_yuv2;
};

void * vidcap_quicktime_thread(void *state);
void InitCursor(void);
void GetPort(GrafPtr *port);
void SetPort(GrafPtr port);
static void usage(SeqGrabComponent grabber, SGChannel video_channel, SGChannel audio_channel);
static unsigned long get_color_spec(uint32_t pixfmt);

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

        if(!s->buffer_captured) {
                s->buffer_captured = (struct avbuffer *) calloc(1, sizeof(struct avbuffer));
        }

        if(c == s->video_channel) {
		Rect rect;
		SGGetBufferInfo(s->video_channel, chRefCon, NULL, &rect, &s->gworld, NULL);
                s->buffer_captured->width = rect.right;
                s->buffer_captured->height = rect.bottom;
                switch(writeType) {
                        case seqGrabWriteReserve:
                                break;
                        case seqGrabWriteFill:
                        case seqGrabWriteAppend:
                                s->buffer_captured->video_data = malloc(len);
                                memcpy(s->buffer_captured->video_data, p, len);
                                s->buffer_captured->video_len = len;
                                s->sg_idle_enough = 1;
                                break;
                }

                s->frames++;
                gettimeofday(&t, NULL);
                double seconds = tv_diff(t, s->t0);
                if (seconds >= 5) {
                        float fps = s->frames / seconds;
                        fprintf(stderr, "[QuickTime cap.] %d frames in %g seconds = %g FPS\n", s->frames,
                                seconds, fps);
                        s->t0 = t;
                        s->frames = 0;
                }
        } else if(c == s->audio_channel) {
                switch(writeType) {
                        case seqGrabWriteReserve:
                                break;
                        case seqGrabWriteFill:
                                s->buffer_captured->audio_data = realloc(s->buffer_captured->audio_data,
                                                s->buffer_captured->audio_len + len);
                                memcpy(s->buffer_captured->audio_data + s->buffer_captured->audio_len, p, len);
                                s->buffer_captured->audio_len += len;
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
            (Component *)(void *) NewPtr(sizeof(Component) * (numberOfPanels + 1));

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

static void usage(SeqGrabComponent grabber, SGChannel video_channel, SGChannel audio_channel)
{
        SGDeviceInputList inputList;
        SGDeviceList deviceList;
        int i, j;

#ifndef QT_ENABLE_AUDIO
        UNUSED(audio_channel);
#endif // QT_ENABLE_AUDIO

        printf("\nUsage:\t-t quicktime[:device=<device>[:mode=<mode>]][:codec=<pixel_type>][:width=<w>:height=<h>]"
#ifdef QT_ENABLE_AUDIO
                        "[:adevice=<audio_device>:amode=<audio_mode>]"
#endif // QT_ENABLE_AUDIO
                        "\n\n");
        if (SGGetChannelDeviceList
                        (video_channel, sgDeviceListIncludeInputs,
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
                                        (video_channel, NULL, NULL,
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
                SGDisposeDeviceList(grabber, deviceList);
                CodecNameSpecListPtr list;
                GetCodecNameList(&list, 1);
                printf("\nCompression types (only those marked with a '*' are supported by UG):\n");
                for (i = 0; i < list->count; i++) {
                        int fcc = list->list[i].cType;
                        printf("\t");
                        if(get_color_spec(list->list[i].cType) != 0xffffffff) {
                                putchar('*');
                        } else {
                                putchar(' ');
                        }
                        printf("%d) ", i);
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
#ifdef QT_ENABLE_AUDIO
        if (SGGetChannelDeviceList
                        (audio_channel, sgDeviceListIncludeInputs,
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
                                        (video_channel, NULL, NULL,
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
                SGDisposeDeviceList(grabber, deviceList);
        }
#endif // QT_ENABLE_AUDIO
}

static bool select_source(SeqGrabComponent grabber, SGChannel video_channel,
                int *major, int *minor)
{
        SGDeviceList deviceList;

        if (*major != -1 && *minor != -1) {
                return true;
        }

        if (*major == -1 && *minor != -1) {
                return false;
        }

        if (SGGetChannelDeviceList
                        (video_channel, sgDeviceListIncludeInputs,
                         &deviceList) == noErr) {
                if (*major == -1)
                        *major = (*deviceList)->selectedIndex;
                SGDisposeDeviceList(grabber, deviceList);
        } else {
                return false;
        }

        short activeInputIndex = 0;
        SGGetChannelDeviceAndInputNames
                (video_channel, NULL, NULL,
                 &activeInputIndex);
        if (*minor == -1) {
                *minor = activeInputIndex;
        }

        return true;
}

static unsigned long get_color_spec(uint32_t pixfmt) {
        int i;
        // first, try to find explicit mapping to UG color_spec...
        for (i = 0; codec_mapping[i].ug_codec != 0xffffffff; i++) {
                if(pixfmt == codec_mapping[i].qt_fourcc) {
                        return codec_mapping[i].ug_codec;
                        break;
                }
        }

        codec_t codec = get_codec_from_name(pixfmt);
        if (codec != VIDEO_CODEC_NONE) {
                return codec;
        } else {
                // try also the other endianity (QT codecs aren't
                // entirely consistent in this regard)
                return get_codec_from_name(ntohl(pixfmt));
        }
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

#ifdef QT_ENABLE_AUDIO
        /* do not check for grab audio in case that we will only display usage */
        if (SGNewChannel(s->grabber, SoundMediaType, &s->audio_channel) !=
                        noErr) {
                fprintf(stderr, "Warning: Creating audio channel failed. "
                                "Disabling sound output.\n");
                s->grab_audio = FALSE;
        }
#endif // QT_ENABLE_AUDIO

        /* Print available devices */
        int i;
        SGDeviceInputList inputList;
        SGDeviceList deviceList;
        if (strcmp(fmt, "help") == 0) {
                usage(s->grabber, s->video_channel, s->audio_channel);
                return 2;
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

        int qt_codec_index = -1;
        int width = -1,
            height = -1;
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

                char *tmp, *item, *save_ptr;
                tmp = fmt;

                s->major = s->minor = -1;
                s->audio_major = s->audio_minor = -1;

                while ((item = strtok_r(tmp, ":", &save_ptr)) != NULL) {
                        if (strncasecmp(item, "device=", strlen("device=")) == 0) {
                                s->major = atoi(item + strlen("device="));
                        } else if (strncasecmp(item, "mode=", strlen("mode=")) == 0) {
                                s->minor = atoi(item + strlen("mode="));
                        } else if (strncasecmp(item, "codec=", strlen("codec=")) == 0) {
                                qt_codec_index = atoi(item + strlen("codec="));
                        } else if (strncasecmp(item, "adevice=", strlen("adevice=")) == 0) {
                                s->audio_major = atoi(item + strlen("adevice="));
                        } else if (strncasecmp(item, "amode=", strlen("amode=")) == 0) {
                                s->audio_minor = atoi(item + strlen("amode="));
                        } else if (strncasecmp(item, "width=", strlen("width=")) == 0) {
                                width = atoi(item + strlen("width="));
                        } else if (strncasecmp(item, "height=", strlen("height=")) == 0) {
                                height = atoi(item + strlen("height="));
                        }

                        tmp = NULL;
                }

                if (!select_source(s->grabber, s->video_channel, &s->major, &s->minor)) {
                        fprintf(stderr, "[Quicktime disp.] Unable to select device/mode. Please select it "
                                        "manually!\n");
                        return 0;
                }

                if ((width != -1 && height == -1) || (width == -1 && height != -1)) {
                        fprintf(stderr, "[Quicktime disp.] If setting input size, you must set both dimensions.\n");
                        return 0;
                }

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
                                "\tPlease report it to " PACKAGE_BUGREPORT "\n\n");
        }

        printf("Quicktime: Video size: %dx%d\n", s->bounds.right,
               s->bounds.bottom);

        if (width != -1 && height != -1) {
                s->bounds.right = width;
                s->bounds.bottom = height;
        }

        if (SGSetChannelBounds(s->video_channel, &(s->bounds)) != noErr) {
                debug_msg("Unable to set channel bounds\n");
                return 0;
        }

        /* Set selected fmt->codec and get pixel format of that codec */
        int pixfmt = -1;
        if (qt_codec_index != -1) {
                CodecNameSpecListPtr list;
                GetCodecNameList(&list, 1);
                pixfmt = list->list[qt_codec_index].cType;
                printf("Quicktime: SetCompression: %s\n",
                       (int)SGSetVideoCompressor(s->video_channel, 0,
                                                 list->list[qt_codec_index].codec, 0,
                                                 0, 0)==0?"OK":"Failed!");
        } else {
#if 0
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
#endif
                unsigned long default_fcc = '2vuy';
                CodecNameSpecListPtr list;
                GetCodecNameList(&list, 1);
                for (i = 0; i < list->count; i++) {
                        if (default_fcc == list->list[i].cType) {
                                pixfmt = default_fcc;
                                qt_codec_index = i;
                                break;
                        }
                }
                printf("Quicktime: SetCompression: %s\n",
                       (int)SGSetVideoCompressor(s->video_channel, 0,
                                                 list->list[qt_codec_index].codec, 0,
                                                 0, 0)==0?"OK":"Failed!");
        }
        if(pixfmt == -1) {
                fprintf(stderr, "[QuickTime] Unknown pixelformat!\n");
                goto error;
        }

        if (get_color_spec(pixfmt) == VIDEO_CODEC_NONE) {
                fprintf(stderr, "[QuickTime] Cannot find UltraGrid codec matching: %c%c%c%c!\n",
				pixfmt >> 24, (pixfmt >> 16) & 0xff, (pixfmt >> 8) & 0xff, (pixfmt) & 0xff);
                goto error;
        }

        // thiw is a workaround - codec identified by exactly this FourCC seems to have
        // different semantics than ordinary UYVY
        if (pixfmt == ('2' | 'v' << 8 | 'u' << 16 | 'y' << 24)) {
                s->translate_yuv2 = true;
        }

        s->frame->color_spec = get_color_spec(pixfmt);

        printf("Quicktime: Selected pixel format: %c%c%c%c\n",
               pixfmt >> 24, (pixfmt >> 16) & 0xff, (pixfmt >> 8) & 0xff,
               (pixfmt) & 0xff);

        s->tile->data_len = vc_get_linesize(s->tile->width, s->frame->color_spec) * s->tile->height;

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
                s->audio.ch_count = audio_capture_channels;

                if(ret != noErr) {
                        fprintf(stderr, "Quicktime: failed to get audio properties");
                        s->grab_audio = FALSE;
                        goto AFTER_AUDIO;
                }
                /* if we need to specify format explicitly, we would use it here
                 * but take care that sowt is 16-bit etc.! */
                ret = SGSetSoundInputParameters(s->audio_channel, s->audio.bps,
                                s->audio.ch_count, 'sowt');
                if(ret != noErr) {
                        fprintf(stderr, "Quicktime: failed to set audio properties");
                        s->grab_audio = FALSE;
                        goto AFTER_AUDIO;
                }
                s->audio.bps /= 8; /* bits -> bytes */
                Fixed tmp;
                tmp = SGGetSoundInputRate(s->audio_channel);
                /* next line solves common Fixed overflow (wtf QT?) */
                s->audio.sample_rate = Fix2X(UnsignedFixedMulDiv(tmp, X2Fix(1), X2Fix(2)))* 2.0;
                
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

        return 1;
error:
        return 0;
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
void *vidcap_quicktime_init(const struct vidcap_params *params)
{
        struct qt_grabber_state *s;

        s = (struct qt_grabber_state *)calloc(1,sizeof(struct qt_grabber_state));
        
        s->frame = vf_alloc(1);
        s->tile = vf_get_tile(s->frame, 0);

        fprintf(stderr, "\033[0;31m[QuickTime cap.] \033[1;31mWarning:\033[0;31m This module is deprecated and will be removed in future. Please use AV Foundation module instead (-t avfoundation).\n\033[0;49m");
        
        if (s != NULL) {
                s->magic = MAGIC_QT_GRABBER;
                s->grabber = 0;
                s->video_channel = 0;
                s->seqID = 0;
                s->bounds.top = 0;
                s->bounds.left = 0;
                s->sg_idle_enough = 0;
                s->frame->color_spec = 0xffffffff;

                s->grab_audio = FALSE;
                if(vidcap_params_get_flags(params) & VIDCAP_FLAG_AUDIO_EMBEDDED) {
#ifdef QT_ENABLE_AUDIO
                        s->grab_audio = TRUE;
#else
                        fprintf(stderr, "QuickTime audio support is unmaintained and deprecated.\n"
                                        "Please use \"coreaudio\" audio driver instead.\n");
                        return NULL;
#endif
                }

                char *fmt = strdup(vidcap_params_get_fmt(params));
                int ret = qt_open_grabber(s, fmt);
                free(fmt);

                if (ret != 1) {
                        free(s);
                        if(ret == 2) {
                                return &vidcap_init_noerr;
                        } else {
                                return NULL;
                        }
                }

                gettimeofday(&s->t0, NULL);

                pthread_mutex_init(&s->lock, NULL);
                pthread_cond_init(&s->boss_cv, NULL);

		s->buffer_captured = s->buffer_ready_to_send =
			s->buffer_network = 0;

                pthread_create(&s->thread_id, NULL, vidcap_quicktime_thread, s);
        }

        return s;
}

/* Finalize the grabbing system */
void vidcap_quicktime_done(void *state)
{
        struct qt_grabber_state *s = (struct qt_grabber_state *)state;

        assert(s != NULL);

        pthread_mutex_lock(&s->lock);
        s->should_exit = true;
        pthread_mutex_unlock(&s->lock);
        pthread_join(s->thread_id, NULL);

        if (s != NULL) {
                assert(s->magic == MAGIC_QT_GRABBER);
                SGStop(s->grabber);
                CloseComponent(s->grabber);
                ExitMovies();

                pthread_mutex_destroy(&s->lock);
                pthread_cond_destroy(&s->boss_cv);

                vf_free(s->frame);
                free(s);
        }
}

void * vidcap_quicktime_thread(void *state) 
{
        struct qt_grabber_state *s = (struct qt_grabber_state *)state;

        while(!s->should_exit) {
                /* Run the QuickTime sequence grabber idle function, which provides */
                /* processor time to out data proc running as a callback.           */

                /* The while loop done in this way is also sort of nice bussy waiting */
                /* and synchronizes capturing and sending.                            */
                s->sg_idle_enough = 0;
                while (!s->sg_idle_enough && !s->should_exit) {
                        if (SGIdle(s->grabber) != noErr) {
                                debug_msg("Error in SGIDle\n");
                        }
                }
		if(s->should_exit) {
			break;
		}
		pthread_mutex_lock(&s->lock);
                if(s->buffer_ready_to_send) {
                        free_avbuffer(s->buffer_ready_to_send);
                }
                s->buffer_ready_to_send = s->buffer_captured;
                s->buffer_captured = NULL;

                pthread_cond_signal(&s->boss_cv);
                pthread_mutex_unlock(&s->lock);
        }
        pthread_mutex_lock(&s->lock);
        pthread_cond_signal(&s->boss_cv);
        pthread_mutex_unlock(&s->lock);

        return NULL;
}

/* Grab a frame */
struct video_frame *vidcap_quicktime_grab(void *state, struct audio_frame **audio)
{
        struct qt_grabber_state *s = (struct qt_grabber_state *)state;

        assert(s != NULL);
        assert(s->magic == MAGIC_QT_GRABBER);

        free_avbuffer(s->buffer_network);
        s->buffer_network = NULL;

        pthread_mutex_lock(&s->lock);
	int ret = 0;
        while(!s->buffer_ready_to_send && ret == 0) {
		struct timeval tv;
		struct timespec ts;
		gettimeofday(&tv, NULL);
		ts.tv_sec = tv.tv_sec + 1;
		ts.tv_nsec = tv.tv_usec * 1000;
                ret = pthread_cond_timedwait(&s->boss_cv, &s->lock, &ts);
        }
	if(ret == ETIMEDOUT) {
		pthread_mutex_unlock(&s->lock);
		return NULL;
	}
        s->buffer_network = s->buffer_ready_to_send;
        s->buffer_ready_to_send = 0;
        pthread_mutex_unlock(&s->lock);

        s->audio.data_len = s->buffer_network->audio_len;
        s->tile->data_len = s->buffer_network->video_len;
        if(s->buffer_network->width)
                s->tile->width = s->buffer_network->width;
        if(s->buffer_network->height)
                s->tile->height = s->buffer_network->height;
        s->tile->data = s->buffer_network->video_data;
        s->audio.data = s->buffer_network->audio_data;

        if(s->grab_audio && s->audio.data_len > 0) {
                *audio = &s->audio;
        } else {
                *audio = NULL;
        }

        // Mac 10.7 seems to change semantics for codec identified with 'yuv2' FourCC,
        // which is mapped to our UYVY. This simply makes it correct.
        if (s->translate_yuv2) {
                for (unsigned int i = 0; i < s->frame->tiles->data_len; i += 4) {
                        int a = s->frame->tiles[0].data[i];
                        int b = s->frame->tiles[0].data[i + 1];
                        int c = s->frame->tiles[0].data[i + 2];
                        int d = s->frame->tiles[0].data[i + 3];
                        s->frame->tiles[0].data[i] = b + 128;
                        s->frame->tiles[0].data[i + 1] = a;
                        s->frame->tiles[0].data[i + 2] = d + 128;
                        s->frame->tiles[0].data[i + 3] = c;
                }
        }
        
        return s->frame;
}

#endif                          /* HAVE_MACOSX */
