/**
 * @file   video_display/bluefish444.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2013-2015 CESNET, z. s. p. o.
 * All rights reserved.
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
 * 3. Neither the name of CESNET nor the names of its contributors may be
 *    used to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHORS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING,
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
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif

#include "video_display.h"

#include <iomanip>
#include <iostream>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "bluefish444_common.h"

#include "audio/audio.h"
#include "debug.h"
#include "host.h"
#include "lib_common.h"
#include "tv.h"
#include "utils/ring_buffer.h"
#include "video.h"
#include "video_display.h"

#define BLUEFISH444_MAGIC 0x15b75db8

#define BUFFER_COUNT 4

#define MAX_BLUE_OUT_CHANNELS 4

using namespace std;

static void InitOutputChannel(CBLUEVELVET_H pSDK, ULONG DefaultOutputChannel,
                ULONG UpdateFormat, ULONG MemoryFormat, ULONG VideoEngine);
static void RouteChannel(CBLUEVELVET_H pSDK, ULONG Source, ULONG Destination,
                ULONG LinkType);

struct av_buffer
{
        av_buffer(int TilesCount, int GoldenSize) :
                pVideoBuffer(TilesCount), BufferId(0)
        {
                for(int i = 0; i < TilesCount; ++i) {
                        pVideoBuffer[i] = (unsigned int *) page_aligned_alloc(GoldenSize);
                }
        }
        virtual ~av_buffer() {
                for(unsigned int i = 0; i < pVideoBuffer.size(); ++i) {
                        page_aligned_free(pVideoBuffer[i]);
                }
        }
        vector<unsigned int *> pVideoBuffer;
        unsigned int  BufferId;
};

struct display_bluefish444_state {
        public:
                                    display_bluefish444_state(unsigned int flags, int deviceId = 1)
                                                                   throw(runtime_error);
                virtual            ~display_bluefish444_state()    throw();
                struct video_frame *getf()                         throw();
                void                putf(struct video_frame *)     throw(runtime_error, logic_error);
                void                reconfigure(struct video_desc) throw(runtime_error, logic_error);
                void                cleanup()                      throw();
                static void        *playback_thread(void *arg)     throw();
                void               *playback_loop()                throw();

#ifdef HAVE_BLUE_AUDIO
                /// AUDIO
                void                reconfigure_audio(int quant_samples, int channels,
                                int sample_rate)                   throw(runtime_error, logic_error);
                void                put_audio_frame(struct audio_frame *) throw();
#endif
        private:
                uint32_t            m_magic;

                struct video_frame *m_frame;

                int                 m_deviceId;
                CBLUEVELVET_H       m_pSDK[MAX_BLUE_OUT_CHANNELS];
                int                 m_AttachedDevicesCount;
                int                 m_CardType;
                int                 m_CardFirmware;
                int                 m_TileCount;
                int                 m_Offset[MAX_BLUE_OUT_CHANNELS];

                ULONG               m_InvalidVideoModeFlag;
                ULONG               m_CurrentVideoMode;
                ULONG               m_GoldenSize;
                ULONG               m_LastFieldCount;

                pthread_mutex_t     m_lock;
                pthread_cond_t      m_WorkerCv;
                pthread_cond_t      m_BossCv;
                pthread_t           m_thread;
                queue<struct av_buffer *> m_ReadyFrameQueue;
                bool                m_WorkerWaiting;
                queue<struct av_buffer *> m_FreeFrameQueue;
                struct av_buffer   *m_pPlayingBuffer;

                // this is temporary storage between getf/putf calls
                av_buffer          *m_pTmpFrame;
#ifdef HAVE_BLUE_AUDIO
                struct audio_desc   m_AudioDesc;
                hanc_stream_info_struct m_HancInfo;
                pthread_spinlock_t  m_AudioSpinLock;
                struct ring_buffer *m_AudioRingBuffer;
#endif
                bool                m_PlayAudio;
 };

display_bluefish444_state::display_bluefish444_state(unsigned int flags,
                int deviceId) throw(runtime_error) :
        m_magic(BLUEFISH444_MAGIC),
        m_frame(NULL),
        m_deviceId(deviceId),
        m_AttachedDevicesCount(0),
        m_GoldenSize(0),
        m_LastFieldCount(0),
        m_pPlayingBuffer(NULL),
#ifdef HAVE_BLUE_AUDIO
        m_AudioRingBuffer(ring_buffer_init(48000*4*16*8)),
#endif
        m_PlayAudio(false)
{
        int iDevices = 0;
        uint32_t val32;

        if(flags) {
#ifdef HAVE_BLUE_AUDIO
                m_PlayAudio = true;
#else
                throw runtime_error("Audio support missing in this build");
#endif
        }
        pthread_mutex_init(&m_lock, NULL);
        pthread_cond_init(&m_WorkerCv, NULL);
        pthread_cond_init(&m_BossCv, NULL);

        CBLUEVELVET_H pSDK = bfcFactory();
        bfcEnumerate(pSDK, iDevices);
        if(iDevices < 1) {
                bfcDestroy(pSDK);
                throw runtime_error("No Bluefish card detected");
        }

        m_CardType = bfcQueryCardType(pSDK);
        bfcQueryCardProperty32(pSDK, INVALID_VIDEO_MODE_FLAG, val32);
        m_InvalidVideoModeFlag = val32;
                
        if(BLUE_FAIL(bfcAttach(pSDK, m_deviceId))) {
                bfcDestroy(pSDK);
                throw runtime_error("Unable to attach card");
        }

        bfcQueryCardProperty32(pSDK, EPOCH_GET_PRODUCT_ID, val32);

        m_CardFirmware = val32;
        m_deviceId = deviceId;

        bfcDetach(pSDK);
        bfcDestroy(pSDK);

#ifdef HAVE_BLUE_AUDIO
        memset(&m_AudioDesc, 0, sizeof(m_AudioDesc));
        memset(&m_HancInfo, 0, sizeof(m_HancInfo));
        for(int i = 0; i < 4; ++i) {
                m_HancInfo.AudioDBNArray[i] = -1;
        }
        m_HancInfo.hanc_data_ptr = (unsigned int *) page_aligned_alloc(MAX_HANC_SIZE);
        pthread_spin_init(&m_AudioSpinLock, 0);
#endif

        pthread_create(&m_thread, NULL, playback_thread, this);
}

display_bluefish444_state::~display_bluefish444_state() throw()
{
        assert(m_magic == BLUEFISH444_MAGIC);
        // Kill thread
        pthread_mutex_lock(&m_lock);
        m_ReadyFrameQueue.push(NULL);
        pthread_cond_signal(&m_WorkerCv);
        pthread_mutex_unlock(&m_lock);
        ULONG FieldCount;
        //cards must be genlocked; only then all for output channels are completely in synch
        bfcWaitVideoOutputSync(m_pSDK[0], UPD_FMT_FRAME, FieldCount);

        pthread_join(m_thread, NULL);

        cleanup();

        pthread_mutex_destroy(&m_lock);
        pthread_cond_destroy(&m_WorkerCv);
        pthread_cond_destroy(&m_BossCv);

#ifdef HAVE_BLUE_AUDIO
        ring_buffer_destroy(m_AudioRingBuffer);
        page_aligned_free(m_HancInfo.hanc_data_ptr);
        pthread_spin_destroy(&m_AudioSpinLock);
#endif
}

void *display_bluefish444_state::playback_thread(void *arg) throw()
{
        display_bluefish444_state *s = 
                (display_bluefish444_state *) arg;
        return s->playback_loop();
}

void *display_bluefish444_state::playback_loop() throw()
{
        uint32_t FrameCount = 0;
        struct timeval t0;
#ifdef HAVE_BLUE_AUDIO
        uint32_t EmbAudioFlag = blue_enable_hanc_timestamp_pkt | blue_emb_audio_enable; // enable timecode
#endif

        gettimeofday(&t0, NULL);

#ifdef WIN32
        OVERLAPPED Overlapped[MAX_BLUE_OUT_CHANNELS];
        for (int i = 0;i < MAX_BLUE_OUT_CHANNELS; ++ i) {
                Overlapped[i].hEvent = CreateEvent(NULL, TRUE, FALSE, NULL);
        }
#endif

        while(1) {
                struct av_buffer *frame;
                pthread_mutex_lock(&m_lock);
                while(m_ReadyFrameQueue.empty()) {
                        m_WorkerWaiting = true;
                        pthread_cond_wait(&m_WorkerCv, &m_lock);
                        m_WorkerWaiting = false;
                }

                frame = m_ReadyFrameQueue.front();
                m_ReadyFrameQueue.pop();
                pthread_mutex_unlock(&m_lock);

                if(frame == NULL) {
                        return NULL;
                }

                ULONG FieldCount = 0;

                // for more than one input, cards must be genlocked
                bfcWaitVideoOutputSync(m_pSDK[0], UPD_FMT_FRAME, FieldCount);
                if(m_pPlayingBuffer) {
                        pthread_mutex_lock(&m_lock);
                        m_FreeFrameQueue.push(m_pPlayingBuffer);
                        pthread_cond_signal(&m_BossCv);
                        pthread_mutex_unlock(&m_lock);
                }

                bool WaitResult = true;

                for(int i = 0; i < m_AttachedDevicesCount; ++i) {
                        unsigned char *videoBuffer;
#ifdef WIN32
                        OVERLAPPED *OverlapCh = &Overlapped[i];
#else
                        OVERLAPPED *OverlapCh = NULL;
#endif
                        if(m_TileCount == m_AttachedDevicesCount) {
                                videoBuffer = (unsigned char *) frame->pVideoBuffer[i];
                        } else { // untiled 4K
                                videoBuffer = (unsigned char *) frame->pVideoBuffer[0];
                        }
                        int err = bfcSystemBufferWriteAsync(m_pSDK[i], videoBuffer + m_Offset[i],
                                        m_GoldenSize, OverlapCh,
                                        (m_PlayAudio && i == 0 ?
                                         BlueImage_HANC_DMABuffer(frame->BufferId, BLUE_DATA_IMAGE) :
                                         BlueImage_DMABuffer(frame->BufferId, BLUE_DATA_IMAGE))
                                        );
                        if(!BLUE_OK(err)) {
                                cerr << "Write failed (Channel " << (char) ('A' + i) << ")" << endl;
                                WaitResult = false;
                        }
                }
#ifdef WIN32
                if(WaitResult) {
                        for(int i = 0; i < m_AttachedDevicesCount; ++i) {
                                DWORD BytesReturned = 0;
                                GetOverlappedResult(bfcGetHandle(m_pSDK[i]), &Overlapped[i], &BytesReturned, TRUE);
                                ResetEvent(Overlapped[i].hEvent);
                        }
                }
#else
                UNUSED(WaitResult);
#endif

#ifdef HAVE_BLUE_AUDIO
                if(m_PlayAudio) {
                        m_HancInfo.time_code = 0LL;                       //RP188 VITC time code
                        m_HancInfo.rp188_ltc_time_code = 0LL;     //RP188 LTC time code
                        m_HancInfo.ltc_time_code = 0LL;           //external LTC time code
                        m_HancInfo.sd_vitc_time_code = 0LL;       //this field is only valid for SD video signal
                        m_HancInfo.video_mode = m_CurrentVideoMode;

                        uint32_t NrSamples = 0;
                        pthread_spin_lock(&m_AudioSpinLock);
                        uint32_t NumberAudioChannels = m_AudioDesc.ch_count;
                        char audio_data[2002*4*16];
                        unsigned int nSampleType;
                        ULONG EmbAudioProp = 0;

                        if(NumberAudioChannels > 0) {
                                NrSamples = GetNumberOfAudioSamplesPerFrame(m_CurrentVideoMode,
                                                FrameCount);
                                switch(m_AudioDesc.bps) {
                                        case 2:
                                                nSampleType = AUDIO_CHANNEL_16BIT; // AUDIO_CHANNEL_LITTLEENDIAN
                                                break;
                                        case 3:
                                                nSampleType = AUDIO_CHANNEL_24BIT;
                                                break;
                                        default:
                                                fprintf(stderr, "[Blue] Unhandled audio sample type!\n");
                                                abort();
                                }

                                if(NumberAudioChannels > 0)
                                        EmbAudioProp |= (blue_emb_audio_group1_enable);
                                if(NumberAudioChannels > 4)
                                        EmbAudioProp |= (blue_emb_audio_group2_enable);
                                if(NumberAudioChannels > 8)
                                        EmbAudioProp |= (blue_emb_audio_group3_enable);
                                if(NumberAudioChannels > 12)
                                        EmbAudioProp |= (blue_emb_audio_group4_enable);

                                int bytes = ring_buffer_read(m_AudioRingBuffer, audio_data, NrSamples *
                                                m_AudioDesc.bps *m_AudioDesc.ch_count);
                                NrSamples = bytes / m_AudioDesc.bps / m_AudioDesc.ch_count;
                        }
                        pthread_spin_unlock(&m_AudioSpinLock);

                        if(NrSamples > 0) {
                                bfcEncodeHancFrameEx(m_pSDK[0], m_CardType, &m_HancInfo, audio_data,
                                                m_AudioDesc.ch_count, NrSamples,
                                                nSampleType, EmbAudioProp | EmbAudioFlag);

                                //wait for both DMA transfers to be finished
                                //GetOverlappedResult(pSDK->m_hDevice, &OverlapChA, &BytesReturnedChA, TRUE);
                                //ResetEvent(OverlapChA.hEvent);

                                //now we can DMA the HANC frame
                                bfcSystemBufferWriteAsync(m_pSDK[0], (unsigned char *) m_HancInfo.hanc_data_ptr,
                                                MAX_HANC_SIZE,
                                                NULL, BlueImage_HANC_DMABuffer(frame->BufferId, BLUE_DATA_HANC));
                        }
                }
#endif
                //tell the card to playback the frames on the next interrupt
                for(int i = 0; i < m_AttachedDevicesCount; ++i) {
                        bfcRenderBufferUpdate(m_pSDK[i],
                                        (m_PlayAudio && i == 0 ?
                                         BlueBuffer_Image_HANC(frame->BufferId) :
                                         BlueBuffer_Image(frame->BufferId))
                                        );
                }

                if(FieldCount != m_LastFieldCount + 2) {
                        cerr << "[Blue444 disp] " << "Dropped " <<
                                ((FieldCount - m_LastFieldCount + 2)/2) <<
                                " frames." << endl;
                }

                m_LastFieldCount = FieldCount;

                FrameCount++;

                struct timeval t;
                gettimeofday(&t, NULL);
                double seconds = tv_diff(t, t0);
                if (seconds >= 5) {
                        float fps  = FrameCount / seconds;
                        log_msg(LOG_LEVEL_INFO, "[Blue disp] %d frames in %g seconds = %g FPS\n",
                                        FrameCount, seconds, fps);
                        t0 = t;
                        FrameCount = 0;
                }

                m_pPlayingBuffer = frame;
        }

#ifdef WIN32
                for(int i = 0; i < m_AttachedDevicesCount; ++i) {
                        CloseHandle(Overlapped[i].hEvent);
                }
#endif

        return NULL;
}

void display_bluefish444_state::cleanup() throw()
{
        //turn on black generator (unless we want to keep displaying the last rendered frame)
        for(int i = 0; i < m_AttachedDevicesCount; ++i) {
                uint32_t value;

                value = ENUM_BLACKGENERATOR_ON;
                bfcSetCardProperty32(m_pSDK[i], VIDEO_BLACKGENERATOR, value);

                bfcVideoPlaybackStop(m_pSDK[i], 1, 1);
        }

        while(!m_FreeFrameQueue.empty()) {
                struct av_buffer *frame = m_FreeFrameQueue.front();
                m_FreeFrameQueue.pop();
                delete frame;
        }

        if(m_pPlayingBuffer) {
                delete m_pPlayingBuffer;
                m_pPlayingBuffer = NULL;
        }

        vf_free(m_frame);
        m_frame = NULL;

        for(int i = 0; i < m_AttachedDevicesCount; ++i) {
                bfcDetach(m_pSDK[i]);
                bfcDestroy(m_pSDK[i]);
        }

        m_AttachedDevicesCount = 0;
}

void display_bluefish444_state::reconfigure(struct video_desc desc) 
        throw(runtime_error, logic_error)
{
        ULONG VideoMode;
        uint32_t val32;
        bool is4K;
        bool isTiled4K;
        int tile_width, tile_height;

        pthread_mutex_lock(&m_lock);
        while(!m_ReadyFrameQueue.empty() ||
                !m_WorkerWaiting)
                pthread_cond_wait(&m_BossCv, &m_lock);
        pthread_mutex_unlock(&m_lock);

        cleanup();

        tile_width = desc.width;
        tile_height = desc.height;

        // 4K
        if((desc.width == 4096 || desc.width == 3840) && desc.height == 2160) {
                is4K = true;
                isTiled4K = false;
                tile_width /= 2;
                tile_height /= 2;
        } else if(desc.tile_count == 4 &&
                        (desc.width == 2048 || desc.width == 1920) &&
                        desc.height == 1080) {
                // tiled 4K
                is4K = true;
                isTiled4K = true;
        } else {
                is4K = false;
                isTiled4K = false;
        }

        m_TileCount = desc.tile_count;

        if(is4K) {
                // we need a SuperNova QuadOut card
                if(m_CardType != CRD_BLUE_SUPER_NOVA) {
                        throw runtime_error("We need a SuperNova card for 4K mode! "
                                        "Let us know if you need a different combination "
                                        "(eg. 2x Bluefish card");
                }

                if(m_CardFirmware != ORAC_4SDIOUTPUT_FIRMWARE_PRODUCTID)
                {
                        throw runtime_error("Wrong firmware; need 4 output channels (QuadOut)");
                }
                
                m_AttachedDevicesCount = 4;
        } else {
                m_AttachedDevicesCount = 1;
        }

        for(int i = 0; i < m_AttachedDevicesCount; ++i) {
                m_pSDK[i] = bfcFactory();

                //Attach the SDK object to a specific card, in this case card 1
                if(BLUE_FAIL(bfcAttach(m_pSDK[i], m_deviceId)))
                {
                        throw runtime_error("Error on device attach");
                }
        }

        VideoMode = m_InvalidVideoModeFlag;

        for(int i = 0; i < bluefish_frame_modes_count; ++i) {
                if((int) bluefish_frame_modes[i].width == tile_width &&
                                (int) bluefish_frame_modes[i].height == tile_height &&
                                fabs(bluefish_frame_modes[i].fps - desc.fps) < 0.01 &&
                                bluefish_frame_modes[i].interlacing == desc.interlacing)
                {
                        VideoMode = 
                                bluefish_frame_modes[i].mode;
                }
        }

        if(VideoMode == m_InvalidVideoModeFlag) {
                ostringstream msg_ss;
                msg_ss << "Unable to find appropriate video format for " <<
                        tile_width << "x" << tile_height <<
                        get_interlacing_suffix(desc.interlacing) <<
                        " @" << fixed << setprecision(2) << desc.fps;
                throw runtime_error(msg_ss.str());
        }

        m_CurrentVideoMode = VideoMode;

        ULONG UpdateFormat = UPD_FMT_FRAME;
        ULONG MemoryFormat;
        switch(desc.color_spec) {
                case UYVY:
                        MemoryFormat = MEM_FMT_2VUY;
                        break;
                case v210:
                        MemoryFormat = MEM_FMT_V210;
                        break;
                case RGBA:
                        MemoryFormat = MEM_FMT_RGBA;
                        break;
                case DPX10:
                        MemoryFormat = MEM_FMT_CINEON_LITTLE_ENDIAN;
                        break;
                default:
                        throw logic_error("Unknown video format!!!");
        }

        ULONG VideoEngine = VIDEO_ENGINE_FRAMESTORE;

        ULONG OutputChannels[4] = {
                BLUE_VIDEO_OUTPUT_CHANNEL_A,
                BLUE_VIDEO_OUTPUT_CHANNEL_B,
                BLUE_VIDEO_OUTPUT_CHANNEL_C,
                BLUE_VIDEO_OUTPUT_CHANNEL_D
        };
        ULONG Sources[4] = { EPOCH_SRC_OUTPUT_MEM_INTERFACE_CHA,
                EPOCH_SRC_OUTPUT_MEM_INTERFACE_CHB,
                EPOCH_SRC_OUTPUT_MEM_INTERFACE_CHC,
                EPOCH_SRC_OUTPUT_MEM_INTERFACE_CHD
        };
        ULONG Destinations[4] = { EPOCH_DEST_SDI_OUTPUT_A,
                EPOCH_DEST_SDI_OUTPUT_B,
                EPOCH_DEST_SDI_OUTPUT_C,
                EPOCH_DEST_SDI_OUTPUT_D
        };

        for(int i = 0; i < m_AttachedDevicesCount; ++i) {
                InitOutputChannel(m_pSDK[i], OutputChannels[i],
                                UpdateFormat, MemoryFormat, VideoEngine);
                RouteChannel(m_pSDK[i], Sources[i],
                                Destinations[i],
                                BLUE_CONNECTOR_PROP_SINGLE_LINK);
        }

        if(m_PlayAudio) {
                //route the same audio to AES as well (AES only supports 8 channels, though)
                val32 = EPOCH_SET_ROUTING(EPOCH_SRC_OUTPUT_MEM_INTERFACE_CHA,
                                EPOCH_DEST_AES_ANALOG_AUDIO_OUTPUT, BLUE_CONNECTOR_PROP_SINGLE_LINK);
                bfcSetCardProperty32(m_pSDK[0], MR2_ROUTING, val32);

                val32 = blue_emb_audio_group1_enable | blue_emb_audio_enable | blue_enable_hanc_timestamp_pkt;
                bfcSetCardProperty32(m_pSDK[0], EMBEDDED_AUDIO_OUTPUT, val32);
        }

        //Set the required video mode
        for(int i = 0; i < m_AttachedDevicesCount; ++i) {
                val32 = VideoMode;
                bfcSetCardProperty32(m_pSDK[i], VIDEO_MODE, val32);
                bfcQueryCardProperty32(m_pSDK[i], VIDEO_MODE, val32);
                if(val32 != VideoMode)
                {
                        throw logic_error("Can't set video mode; FIFO running already?");
                }
        }

        for(int i = 0; i < m_AttachedDevicesCount; ++i) {
                val32 = vc_get_linesize(tile_width, desc.color_spec);
                bfcSetCardProperty32(m_pSDK[i], VIDEO_IMAGE_WIDTH, val32);
                val32 = tile_height;
                bfcSetCardProperty32(m_pSDK[i], VIDEO_IMAGE_HEIGHT, val32);
                val32 = vc_get_linesize(desc.width, desc.color_spec);
                bfcSetCardProperty32(m_pSDK[i], VIDEO_IMAGE_PITCH, val32);
        }

        ULONG GoldenSize = BlueVelvetGolden(VideoMode, MemoryFormat, UpdateFormat);
        ULONG PixelsPerLine = BlueVelvetLinePixels(VideoMode);
        ULONG VideoLines =  BlueVelvetFrameLines(VideoMode, UpdateFormat);
        ULONG BytesPerFrame = BlueVelvetFrameBytes(VideoMode, MemoryFormat, UpdateFormat);
        ULONG BytesPerLine = BlueVelvetLineBytes(VideoMode, MemoryFormat);

        ULONG FrameSize = GoldenSize;
        if(is4K && !isTiled4K) {
                BytesPerFrame = vc_get_linesize(desc.width, desc.color_spec) *
                        desc.height;
                // 4 * GoldenSize should be AFAIK correct but it keeps failing during
                // DMA transfer of last segment of 4K video
                FrameSize = 5 * GoldenSize;
        }

        cout << "Video Golden:          " << GoldenSize << endl;
        cout << "Video Pixels per line: " << PixelsPerLine << endl;
        cout << "Video lines:           " << VideoLines << endl;
        cout << "Video Bytes per frame: " << BytesPerFrame << endl;
        cout << "Video Bytes per line:  " << BytesPerLine << endl;

        for(int i = 0; i < BUFFER_COUNT; ++i) {
                av_buffer *new_buffer = new av_buffer(desc.tile_count, FrameSize);
                m_FreeFrameQueue.push(new_buffer);
        }

        m_frame = vf_alloc_desc(desc);
        for(unsigned int i = 0; i < desc.tile_count; ++i) {
                m_frame->tiles[i].data_len = BytesPerFrame;
        }

        if(!is4K || isTiled4K) {
                m_GoldenSize = GoldenSize;
                for(int i = 0; i < m_AttachedDevicesCount; ++i) {
                        m_Offset[i] = 0;
                }
        } else {
                m_GoldenSize = vc_get_linesize(desc.width, desc.color_spec) * tile_height;
                for(int i = 0; i < m_AttachedDevicesCount; ++i) {
                        int x = i % 2;
                        int y = i / 2;
                        m_Offset[i] = y * m_GoldenSize + x *
                                vc_get_linesize(tile_width, desc.color_spec);
                }
        }

        for(int i = 0; i < m_AttachedDevicesCount; ++i) {
                bfcVideoPlaybackStart(m_pSDK[i], 0, 0);
        }
}

struct video_frame *display_bluefish444_state::getf() throw()
{
        pthread_mutex_lock(&m_lock);
        while(m_FreeFrameQueue.empty()) {
                pthread_cond_wait(&m_BossCv, &m_lock);
        }
        m_pTmpFrame = m_FreeFrameQueue.front();
        m_FreeFrameQueue.pop();
        for(int i = 0; i < m_TileCount; ++i) {
                m_frame->tiles[i].data = (char *) m_pTmpFrame->pVideoBuffer[i];
        }
        pthread_mutex_unlock(&m_lock);

        return m_frame;
}

void display_bluefish444_state::putf(struct video_frame *frame) throw (runtime_error, logic_error)
{
        if (!frame)
                return;

        if (frame->tiles[0].data !=
                        (char *) m_pTmpFrame->pVideoBuffer[0]) {
                throw invalid_argument("Invalid frame supplied!");
        }

        pthread_mutex_lock(&m_lock);
        for(int i = 0; i < m_TileCount; ++i) {
                m_pTmpFrame->pVideoBuffer[i] = (unsigned int *)(void *) m_frame->tiles[i].data;
        }
        m_ReadyFrameQueue.push(m_pTmpFrame);
        pthread_cond_signal(&m_WorkerCv);
        pthread_mutex_unlock(&m_lock);
}

/*
 * Utility functions - from SDK
 */
static void InitOutputChannel(CBLUEVELVET_H pSDK, ULONG DefaultOutputChannel, ULONG UpdateFormat,
                ULONG MemoryFormat, ULONG VideoEngine)
{
        uint32_t val32;

        //MOST IMPORTANT: as the first step set the channel that we want to work with
        val32 = DefaultOutputChannel;
        bfcSetCardProperty32(pSDK, DEFAULT_VIDEO_OUTPUT_CHANNEL, val32);

        //make sure the FIFO hasn't been left running (e.g. application crash before), otherwise we can't change card properties
        bfcVideoPlaybackStop(pSDK, 1, 1);

        val32 = UpdateFormat;
        bfcSetCardProperty32(pSDK, VIDEO_UPDATE_TYPE, val32);

        val32 = MemoryFormat;
        bfcSetCardProperty32(pSDK, VIDEO_MEMORY_FORMAT, val32);

        //Only set the Video Engine after setting up the required video mode, update type and memory format
        val32 = VideoEngine;
        bfcSetCardProperty32(pSDK, VIDEO_OUTPUT_ENGINE, val32);

        val32 = ENUM_BLACKGENERATOR_OFF;
        bfcSetCardProperty32(pSDK, VIDEO_BLACKGENERATOR, val32);
}

static void RouteChannel(CBLUEVELVET_H pSDK, ULONG Source, ULONG Destination, ULONG LinkType)
{
        uint32_t val32;

        val32 = EPOCH_SET_ROUTING(Source, Destination, LinkType);
        bfcSetCardProperty32(pSDK, MR2_ROUTING, val32);
}

/*
 * AUDIO
 */
#ifdef HAVE_BLUE_AUDIO
void display_bluefish444_state::reconfigure_audio(int quant_samples, int channels,
                int sample_rate) throw(runtime_error, logic_error)
{
        if(quant_samples <= 0 || channels <= 0 || sample_rate <= 0) {
                throw logic_error("Wrong audio attributes");
        }

        if(sample_rate != 48000) {
                throw runtime_error("Audio: only 48 kHz audio supported");
        }
        if(channels > 16) {
                throw runtime_error("Audio: more than 16 channels not supported");
        }
        // TODO: fix if we want more this needs some property reset
        if(channels > 4) {
                throw runtime_error("Audio: more than 4 channels not supported");
        }
        if(quant_samples != 16 && quant_samples != 24) {
                throw runtime_error("Audio: only 16 or 24 bits per second supported");
        }

        pthread_spin_lock(&m_AudioSpinLock);
        m_AudioDesc.bps = quant_samples / 8;
        m_AudioDesc.ch_count = channels;
        m_AudioDesc.sample_rate = sample_rate;
        ring_buffer_flush(m_AudioRingBuffer);
        pthread_spin_unlock(&m_AudioSpinLock);
}

void display_bluefish444_state::put_audio_frame(struct audio_frame *frame) throw()
{
        if(!m_PlayAudio)
                return;
        ring_buffer_write(m_AudioRingBuffer, frame->data, frame->data_len);
}
#endif // defined HAVE_BLUE_AUDIO

/*
 * API
 */
static void show_help(void);

static void show_help(void)
{
        int iDevices;
        CBLUEVELVET_H pSDK = bfcFactory();
        bfcEnumerate(pSDK, iDevices);
        bfcDestroy(pSDK);

        cout << "bluefish444 (output) options:" << endl
                << "\tbluefish444[:device=<device_id>]" << endl
                << "\t\tID of the Bluefish device (if more present)" << endl
                << "\t\t" << iDevices << " Bluefish devices found in this system" << endl;
        if(iDevices == 1) {
                cout << " (default)" << endl;
        } else if(iDevices > 1) {
                cout << ", valid indices [1," << iDevices  << "]" << endl;
                cout << "default is 1" << endl;
        }
}

static void display_bluefish444_probe(struct device_info **available_cards, int *count, void (**deleter)(void *))
{
        UNUSED(deleter);
        int iDevices;
        CBLUEVELVET_H pSDK = bfcFactory();
        bfcEnumerate(pSDK, iDevices);
        bfcDestroy(pSDK);

        *available_cards = (struct device_info *) calloc(iDevices, sizeof(struct device_info));
        *count = iDevices;
        for (int i = 0; i < iDevices; i++) {
                sprintf((*available_cards)[i].id, "bluefish444:device=%d", iDevices);
                sprintf((*available_cards)[i].name, "Bluefish444 card #%d", iDevices);
        }
}

static void *display_bluefish444_init(struct module *parent, const char *fmt, unsigned int flags)
{
        UNUSED(parent);
        int deviceId = 1;
        if(fmt){
                if(strcmp(fmt, "help") == 0) {
                        show_help();
                        return NULL;
                } else if(strncasecmp(fmt, "device=", strlen("device=")) == 0) {
                        deviceId = atoi(fmt + strlen("device="));
                } else {
                        cerr << "[Blue444 disp] Unknown parameter: " << fmt << endl;
                        return NULL;
                }
        }

        void *state = NULL;

        try {
                state = new display_bluefish444_state(flags, deviceId);
        } catch(runtime_error &e) {
                cerr << "[Blue444 disp] " << e.what() << endl;
        }
        return state;
}

static struct video_frame *
display_bluefish444_getf(void *state)
{
        display_bluefish444_state *s =
                (display_bluefish444_state *) state;
        struct video_frame *frame = NULL;

        try {
                frame = s->getf();
        } catch(runtime_error &e) {
                cerr << "[Blue444 disp] " << e.what() << endl;
        }

        return frame;
}

static int display_bluefish444_putf(void *state, struct video_frame *frame, int nonblock)
{
        UNUSED(nonblock);
        display_bluefish444_state *s =
                (display_bluefish444_state *) state;
        int ret;

        try {
                s->putf(frame);
                ret = 0;
        } catch(runtime_error &e) {
                cerr << "[Blue444 disp] " << e.what() << endl;
                ret = -1;
        }

        return ret;
}

static int
display_bluefish444_reconfigure(void *state, struct video_desc desc)
{
        display_bluefish444_state *s =
                (display_bluefish444_state *) state;
        int ret;

        try {
                s->reconfigure(desc);
                ret = TRUE;
        } catch(runtime_error &e) {
                cerr << "[Blue444 disp] " << e.what() << endl;
                ret = FALSE;
        }

        return ret;
}

static void display_bluefish444_run(void *state)
{
        UNUSED(state);
}

static void display_bluefish444_done(void *state)
{
        display_bluefish444_state *s =
                (display_bluefish444_state *) state;

        delete s;
}

static int display_bluefish444_get_property(void *state, int property, void *val, size_t *len)
{
        UNUSED(state);
        codec_t codecs[] = { UYVY };
        interlacing_t supported_il_modes[] = {PROGRESSIVE, INTERLACED_MERGED, SEGMENTED_FRAME};
        int rgb_shift[] = {0, 8, 16};
        
        switch (property) {
                case DISPLAY_PROPERTY_CODECS:
                        if(sizeof(codecs) <= *len) {
                                memcpy(val, codecs, sizeof(codecs));
                        } else {
                                return FALSE;
                        }
                        
                        *len = sizeof(codecs);
                        break;
                case DISPLAY_PROPERTY_RGB_SHIFT:
                        if(sizeof(rgb_shift) > *len) {
                                return FALSE;
                        }
                        memcpy(val, rgb_shift, sizeof(rgb_shift));
                        *len = sizeof(rgb_shift);
                        break;
                case DISPLAY_PROPERTY_BUF_PITCH:
                        *(int *) val = PITCH_DEFAULT;
                        *len = sizeof(int);
                        break;
                case DISPLAY_PROPERTY_SUPPORTED_IL_MODES:
                        if(sizeof(supported_il_modes) <= *len) {
                                memcpy(val, supported_il_modes, sizeof(supported_il_modes));
                        } else {
                                return FALSE;
                        }
                        *len = sizeof(supported_il_modes);
                        break;
                case DISPLAY_PROPERTY_VIDEO_MODE:
                        *(int *) val = DISPLAY_PROPERTY_VIDEO_SEPARATE_TILES;
                        break;
                case DISPLAY_PROPERTY_AUDIO_FORMAT:
                        {
                                assert(*len == sizeof(struct audio_desc));
                                struct audio_desc *desc = (struct audio_desc *) val;
                                desc->sample_rate = 48000;
                                desc->ch_count = max(desc->ch_count, 4);
                                desc->codec = AC_PCM;
                                desc->bps = desc->bps < 3 ? 2 : 3;
                        }
                        break;
                default:
                        return FALSE;
        }
        return TRUE;
}

static int display_bluefish444_reconfigure_audio(void *state, int quant_samples, int channels,
                                int sample_rate)
{
#ifdef HAVE_BLUE_AUDIO
        display_bluefish444_state *s =
                (display_bluefish444_state *) state;
        int ret;

        try {
                s->reconfigure_audio(quant_samples, channels, sample_rate);
                ret = TRUE;
        } catch(runtime_error &e) {
                cerr << "[Blue444 disp] " << e.what() << endl;
                ret = FALSE;
        }

        return ret;
#else
        return FALSE;
#endif
}

static void display_bluefish444_put_audio_frame(void *state, struct audio_frame *frame)
{
#ifdef HAVE_BLUE_AUDIO
        display_bluefish444_state *s =
                (display_bluefish444_state *) state;

        try {
                s->put_audio_frame(frame);
        } catch(runtime_error &e) {
                cerr << "[Blue444 disp] " << e.what() << endl;
        }
#endif
}

static const struct video_display_info display_bluefish444_info = {
        display_bluefish444_probe,
        display_bluefish444_init,
        display_bluefish444_run,
        display_bluefish444_done,
        display_bluefish444_getf,
        display_bluefish444_putf,
        display_bluefish444_reconfigure,
        display_bluefish444_get_property,
        display_bluefish444_put_audio_frame,
        display_bluefish444_reconfigure_audio,
};

REGISTER_MODULE(bluefish444, &display_bluefish444_info, LIBRARY_CLASS_VIDEO_DISPLAY, VIDEO_DISPLAY_ABI_VERSION);

