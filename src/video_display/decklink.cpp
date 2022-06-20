/**
 * @file   video_display/decklink.cpp
 * @author Martin Benes     <martinbenesh@gmail.com>
 * @author Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 * @author Petr Holub       <hopet@ics.muni.cz>
 * @author Milos Liska      <xliska@fi.muni.cz>
 * @author Jiri Matela      <matela@ics.muni.cz>
 * @author Dalibor Matura   <255899@mail.muni.cz>
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2010-2021 CESNET, z. s. p. o.
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
#endif // HAVE_CONFIG_H

#define MOD_NAME "[Decklink display] "

#include "audio/types.h"
#include "blackmagic_common.h"
#include "compat/platform_time.h"
#include "debug.h"
#include "host.h"
#include "lib_common.h"
#include "messaging.h"
#include "module.h"
#include "rang.hpp"
#include "rtp/audio_decoders.h"
#include "tv.h"
#include "ug_runtime_error.hpp"
#include "utils/misc.h"
#include "video.h"
#include "video_display.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <iomanip>
#include <mutex>
#include <queue>
#include <string>
#include <vector>

#include "DeckLinkAPIVersion.h"

#ifndef WIN32
#define STDMETHODCALLTYPE
#endif

#define MAX_RESAMPLE_DELTA_DEFAULT 30
#define MIN_RESAMPLE_DELTA_DEFAULT 1
#define TARGET_BUFFER_DEFAULT 2700

static void print_output_modes(IDeckLink *);
static void display_decklink_done(void *state);

// performs command, if failed, displays error and jumps to error label
#define EXIT_IF_FAILED(cmd, name) \
        do {\
                HRESULT result = cmd;\
                if (FAILED(result)) {;\
                        LOG(LOG_LEVEL_ERROR) << MOD_NAME << name << ": " << bmd_hresult_to_string(result) << "\n";\
                        goto error;\
                }\
        } while (0)

// similar as above, but only displays warning
#define CALL_AND_CHECK(cmd, name) \
        do {\
                HRESULT result = cmd;\
                if (FAILED(result)) {;\
                        LOG(LOG_LEVEL_WARNING) << MOD_NAME << name << ": " << bmd_hresult_to_string(result) << "\n";\
                }\
        } while (0)

using namespace std;
using rang::fg;
using rang::style;

static int display_decklink_putf(void *state, struct video_frame *frame, int nonblock);

namespace {
class MovingAverage {
public:
	MovingAverage(unsigned int period) :
		period(period), 
                window(new double[period]), 
                head(NULL), 
                tail(NULL),
		total(0) {
		
                assert(period >= 1);
	}

	~MovingAverage() {
		delete[] window;
	}

        void add(double val) {
		// Init
		if (head == NULL) {
			head = window;
			*head = val;
			tail = head;
			inc(tail);
			total = val;
			return;
		}
		// full?
		if (head == tail) {
			total -= *head;
			inc(head);
		}
 
		*tail = val;
		inc(tail);
		total += val;
	}
 
	// Returns the average of the last P elements added.
	// If no elements have been added yet, returns 0.0
	double avg() const {
		ptrdiff_t size = this->size();
		if (size == 0) {
			return 0; // No entries => 0 average
		}
		return total / (double)size; 
	}
        
        ptrdiff_t size() const {
		if (head == NULL)
			return 0;
		if (head == tail)
			return period;
		return (period + tail - head) % period;
	}
        // returns true if we have filled the period with samples
        ptrdiff_t filled(){
                bool filled = false;
                if (this->size() >= this->period ){
                        filled = true;
                }
                return filled;
        }

        double getTotal() const {
                return total;
        }
 
private:
	unsigned int period;
	double* window; // Holds the values to calculate the average of.
	double* head; 
	double* tail; 
	double total; // Cache the total
 
	void inc(double* &p) {
		if (++p >= window + period) {
			p = window;
		}
	}
};
class PlaybackDelegate : public IDeckLinkVideoOutputCallback // , public IDeckLinkAudioOutputCallback
{
private:
        uint64_t frames_dropped = 0;
        uint64_t frames_flushed = 0;
        uint64_t frames_late = 0;

        friend int ::display_decklink_putf(void *state, struct video_frame *frame, int nonblock);
public:
        virtual ~PlaybackDelegate() = default;
        // IUnknown needs only a dummy implementation
        virtual HRESULT STDMETHODCALLTYPE        QueryInterface (REFIID , LPVOID *)        { return E_NOINTERFACE;}
        virtual ULONG STDMETHODCALLTYPE          AddRef ()                                                                       {return 1;}
        virtual ULONG STDMETHODCALLTYPE          Release ()                                                                      {return 1;}

        virtual HRESULT STDMETHODCALLTYPE        ScheduledFrameCompleted (IDeckLinkVideoFrame* completedFrame, BMDOutputFrameCompletionResult result)
	{
                if (result == bmdOutputFrameDisplayedLate){
                        frames_late += 1;
                        LOG(LOG_LEVEL_VERBOSE) << MOD_NAME "Late frame (total: " << frames_late << ")\n";
                } else if (result == bmdOutputFrameDropped){
                        frames_dropped += 1;
                        LOG(LOG_LEVEL_WARNING) << MOD_NAME "Dropped frame (total: " << frames_dropped << ")\n";
                } else if (result == bmdOutputFrameFlushed){
                        frames_flushed += 1;
                        LOG(LOG_LEVEL_WARNING) << MOD_NAME "Flushed frame (total: " << frames_flushed << ")\n";
                }

		if (log_level >= LOG_LEVEL_DEBUG) {
			IDeckLinkTimecode *timecode = NULL;
			if (completedFrame->GetTimecode ((BMDTimecodeFormat) 0, &timecode) == S_OK) {
				BMD_STR timecode_str;
				if (timecode && timecode->GetString(&timecode_str) == S_OK) {
                                        char *timecode_cstr = get_cstr_from_bmd_api_str(timecode_str);
					LOG(LOG_LEVEL_DEBUG) << "Frame " << timecode_cstr << " output at " <<  time_since_epoch_in_ms() / (double) 1e3 << '\n';
                                        release_bmd_api_str(timecode_str);
                                        free(timecode_cstr);
				}
			}
		}


		completedFrame->Release();
		return S_OK;
	}

        virtual HRESULT STDMETHODCALLTYPE        ScheduledPlaybackHasStopped (){
        	return S_OK;
	}
        //virtual HRESULT         RenderAudioSamples (bool preroll);
};

class DeckLinkFrame;

struct buffer_pool_t {
        queue<DeckLinkFrame *> frame_queue;
        mutex lock;
};

class DeckLinkTimecode : public IDeckLinkTimecode{
                BMDTimecodeBCD timecode;
        public:
                DeckLinkTimecode() : timecode(0) {}
                virtual ~DeckLinkTimecode() = default;
                /* IDeckLinkTimecode */
                virtual BMDTimecodeBCD STDMETHODCALLTYPE GetBCD (void) { return timecode; }
                virtual HRESULT STDMETHODCALLTYPE GetComponents (/* out */ uint8_t *hours, /* out */ uint8_t *minutes, /* out */ uint8_t *seconds, /* out */ uint8_t *frames) { 
                        *frames =   (timecode & 0xf)              + ((timecode & 0xf0) >> 4) * 10;
                        *seconds = ((timecode & 0xf00) >> 8)      + ((timecode & 0xf000) >> 12) * 10;
                        *minutes = ((timecode & 0xf0000) >> 16)   + ((timecode & 0xf00000) >> 20) * 10;
                        *hours =   ((timecode & 0xf000000) >> 24) + ((timecode & 0xf0000000) >> 28) * 10;
                        return S_OK;
                }
                virtual HRESULT STDMETHODCALLTYPE GetString (/* out */ BMD_STR *timecode) {
                        uint8_t hours, minutes, seconds, frames;
                        GetComponents(&hours, &minutes, &seconds, &frames);
                        char *out = (char *) malloc(14);
                        assert(minutes <= 59 && seconds <= 59);
                        sprintf(out, "%02" PRIu8 ":%02" PRIu8 ":%02" PRIu8 ":%02" PRIu8, hours, minutes, seconds, frames);
                        *timecode = get_bmd_api_str_from_cstr(out);
                        free(out);
                        return *timecode ? S_OK : E_FAIL;
                }
                virtual BMDTimecodeFlags STDMETHODCALLTYPE GetFlags (void)        { return bmdTimecodeFlagDefault; }
                virtual HRESULT STDMETHODCALLTYPE GetTimecodeUserBits (/* out */ BMDTimecodeUserBits *userBits) { if (!userBits) return E_POINTER; else return S_OK; }

                /* IUnknown */
                virtual HRESULT STDMETHODCALLTYPE QueryInterface (REFIID , LPVOID *)        {return E_NOINTERFACE;}
                virtual ULONG STDMETHODCALLTYPE         AddRef ()                                                                       {return 1;}
                virtual ULONG STDMETHODCALLTYPE          Release ()                                                                      {return 1;}
                
                void STDMETHODCALLTYPE SetBCD(BMDTimecodeBCD timecode) { this->timecode = timecode; }
};

struct ChromaticityCoordinates
{
        double RedX;
        double RedY;
        double GreenX;
        double GreenY;
        double BlueX;
        double BlueY;
        double WhiteX;
        double WhiteY;
};

constexpr ChromaticityCoordinates kDefaultRec2020Colorimetrics = { 0.708, 0.292, 0.170, 0.797, 0.131, 0.046, 0.3127, 0.3290 };
constexpr double kDefaultMaxDisplayMasteringLuminance        = 1000.0;
constexpr double kDefaultMinDisplayMasteringLuminance        = 0.0001;
constexpr double kDefaultMaxCLL                              = 1000.0;
constexpr double kDefaultMaxFALL                             = 50.0;
enum class HDR_EOTF { NONE = -1, SDR = 0, HDR = 1, PQ = 2, HLG = 3 };

struct HDRMetadata
{
        int64_t                                 EOTF{static_cast<int64_t>(HDR_EOTF::NONE)};
        ChromaticityCoordinates referencePrimaries{kDefaultRec2020Colorimetrics};
        double                                  maxDisplayMasteringLuminance{kDefaultMaxDisplayMasteringLuminance};
        double                                  minDisplayMasteringLuminance{kDefaultMinDisplayMasteringLuminance};
        double                                  maxCLL{kDefaultMaxCLL};
        double                                  maxFALL{kDefaultMaxFALL};

        void                                    Init(const string & fmt);
};

class DeckLinkFrame : public IDeckLinkMutableVideoFrame, public IDeckLinkVideoFrameMetadataExtensions
{
                long width;
                long height;
                long rawBytes;
                BMDPixelFormat pixelFormat;
                unique_ptr<char []> data;

                IDeckLinkTimecode *timecode;

                long ref;

                buffer_pool_t &buffer_pool;
                struct HDRMetadata m_metadata;
        protected:
                DeckLinkFrame(long w, long h, long rb, BMDPixelFormat pf, buffer_pool_t & bp, HDRMetadata const & hdr_metadata);

        public:
                virtual ~DeckLinkFrame();
                static DeckLinkFrame *Create(long width, long height, long rawBytes, BMDPixelFormat pixelFormat, buffer_pool_t & buffer_pool, HDRMetadata const & hdr_metadata);

                /* IUnknown */
                HRESULT STDMETHODCALLTYPE QueryInterface(REFIID, void**) override;
                ULONG STDMETHODCALLTYPE AddRef() override;
                ULONG STDMETHODCALLTYPE Release() override;
                
                /* IDeckLinkVideoFrame */
                long STDMETHODCALLTYPE GetWidth (void) override;
                long STDMETHODCALLTYPE GetHeight (void) override;
                long STDMETHODCALLTYPE GetRowBytes (void) override;
                BMDPixelFormat STDMETHODCALLTYPE GetPixelFormat (void) override;
                BMDFrameFlags STDMETHODCALLTYPE GetFlags (void) override;
                HRESULT STDMETHODCALLTYPE GetBytes (/* out */ void **buffer) override;
                HRESULT STDMETHODCALLTYPE GetTimecode (/* in */ BMDTimecodeFormat format, /* out */ IDeckLinkTimecode **timecode) override;
                HRESULT STDMETHODCALLTYPE GetAncillaryData (/* out */ IDeckLinkVideoFrameAncillary **ancillary) override;

                /* IDeckLinkMutableVideoFrame */
                HRESULT STDMETHODCALLTYPE SetFlags(BMDFrameFlags) override;
                HRESULT STDMETHODCALLTYPE SetTimecode(BMDTimecodeFormat, IDeckLinkTimecode*) override;
                HRESULT STDMETHODCALLTYPE SetTimecodeFromComponents(BMDTimecodeFormat, uint8_t, uint8_t, uint8_t, uint8_t, BMDTimecodeFlags) override;
                HRESULT STDMETHODCALLTYPE SetAncillaryData(IDeckLinkVideoFrameAncillary*) override;
                HRESULT STDMETHODCALLTYPE SetTimecodeUserBits(BMDTimecodeFormat, BMDTimecodeUserBits) override;

                // IDeckLinkVideoFrameMetadataExtensions interface
                HRESULT STDMETHODCALLTYPE GetInt(BMDDeckLinkFrameMetadataID metadataID, int64_t* value) override;
                HRESULT STDMETHODCALLTYPE GetFloat(BMDDeckLinkFrameMetadataID metadataID, double* value) override;
                HRESULT STDMETHODCALLTYPE GetFlag(BMDDeckLinkFrameMetadataID metadataID, BMD_BOOL* value) override;
                HRESULT STDMETHODCALLTYPE GetString(BMDDeckLinkFrameMetadataID metadataID, BMD_STR * value) override;
                HRESULT STDMETHODCALLTYPE GetBytes(BMDDeckLinkFrameMetadataID metadataID, void* buffer, uint32_t* bufferSize) override;

};

class DeckLink3DFrame : public DeckLinkFrame, public IDeckLinkVideoFrame3DExtensions
{
        private:
                using DeckLinkFrame::DeckLinkFrame;
                DeckLink3DFrame(long w, long h, long rb, BMDPixelFormat pf, buffer_pool_t & buffer_pool, HDRMetadata const & hdr_metadata);
                unique_ptr<DeckLinkFrame> rightEye; // rightEye ref count is always >= 1 therefore deleted by owner (this class)

        public:
                ~DeckLink3DFrame();
                static DeckLink3DFrame *Create(long width, long height, long rawBytes, BMDPixelFormat pixelFormat, buffer_pool_t & buffer_pool, HDRMetadata const & hdr_metadata);
                
                /* IUnknown */
                HRESULT STDMETHODCALLTYPE QueryInterface(REFIID, void**) override;
                ULONG STDMETHODCALLTYPE AddRef() override;
                ULONG STDMETHODCALLTYPE Release() override;

                /* IDeckLinkVideoFrame3DExtensions */
                BMDVideo3DPackingFormat STDMETHODCALLTYPE Get3DPackingFormat() override;
                HRESULT STDMETHODCALLTYPE GetFrameForRightEye(IDeckLinkVideoFrame**) override;
};
} // end of unnamed namespace

class DecklinkAudioSummary {
public:
        /**
         * @brief This will detail out the longer running stats of the Decklink. It should be called on every audio frame
         *        but will only print out the report once every 30 seconds.
         */
        void report() {
                std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
                if(std::chrono::duration_cast<std::chrono::seconds>(now - this->last_summary).count() > 10) {                
                        LOG(LOG_LEVEL_INFO) << rang::style::underline << "Decklink stats (cumulative)" 
                                        << rang::style::reset     << " - Total Audio Frames Played: "
                                        << rang::style::bold      << this->frames_played 
                                        << rang::style::reset     << " / Missing Audio Frames: "
                                        << rang::style::bold      << this->frames_missed
                                        << rang::style::reset     << " / Buffer Underflows: " 
                                        << rang::style::bold      << this->buffer_underflow
                                        << rang::style::reset     << " / Buffer Overflows: "
                                        << rang::style::bold      << this->buffer_overflow
                                        << rang::style::reset     << " / Resample (Higher Hz): "
                                        << rang::style::bold      << this->resample_high
                                        << rang::style::reset     << " / Resample (Lower Hz): "
                                        << rang::style::bold      << this->resample_low
                                        << rang::style::reset     << " / Average Buffer: "
                                        << rang::style::bold      << this->buffer_average
                                        << rang::style::reset     << " / Average Added Frames: "
                                        << rang::style::bold      << this->avg_added_frames.avg()
                                        << rang::style::reset     << " / Max time diff audio (ms): "
                                        << rang::style::bold      << this->audio_time_diff_max
                                        << rang::style::reset     << " / Min time diff audio (ms): "
                                        << rang::style::bold      << this->audio_time_diff_min
                                        << "\n";
                        // Reset some of the variables
                        this->audio_time_diff_max = 0;
                        this->audio_time_diff_min = std::numeric_limits<long long>().max();
                        // Ensure that the summary gets called 30 seconds from now
                        this->last_summary = now;
                }
        }

        /**
         * @brief This should be called when a resample is requested that is lower than the
         *        original sample rate.
         */
        void increment_resample_low() {
                this->resample_low++;
        }

        /**
         * @brief This should be called when a resample is requested that is higher than the
         *        original sample rate.
         */
        void increment_resample_high() {
                this->resample_high++;
        }

        /**
         * @brief This should be called when an overflow has occured.
         */
        void increment_buffer_overflow() {
                this->buffer_overflow++;
        }

        /**
         * @brief This should be called when an underflow has occured.
         */
        void increment_buffer_underflow() {
                this->buffer_underflow++;
        }

        /**
         * @brief This should be called when an call to audio put has been called.
         */
        void increment_audio_frames_played() {
                this->frames_played++;
        }

        /**
         * @brief Set the buffer average object
         * 
         * @param buffer_average The average samples in the buffer per channel
         */
        void set_buffer_average(double buffer_average) {
                this->buffer_average = (int32_t)round(buffer_average);
        }

        /**
         * @brief A quick way of roughly calculating if the buffer has emptied by the size of a single audio frame
         *        to keep track of missing audio frames. This doesn't mean that the audio frame was not played, just
         *        that the length of time between audio put calls caused the buffer to empty by half of the average
         *        size of a frame.
         * 
         * @param buffer_samples The amount of audio samples in the buffer.
         * @param samples        The amount of samples that will be written to the buffer.
         */
        void calculate_missing(uint32_t buffer_samples, uint32_t samples) {
                this->avg_added_frames.add(samples);
                if(this->avg_added_frames.filled()) {
                        samples = (uint32_t)this->avg_added_frames.avg();
                }
                // Check to see if the amount in the buffer has dropped by over half the average
                // number of samples being written. If so, we likely dropped a frame.
                if(prev_buffer_samples >= 0 && (uint32_t)this->prev_buffer_samples > buffer_samples + (samples / 2)) {
                        this->frames_missed++;
                }
                this->prev_buffer_samples = buffer_samples;
        }

        /**
         * @brief This function should be called at the beginning of put audio to record the
         *        difference between calls.
         * 
         */
        void record_audio_time_diff() {
                // CHeck the previous time has been initialised
                if(this->prev_audio_end.time_since_epoch().count() != 0) {
                        // Collect the time now and do a comparison to the time when we ended the previous function call
                        std::chrono::high_resolution_clock::time_point audio_begin = std::chrono::high_resolution_clock::now();
                        std::chrono::milliseconds time_diff = std::chrono::duration_cast<std::chrono::milliseconds>(audio_begin - this->prev_audio_end);

                        // Set a max or min if the timing is outside of whats already been collected
                        long long duration_diff = time_diff.count();
                        if(duration_diff > this->audio_time_diff_max) {
                                this->audio_time_diff_max = duration_diff;
                        }
                        else if(duration_diff < this->audio_time_diff_min) {
                                this->audio_time_diff_min = duration_diff;
                        }
                }
        }

        /**
         * @brief Mark the end of the put audio function
         * 
         */
        void mark_audio_time_end() {
                this->prev_audio_end = std::chrono::high_resolution_clock::now();
        }
private:
        // Keep a track of the amount in the decklink buffer
        int32_t prev_buffer_samples = -1;
        // How many frames have been successfully written
        uint32_t frames_played = 0;
        // How many times the buffer dropped avg amount of frames being added
        uint32_t frames_missed = 0;
        MovingAverage avg_added_frames{250};
        // How many buffer underflows and overflows have occured.
        uint32_t buffer_underflow = 0;
        uint32_t buffer_overflow = 0;
        // How many times it was requested a higher or lower sample rate
        uint32_t resample_high = 0;
        uint32_t resample_low = 0;
        // Sample count average
        uint32_t buffer_average = 0;
        // Timing between calls of audio put
        std::chrono::high_resolution_clock::time_point prev_audio_end{};
        long long audio_time_diff_max = 0;
        long long audio_time_diff_min = std::numeric_limits<long long>().max();
        // We want to the summary to be outputted every 30 or so seconds. So keep track of
        // the last we outputted data.
        std::chrono::steady_clock::time_point last_summary = std::chrono::steady_clock::now();
};


/**
 * @todo
 * - handle network losses
 * - handle underruns
 * - what about jitter - while computing the dst sample rate, the sampling interval (m_total) must be "long"
 */
class AudioDriftFixer {
public:
        AudioDriftFixer(int buffer_samples, int delta_samples, 
                          int target_buffer_fill = 0, int positive_jitter = 0,
                          int negative_jitter = 0) : average_buffer_samples(buffer_samples),average_delta(delta_samples),
                                                     target_buffer_fill(target_buffer_fill), pos_jitter(positive_jitter),
                                                     neg_jitter(negative_jitter){}

        bool m_enabled = true;

        /**
         * @brief Set the max hz object
         * 
         * @param max_hz The maximum hz delta that can be applied to fix the drift
         */
        void set_max_hz(uint32_t max_hz) {
                this->max_hz = max_hz;
        }

        /**
         * @brief Set the min hz object
         * 
         * @param min_hz The minimum hz delta that can be applied to fix the drift
         */
        void set_min_hz(uint32_t min_hz) {
                this->min_hz = min_hz;
        }

        /**
         * @brief Set the target buffer object
         * 
         * @param target_buffer The target buffer of samples per channel
         */
        void set_target_buffer(uint32_t target_buffer) {
                this->target_buffer_fill = target_buffer;
        }

        /**
         * @brief Set the summary object
         * 
         * @param audio_summary The audio summary pointer
         */
        void set_summary(DecklinkAudioSummary *audio_summary) {
                this->audio_summary = audio_summary;
        }

        /**
         * @brief Set the root object
         * 
         * @param root The root module
         */
        void set_root(module *root) {
                m_root = root;
        }

        /**
         * @brief Get the average sample count per channel
         * 
         * @return double The average of the buffer over the last X frames
         */
        double get_buffer_avg() {
                return this->average_buffer_samples.avg();
        }

        /**
         * @brief This function will check the buffer delta and will return a delta in the sample rate
         *        that is required in order to offset the delta. This is scaled between the class
         *        members of min_hz and max_hz. The delta also has a max and min for the scaling which
         *        are defined by the min_buffer and the max_buffer. This ensures that very large deltas
         *        cannot cause large jumps in the resample rate that are audible (and that small deltas)
         *        do not create a resampling rate that is too small to have impact on the buffer.
         * 
         * @param delta The delta between the average buffer size and the target buffer size to calculate a resample
         *               delta to offset the difference.
         * @return double A resample delta that can be added or subtracted from the original resample rate to move the
         *                average buffer size to the target buffer size.
         */
        double scale_buffer_delta(int delta) {
                // Get a positive delta so that the scale can be calculated properly
                delta = abs(delta);
                // Check the boundaries for the scaling calculation
                if((uint32_t)delta > this->max_buffer) {
                        delta = this->max_buffer;
                }
                else if ((uint32_t)delta < this->min_buffer) {
                        delta = this->min_buffer;
                }
                return (((this->max_hz - this->min_hz) * (delta - this->min_buffer)) / (this->max_buffer - this->min_buffer)) + this->min_hz;
        }

        /// @retval flag if the audio frame should be written
        bool update(int buffered_count) {
                if (!this->m_enabled) {
                        return true;
                }

                // Add the amount currently in the buffer to the moving average, and calculate the delta between that and the previous amount
                // Store the previous buffer count so we can calculate this next frame.
                this->average_buffer_samples.add((double)buffered_count);
                this->average_delta.add((double)abs((int32_t)buffered_count - (int32_t)previous_buffer));
                this->previous_buffer = buffered_count;
                
                long long dst_frame_rate = 0;
                // Calculate the average
                uint32_t average_buffer_depth = (uint32_t)(this->average_buffer_samples.avg());

                int resample_hz = dst_frame_rate = (bmdAudioSampleRate48kHz) * BASE;

                // Check to see if our buffered samples has enough to calculate a good average
                if (this->average_buffer_samples.filled()) {
                        // @todo might be worth trying to make this more dynamic so that certain input values
                        // for different cards can be applied
                        // Check to see if we have a target amount of the buffer we'd like to fill
                        if(this->pos_jitter == 0) {                           
                                this->pos_jitter = AudioDriftFixer::POS_JITTER_DEFAULT;
                        }
                        if(this->neg_jitter == 0) {
                                this->neg_jitter = AudioDriftFixer::NEG_JITTER_DEFAULT;
                        }

                        // Check whether there needs to be any resampling                        
                        if (average_buffer_depth  > target_buffer_fill + this->pos_jitter)
                        {
                                // The buffer is too large, so we need to resample down to remove some frames
                                resample_hz = (int)this->scale_buffer_delta(average_buffer_depth - target_buffer_fill - this->pos_jitter);
                                dst_frame_rate = (bmdAudioSampleRate48kHz - resample_hz) * BASE;
                                this->audio_summary->increment_resample_low();
                        } else if(average_buffer_depth < target_buffer_fill - this->neg_jitter) {
                                 // The buffer is too small, so we need to resample up to generate some additional frames
                                resample_hz = (int)this->scale_buffer_delta(target_buffer_fill - average_buffer_depth - this->neg_jitter);
                                dst_frame_rate = (bmdAudioSampleRate48kHz + resample_hz) * BASE;
                                this->audio_summary->increment_resample_high();
                        } else {
                                dst_frame_rate = (bmdAudioSampleRate48kHz) * BASE;
                        }       
                }

                LOG(LOG_LEVEL_DEBUG) << MOD_NAME << " UPDATE playing speed " <<  average_buffer_depth << " vs " << buffered_count << " " << average_delta.avg() << " average_velocity " << resample_hz << " resample_hz\n";

   
                if (dst_frame_rate != 0) {
                        auto *m = new msg_universal((string(MSG_UNIVERSAL_TAG_AUDIO_DECODER) + to_string(dst_frame_rate << ADEC_CH_RATE_SHIFT | BASE)).c_str());
                        LOG(LOG_LEVEL_VERBOSE) << MOD_NAME "Sending resample request " << dst_frame_rate << "/" << BASE << "\n";
                        assert(m_root != nullptr);
                        auto *response = send_message_sync(m_root, "audio.receiver.decoder", reinterpret_cast<message *>(m), 100, SEND_MESSAGE_FLAG_NO_STORE);
                        if (!RESPONSE_SUCCESSFUL(response_get_status(response))) {
                                LOG(LOG_LEVEL_WARNING) << MOD_NAME "Unable to send resample message: " << response_get_text(response) << " (" << response_get_status(response) << ")\n";
                        }
                        free_response(response);
                }

                return true;
        }

private:
        static constexpr unsigned long BASE = (1U<<8U);
        struct module *m_root = nullptr;

        MovingAverage average_buffer_samples;
        MovingAverage average_delta;

        uint32_t target_buffer_fill = TARGET_BUFFER_DEFAULT;
        uint32_t previous_buffer = 0;

        // The min and max Hz changes we can resample between
        uint32_t min_hz = MIN_RESAMPLE_DELTA_DEFAULT;
        uint32_t max_hz = MAX_RESAMPLE_DELTA_DEFAULT;
        // The min and max values to scale between
        uint32_t min_buffer = 100;
        uint32_t max_buffer = 600;
        // Calculate the jitter so that we're within an acceptable range
        uint32_t pos_jitter = 0;
        uint32_t neg_jitter = 0;
        // Currently unused but might form a part of a more dynamic
        // solution to finding good jitter values in the future. @todo
        uint32_t max_avg = 3650;
        uint32_t min_avg = 1800;

        // Store a audio_summary of resampling
        DecklinkAudioSummary *audio_summary = nullptr;

        static const uint32_t POS_JITTER_DEFAULT = 600;
        static const uint32_t NEG_JITTER_DEFAULT = 600;
};

#define DECKLINK_MAGIC 0x12de326b

struct device_state {
        PlaybackDelegate           *delegate;
        IDeckLink                  *deckLink;
        IDeckLinkOutput            *deckLinkOutput;
        IDeckLinkConfiguration     *deckLinkConfiguration;
        IDeckLinkProfileAttributes *deckLinkAttributes;
};

struct state_decklink {
        uint32_t            magic = DECKLINK_MAGIC;
        chrono::high_resolution_clock::time_point t0 = chrono::high_resolution_clock::now();

        vector<struct device_state> state;

        BMDTimeValue        frameRateDuration{};
        BMDTimeScale        frameRateScale{};

        DeckLinkTimecode    *timecode{}; ///< @todo Should be actually allocated dynamically and
                                       ///< its lifespan controlled by AddRef()/Release() methods

        struct video_desc   vid_desc{};
        struct audio_desc   aud_desc{};

        unsigned long int   frames            = 0;
        unsigned long int   frames_last       = 0;
        bool                stereo            = false;
        bool                initialized_audio = false;
        bool                initialized_video = false;
        bool                emit_timecode     = false;
        int                 devices_cnt       = 1;
        bool                play_audio        = false; ///< the BMD device will be used also for output audio

        BMDPixelFormat      pixelFormat{};

        uint32_t            link_req = BMD_OPT_DEFAULT;
        uint32_t            profile_req = BMD_OPT_DEFAULT; // BMD_OPT_DEFAULT, BMD_OPT_KEEP, bmdDuplexHalf or one of BMDProfileID
        char                sdi_dual_channel_level = BMD_OPT_DEFAULT; // 'A' - level A, 'B' - level B
        bool                quad_square_division_split = true;
        BMDVideoOutputConversionMode conversion_mode{};
        HDRMetadata         requested_hdr_mode{};

        buffer_pool_t       buffer_pool;

        bool                low_latency       = true;

        mutex               reconfiguration_lock; ///< for audio and video reconf to be mutually exclusive

        AudioDriftFixer audio_drift_fixer{250, 25, 2700, 5, 5};

        uint32_t            last_buffered_samples;
        int32_t             drift_since_last_correction;
        DecklinkAudioSummary audio_summary{};

 };

static void show_help(bool full);

static void show_help(bool full)
{
        IDeckLinkIterator*              deckLinkIterator;
        IDeckLink*                      deckLink;
        int                             numDevices = 0;

        printf("Decklink (output) options:\n");
        cout << style::bold << fg::red << "\t-d decklink" << fg::reset << "[:fullhelp][:device=<device(s)>][:timecode][:<X>-link][:Level{A|B}][:3D[:HDMI3DPacking=<packing>]][:audio_level={line|mic}][:conversion=<fourcc>][:Use1080PsF][:[no-]low-latency][:profile=<X>|:half-duplex][:quad-[no-]square][:HDR[=<t>]][:maxresample=<N>][:minresample=<N>][:targetbuffer=<N>][:drift_fix]\n" << style::reset;
        cout << "Options:\n";
        cout << style::bold << "\tfullhelp" << style::reset << " displays help for further options\n";
        cout << style::bold << "\t<device(s)>" << style::reset << " is comma-separated indices or names of output devices\n";
        cout << style::bold << "\tsingle-link/dual-link/quad-link" << style::reset << " specifies if the video output will be in a single-link (HD/3G/6G/12G), dual-link HD-SDI mode or quad-link HD/3G/6G/12G\n";
        cout << style::bold << "\tLevelA/LevelB" << style::reset << " specifies 3G-SDI output level\n";
        cout << style::bold << "\tdrift_fix" << style::reset << " activates a drift fix for the Decklink cards. The decklink card clocks will slowly drift overtime which can eventually cause a\n";
        cout << "\tbuffer underflow or overflow. A dynamic resampler is utilised in order to stretch (or reduce) audio by a small amount to counter the drift.\n";
        cout << style::bold << "\tmaxresample=<N> - " << style::reset << "The maximum amount the resample delta can be when scaling is applied. Measured in Hz.\n";
        cout << style::bold << "\tminresample=<N> - " << style::reset << "The minimum amount the resample delta can be when scaling is applied. Measured in Hz.\n";
        cout << style::bold << "\ttargetbuffer=<N> - " << style::reset << "The target amount of samples to have in the buffer (per channel).\n";
        if (!full) {
                cout << style::bold << "\tconversion" << style::reset << " - use '-d decklink:fullhelp' for list of conversions\n";
        } else {
                cout << style::bold << "\t[no-]quad-square" << style::reset << " - set Quad-link SDI is output in Square Division Quad Split mode\n";
                cout << "\toutput conversion can be:\n" <<
                                style::bold << "\t\tnone" << style::reset << " - no conversion\n" <<
                                style::bold << "\t\tltbx" << style::reset << " - down-converted letterbox SD\n" <<
                                style::bold << "\t\tamph" << style::reset << " - down-converted anamorphic SD\n" <<
                                style::bold << "\t\t720c" << style::reset << " - HD720 to HD1080 conversion\n" <<
                                style::bold << "\t\tHWlb" << style::reset << " - simultaneous output of HD and down-converted letterbox SD\n" <<
                                style::bold << "\t\tHWam" << style::reset << " - simultaneous output of HD and down-converted anamorphic SD\n" <<
                                style::bold << "\t\tHWcc" << style::reset << " - simultaneous output of HD and center cut SD\n" <<
                                style::bold << "\t\txcap" << style::reset << " - simultaneous output of 720p and 1080p cross-conversion\n" <<
                                style::bold << "\t\tua7p" << style::reset << " - simultaneous output of SD and up-converted anamorphic 720p\n" <<
                                style::bold << "\t\tua1i" << style::reset << " - simultaneous output of SD and up-converted anamorphic 1080i\n" <<
                                style::bold << "\t\tu47p" << style::reset << " - simultaneous output of SD and up-converted anamorphic widescreen aspcet ratip 14:9 to 720p\n" <<
                                style::bold << "\t\tu41i" << style::reset << " - simultaneous output of SD and up-converted anamorphic widescreen aspcet ratip 14:9 to 1080i\n" <<
                                style::bold << "\t\tup7p" << style::reset << " - simultaneous output of SD and up-converted pollarbox 720p\n" <<
                                style::bold << "\t\tup1i" << style::reset << " - simultaneous output of SD and up-converted pollarbox 1080i\n";
                cout << style::bold << "\tHDMI3DPacking" << style::reset << " can be:\n" <<
				style::bold << "\t\tSideBySideHalf, LineByLine, TopAndBottom, FramePacking, LeftOnly, RightOnly\n" << style::reset;
                cout << style::bold << "\tUse1080PsF[=true|false|keep]" << style::reset << " flag sets use of PsF on output instead of progressive (default is false)\n";
                cout << style::bold << "\tprofile=<P>\n" << style::reset;
                cout << "\t\tUse desired device profile: " << style::bold << "1dfd" << style::reset << ", "
                        << style::bold << "1dhd" << style::reset << ", "
                        << style::bold << "2dfd" << style::reset << ", "
                        << style::bold << "2dhd" << style::reset << " or "
                        << style::bold << "4dhd" << style::reset << ". See SDK manual for details. Use "
                        << style::bold << "keep" << style::reset << " to disable automatic selection.\n";
                cout << style::bold << "\tHDR[=HDR|PQ|HLG|<int>|help]" << style::reset << " - enable HDR metadata (optionally specifying EOTF, int 0-7 as per CEA 861.), help for extended help\n";
        }
        cout << style::bold << "\thalf-duplex" << style::reset
                << " - set a profile that allows maximal number of simultaneous IOs\n";
        cout << "\nIf " << style::bold << "audio_level" << style::reset << " is " << style::bold << "mic" << style::reset << " audio analog level is set to maximum attenuation on audio output.\n";
        cout << "Recognized pixel formats:";
        for_each(uv_to_bmd_codec_map.cbegin(), uv_to_bmd_codec_map.cend(), [](auto const &i) { cout << " " << style::bold << get_codec_name(i.first) << style::reset; } );
        cout << "\n";

        // Create an IDeckLinkIterator object to enumerate all DeckLink cards in the system
        deckLinkIterator = create_decklink_iterator(true);
        if (deckLinkIterator == NULL) {
                return;
        }
        
        // Enumerate all cards in this system
        while (deckLinkIterator->Next(&deckLink) == S_OK)
        {
                string deviceName = bmd_get_device_name(deckLink);
                if (deviceName.empty()) {
                        deviceName = "(unable to get name)";
                }

                // *** Print the model name of the DeckLink card
                cout << "\ndevice: " << style::bold << numDevices << style::reset << ") "
			<< style::bold << deviceName << style::reset << "\n";
                print_output_modes(deckLink);
                
                // Increment the total number of DeckLink cards found
                numDevices++;
        
                // Release the IDeckLink instance when we've finished with it to prevent leaks
                deckLink->Release();
        }
        
        deckLinkIterator->Release();

        decklink_uninitialize();

        // If no DeckLink cards were found in the system, inform the user
        if (numDevices == 0)
        {
                log_msg(LOG_LEVEL_WARNING, "\nNo Blackmagic Design devices were found.\n");
                return;
        } 

        printf("\n");
        if (full) {
                print_decklink_version();
                printf("\n");
        }
}


static struct video_frame *
display_decklink_getf(void *state)
{
        struct state_decklink *s = (struct state_decklink *)state;
        assert(s->magic == DECKLINK_MAGIC);

        if (!s->initialized_video) {
                return nullptr;
        }

        struct video_frame *out = vf_alloc_desc(s->vid_desc);
        auto deckLinkFrames =  new vector<IDeckLinkMutableVideoFrame *>(s->devices_cnt);
        out->callbacks.dispose_udata = (void *) deckLinkFrames;
        out->callbacks.dispose = [](struct video_frame *frame) {
                delete (vector<IDeckLinkMutableVideoFrame *> *) frame->callbacks.dispose_udata;
                vf_free(frame);
        };

        for (unsigned int i = 0; i < s->vid_desc.tile_count; ++i) {
                const int linesize = vc_get_linesize(s->vid_desc.width, s->vid_desc.color_spec);
                IDeckLinkMutableVideoFrame *deckLinkFrame = nullptr;
                lock_guard<mutex> lg(s->buffer_pool.lock);

                while (!s->buffer_pool.frame_queue.empty()) {
                        auto tmp = s->buffer_pool.frame_queue.front();
                        IDeckLinkMutableVideoFrame *frame;
                        if (s->stereo)
                                frame = dynamic_cast<DeckLink3DFrame *>(tmp);
                        else
                                frame = dynamic_cast<DeckLinkFrame *>(tmp);
                        s->buffer_pool.frame_queue.pop();
                        if (!frame || // wrong type
                                        frame->GetWidth() != (long) s->vid_desc.width ||
                                        frame->GetHeight() != (long) s->vid_desc.height ||
                                        frame->GetRowBytes() != linesize ||
                                        frame->GetPixelFormat() != s->pixelFormat) {
                                delete tmp;
                        } else {
                                deckLinkFrame = frame;
                                deckLinkFrame->AddRef();
                                break;
                        }
                }
                if (!deckLinkFrame) {
                        if (s->stereo)
                                deckLinkFrame = DeckLink3DFrame::Create(s->vid_desc.width,
                                                s->vid_desc.height, linesize,
                                                s->pixelFormat, s->buffer_pool, s->requested_hdr_mode);
                        else
                                deckLinkFrame = DeckLinkFrame::Create(s->vid_desc.width,
                                                s->vid_desc.height, linesize,
                                                s->pixelFormat, s->buffer_pool, s->requested_hdr_mode);
                }
                (*deckLinkFrames)[i] = deckLinkFrame;

                deckLinkFrame->GetBytes((void **) &out->tiles[i].data);

                if (s->stereo) {
                        IDeckLinkVideoFrame     *deckLinkFrameRight = nullptr;
                        DeckLink3DFrame *frame3D = dynamic_cast<DeckLink3DFrame *>(deckLinkFrame);
                        assert(frame3D != nullptr);
                        frame3D->GetFrameForRightEye(&deckLinkFrameRight);
                        deckLinkFrameRight->GetBytes((void **) &out->tiles[1].data);
                        // release immedieatelly (parent still holds the reference)
                        deckLinkFrameRight->Release();

                        ++i;
                }
        }

        return out;
}

static void update_timecode(DeckLinkTimecode *tc, double fps)
{
        const float epsilon = 0.005;
        uint8_t hours, minutes, seconds, frames;
        BMDTimecodeBCD bcd;
        bool dropFrame = false;

        if(ceil(fps) - fps > epsilon) { /* NTSCi drop framecode  */
                dropFrame = true;
        }

        tc->GetComponents (&hours, &minutes, &seconds, &frames);
        frames++;

        if((double) frames > fps - epsilon) {
                frames = 0;
                seconds++;
                if(seconds >= 60) {
                        seconds = 0;
                        minutes++;
                        if(dropFrame) {
                                if(minutes % 10 != 0)
                                        seconds = 2;
                        }
                        if(minutes >= 60) {
                                minutes = 0;
                                hours++;
                                if(hours >= 24) {
                                        hours = 0;
                                }
                        }
                }
        }

        bcd = (frames % 10) | (frames / 10) << 4 | (seconds % 10) << 8 | (seconds / 10) << 12 | (minutes % 10)  << 16 | (minutes / 10) << 20 |
                (hours % 10) << 24 | (hours / 10) << 28;

        tc->SetBCD(bcd);
}

static int display_decklink_putf(void *state, struct video_frame *frame, int nonblock)
{
        struct state_decklink *s = (struct state_decklink *)state;

        if (frame == NULL)
                return FALSE;

        UNUSED(nonblock);

        assert(s->magic == DECKLINK_MAGIC);
        /*
        timeInFrame is not dcoumented in the SDK, so putting this here for information.
        The timeInFrame loops in range from 0 to ticksPerFrame-1 for each frame
        The timeInFrame is reset by EOF interrupt from DeckLink hardware on output
        After IDeckLinkOutput::EnableVideoOutput, there is some relocking of output to requested video mode, 
        in this time there are no EOF signals, so the timeInFrame is not reset.
        There is slight variability in the lock time between runs as EnableVideoOutput is called from different timepoint in frame period.
        The below will give an idea of the skew between the source clock and the blackmagic hardware clock.
        */
        BMDTimeValue blk_start_time = 0;
        BMDTimeValue blk_start_timeInFrame = 0;
        BMDTimeValue blk_start_ticksPerFrame =0;
        s->state.at(0).deckLinkOutput->GetHardwareReferenceClock(s->frameRateScale, &blk_start_time, &blk_start_timeInFrame, &blk_start_ticksPerFrame);
        
        uint32_t i;

        s->state.at(0).deckLinkOutput->GetBufferedVideoFrameCount(&i);

        auto t0 = chrono::high_resolution_clock::now();

        //if (i > 2) 
        if (0) 
                fprintf(stderr, "Frame dropped!\n");
        else {
                for (int j = 0; j < s->devices_cnt; ++j) {
                        IDeckLinkMutableVideoFrame *deckLinkFrame =
                                (*((vector<IDeckLinkMutableVideoFrame *> *) frame->callbacks.dispose_udata))[j];
                        if(s->emit_timecode) {
                                deckLinkFrame->SetTimecode(bmdTimecodeRP188Any, s->timecode);
                        }

                        if (s->low_latency) {
                                s->state[j].deckLinkOutput->DisplayVideoFrameSync(deckLinkFrame);
                                deckLinkFrame->Release();
                        } else {
                                s->state[j].deckLinkOutput->ScheduleVideoFrame(deckLinkFrame,
                                                s->frames * s->frameRateDuration, s->frameRateDuration, s->frameRateScale);
                        }
                }
                s->frames++;
                if(s->emit_timecode) {
                        update_timecode(s->timecode, s->vid_desc.fps);
                }
        }

        frame->callbacks.dispose(frame);


        LOG(LOG_LEVEL_DEBUG) << MOD_NAME "putf - " << i << " frames buffered, lasted " << setprecision(2) << chrono::duration_cast<chrono::duration<double>>(chrono::high_resolution_clock::now() - t0).count() * 1000.0 << " ms.\n";
        BMDTimeValue blk_end_time = 0;
        BMDTimeValue blk_end_timeInFrame = 0;
        BMDTimeValue blk_end_ticksPerFrame =0;
        BMDTimeValue blk_write_duration =0;
      

        s->state.at(0).deckLinkOutput->GetHardwareReferenceClock(s->frameRateScale, &blk_end_time, &blk_end_timeInFrame, &blk_end_ticksPerFrame);
        if (blk_end_timeInFrame >= blk_start_timeInFrame){
                blk_write_duration = blk_end_timeInFrame - blk_start_timeInFrame;
        } else{ 
                // we have wrapped 
                BMDTimeValue end_time = blk_end_timeInFrame + blk_end_ticksPerFrame;
                blk_write_duration = end_time - blk_start_timeInFrame;
        }
        LOG(LOG_LEVEL_DEBUG) << MOD_NAME << " putf Video inframe "  << blk_start_timeInFrame << " start, "
                                                                      << blk_end_timeInFrame << " end, " 
                                                                      << blk_write_duration << " duration.\n";
        if (blk_end_timeInFrame - blk_start_timeInFrame > 80) {
                LOG(LOG_LEVEL_DEBUG) << MOD_NAME << " Video Inframe took longer than expected " << blk_write_duration<<"\n";
        }
        LOG(LOG_LEVEL_VERBOSE) << MOD_NAME << " putf Video BlkMagic clock " << blk_start_time << " start, "
                                                                            << blk_end_time << " end, "
                                                                            << blk_end_time - blk_start_time <<" duration.\n";
        auto t1 = chrono::high_resolution_clock::now();
        LOG(LOG_LEVEL_DEBUG) << MOD_NAME "putf - " << i << " frames buffered, lasted " << setprecision(2) << chrono::duration_cast<chrono::duration<double>>(t1 - t0).count() * 1000.0 << " ms.\n";

        if (chrono::duration_cast<chrono::seconds>(t1 - s->t0).count() > 1) {
                double seconds = chrono::duration_cast<chrono::duration<double>>(t1 - s->t0).count();
                double fps = (s->frames - s->frames_last) / seconds;
                if (log_level <= LOG_LEVEL_INFO) {
                        log_msg(LOG_LEVEL_INFO, MOD_NAME "%lu frames in %g seconds = %g FPS\n",
                                        s->frames - s->frames_last, seconds, fps);
                } else {
                        LOG(LOG_LEVEL_DEBUG) << MOD_NAME << s->frames - s->frames_last <<
                                " frames in " << seconds << " seconds = " << fps << " FPS ("
                                << s->state.at(0).delegate->frames_late << " late, "
                                << s->state.at(0).delegate->frames_dropped << " dropped, "
                                << s->state.at(0).delegate->frames_flushed << " flushed cumulative)\n";
                }
                s->t0 = t1;
                s->frames_last = s->frames;
        }

        return 0;
}

static BMDDisplayMode get_mode(IDeckLinkOutput *deckLinkOutput, struct video_desc desc, BMDTimeValue *frameRateDuration,
		BMDTimeScale        *frameRateScale)
{	IDeckLinkDisplayModeIterator     *displayModeIterator;
        IDeckLinkDisplayMode*             deckLinkDisplayMode;
        BMDDisplayMode			  displayMode = bmdModeUnknown;
        
        // Populate the display mode combo with a list of display modes supported by the installed DeckLink card
        if (FAILED(deckLinkOutput->GetDisplayModeIterator(&displayModeIterator)))
        {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Fatal: cannot create display mode iterator.\n");
                return bmdModeUnknown;
        }

        while (displayModeIterator->Next(&deckLinkDisplayMode) == S_OK)
        {
                BMD_STR modeNameString;
                if (deckLinkDisplayMode->GetName(&modeNameString) == S_OK)
                {
                        char *modeNameCString = get_cstr_from_bmd_api_str(modeNameString);
                        if (deckLinkDisplayMode->GetWidth() == (long) desc.width &&
                                        deckLinkDisplayMode->GetHeight() == (long) desc.height)
                        {
                                double displayFPS;
                                BMDFieldDominance dominance;
                                bool interlaced;

                                dominance = deckLinkDisplayMode->GetFieldDominance();
                                if (dominance == bmdLowerFieldFirst ||
                                                dominance == bmdUpperFieldFirst) {
					if (dominance == bmdLowerFieldFirst) {
						log_msg(LOG_LEVEL_WARNING, MOD_NAME "Lower field first format detected, fields can be switched! If so, please report a bug to " PACKAGE_BUGREPORT "\n");
					}
                                        interlaced = true;
                                } else { // progressive, psf, unknown
                                        interlaced = false;
                                }

                                deckLinkDisplayMode->GetFrameRate(frameRateDuration,
                                                frameRateScale);
                                displayFPS = (double) *frameRateScale / *frameRateDuration;
                                if(fabs(desc.fps - displayFPS) < 0.01 && (desc.interlacing == INTERLACED_MERGED ? interlaced : !interlaced)
                                  )
                                {
                                        log_msg(LOG_LEVEL_INFO, MOD_NAME "Selected mode: %s\n", modeNameCString);
                                        displayMode = deckLinkDisplayMode->GetDisplayMode();
                                        release_bmd_api_str(modeNameString);
                                        free(modeNameCString);
                                        deckLinkDisplayMode->Release();
                                        break;
                                }
                        }
                        release_bmd_api_str(modeNameString);
                        free((void *) modeNameCString);
                }
                deckLinkDisplayMode->Release();
        }
        displayModeIterator->Release();
        
        return displayMode;
}

/**
 * @todo
 * In non-low-latency mode, StopScheduledPlayback should be called. However, since this
 * function is called from different thread than audio-related stuff and these things
 * are not synchronized in any way, it looks like to be more appropriate not to call it,
 * as it doesn't break things up. In low latency mode, this is not an issue.
 */
static int
display_decklink_reconfigure_video(void *state, struct video_desc desc)
{
        struct state_decklink            *s = (struct state_decklink *)state;

        BMDDisplayMode                    displayMode;
        BMD_BOOL                          supported;
        HRESULT                           result;

        unique_lock<mutex> lk(s->reconfiguration_lock);

        assert(s->magic == DECKLINK_MAGIC);
        
        s->vid_desc = desc;

        auto it = std::find_if(uv_to_bmd_codec_map.begin(),
                        uv_to_bmd_codec_map.end(),
                        [&desc](const std::pair<codec_t, BMDPixelFormat>& el){ return el.first == desc.color_spec; });
        if (it == uv_to_bmd_codec_map.end()) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unsupported pixel format!\n");
                goto error;
        }
        s->pixelFormat = it->second;

        if (s->initialized_video) {
                for (int i = 0; i < s->devices_cnt; ++i) {
                        CALL_AND_CHECK(s->state.at(i).deckLinkOutput->DisableVideoOutput(),
                                        "DisableVideoOutput");
                }
                s->initialized_video = false;
        }

        if (s->stereo) {
                if ((int) desc.tile_count != 2) {
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "In stereo mode exactly "
                                        "2 streams expected, %d received.\n", desc.tile_count);
                        goto error;
                }
        } else {
                if ((int) desc.tile_count == 2) {
                        log_msg(LOG_LEVEL_WARNING, MOD_NAME "Received 2 streams but stereo mode is not enabled! Did you forget a \"3D\" parameter?\n");
                }
                if ((int) desc.tile_count > s->devices_cnt) {
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Expected at most %d streams. Got %d.\n", s->devices_cnt,
                                        desc.tile_count);
                        goto error;
                } else if ((int) desc.tile_count < s->devices_cnt) {
                        log_msg(LOG_LEVEL_WARNING, MOD_NAME "Received %d streams but %d devices are used!.\n", desc.tile_count, s->devices_cnt);
                }
        }


        for (int i = 0; i < s->devices_cnt; ++i) {
                BMDVideoOutputFlags outputFlags= bmdVideoOutputFlagDefault;
                BMDSupportedVideoModeFlags supportedFlags = bmdSupportedVideoModeDefault;

                displayMode = get_mode(s->state.at(i).deckLinkOutput, desc, &s->frameRateDuration,
                                &s->frameRateScale);
                if (displayMode == bmdModeUnknown) {
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Could not find suitable video mode.\n");
                        goto error;
                }

                if (s->emit_timecode) {
                        outputFlags = (BMDVideoOutputFlags) (outputFlags | bmdVideoOutputRP188);
                }

                if (s->stereo) {
                        outputFlags = (BMDVideoOutputFlags) (outputFlags | bmdVideoOutputDualStream3D);
                        supportedFlags = (BMDSupportedVideoModeFlags) (supportedFlags | bmdSupportedVideoModeDualStream3D);
                }

                BMD_BOOL subsampling_444 = codec_is_a_rgb(desc.color_spec); // we don't have pixfmt for 444 YCbCr
                CALL_AND_CHECK(s->state.at(i).deckLinkConfiguration->SetFlag(bmdDeckLinkConfig444SDIVideoOutput, subsampling_444),
                                "SDI subsampling");

                uint32_t link = s->link_req;

                if (s->link_req == BMD_OPT_DEFAULT) {
                        if (desc.width != 7680) {
                                link = bmdLinkConfigurationSingleLink;
                                LOG(LOG_LEVEL_NOTICE) << MOD_NAME "Setting single link by default.\n";
                        } else {
                                link = bmdLinkConfigurationQuadLink;
                                LOG(LOG_LEVEL_NOTICE) << MOD_NAME "Setting quad-link for 8K by default.\n";
                        }
                }
                CALL_AND_CHECK(s->state.at(i).deckLinkConfiguration->SetInt(bmdDeckLinkConfigSDIOutputLinkConfiguration, link), "Unable set output SDI link mode");

                if (s->profile_req == BMD_OPT_DEFAULT && link == bmdLinkConfigurationQuadLink) {
                        LOG(LOG_LEVEL_WARNING) << MOD_NAME "Quad-link detected - setting 1-subdevice-1/2-duplex profile automatically, use 'profile=keep' to override.\n";
                        decklink_set_duplex(s->state.at(i).deckLink, bmdProfileOneSubDeviceHalfDuplex);
                } else if (link == bmdLinkConfigurationQuadLink && (s->profile_req != BMD_OPT_KEEP && s->profile_req == bmdProfileOneSubDeviceHalfDuplex)) {
                        LOG(LOG_LEVEL_WARNING) << MOD_NAME "Setting quad-link and an incompatible device profile may not be supported!\n";
                }

                BMD_BOOL quad_link_supp;
                if (s->state.at(i).deckLinkAttributes != nullptr && s->state.at(i).deckLinkAttributes->GetFlag(BMDDeckLinkSupportsQuadLinkSDI, &quad_link_supp) == S_OK && quad_link_supp == BMD_TRUE) {
                        CALL_AND_CHECK(s->state.at(i).deckLinkConfiguration->SetFlag(bmdDeckLinkConfigQuadLinkSDIVideoOutputSquareDivisionSplit, s->quad_square_division_split),
                                        "Quad-link SDI Square Division Quad Split mode");
                }

                EXIT_IF_FAILED(s->state.at(i).deckLinkOutput->DoesSupportVideoMode(bmdVideoConnectionUnspecified, displayMode, s->pixelFormat,
                                        IF_NOT_NULL_ELSE(s->conversion_mode, static_cast<BMDVideoOutputConversionMode>(bmdNoVideoOutputConversion)), supportedFlags, nullptr, &supported), "DoesSupportVideoMode");
                if (!supported) {
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Requested parameters "
                                        "combination not supported - %d * %dx%d@%f, timecode %s.\n",
                                        desc.tile_count, desc.width, desc.height, desc.fps,
                                        (outputFlags & bmdVideoOutputRP188 ? "ON": "OFF"));
                        goto error;
                }

                result = s->state.at(i).deckLinkOutput->EnableVideoOutput(displayMode, outputFlags);
                if (FAILED(result)) {
                        if (result == E_ACCESSDENIED) {
                                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unable to access the hardware or output "
                                                "stream currently active (another application using it?).\n");
                        } else {
                                LOG(LOG_LEVEL_ERROR) << MOD_NAME << "EnableVideoOutput: " << bmd_hresult_to_string(result) << "\n";\
                        }
                        goto error;
                }
        }

        // This workaround is needed (at least) for Decklink Extreme 4K when capturing
        // (possibly from another process) and when playback is in low-latency mode.
        // When video is enabled after audio, audio playback becomes silent without
        // an error.
        if (s->initialized_audio) {
                EXIT_IF_FAILED(s->state[0].deckLinkOutput->DisableAudioOutput(), "DisableAudioOutput");
                EXIT_IF_FAILED(s->state[0].deckLinkOutput->EnableAudioOutput(bmdAudioSampleRate48kHz,
                                        s->aud_desc.bps == 2 ? bmdAudioSampleType16bitInteger : bmdAudioSampleType32bitInteger,
                                        s->aud_desc.ch_count,
                                        bmdAudioOutputStreamContinuous),
                                "EnableAudioOutput");
        }

        if (!s->low_latency) {
                for(int i = 0; i < s->devices_cnt; ++i) {
                        EXIT_IF_FAILED(s->state.at(i).deckLinkOutput->StartScheduledPlayback(0, s->frameRateScale, (double) s->frameRateDuration), "StartScheduledPlayback (video)");
                }
        }

        s->initialized_video = true;
        return TRUE;

error:
        // in case we are partially initialized, deinitialize
        for (int i = 0; i < s->devices_cnt; ++i) {
                if (!s->low_latency) {
                        s->state.at(i).deckLinkOutput->StopScheduledPlayback (0, nullptr, 0);
                }
                s->state.at(i).deckLinkOutput->DisableVideoOutput();
        }
        s->initialized_video = false;
        return FALSE;
}

static void display_decklink_probe(struct device_info **available_cards, int *count, void (**deleter)(void *))
{
        UNUSED(deleter);
        IDeckLinkIterator*              deckLinkIterator;
        IDeckLink*                      deckLink;

        *count = 0;
        *available_cards = nullptr;

        deckLinkIterator = create_decklink_iterator(false);
        if (deckLinkIterator == NULL) {
                return;
        }

        // Enumerate all cards in this system
        while (deckLinkIterator->Next(&deckLink) == S_OK)
        {
                string deviceName = bmd_get_device_name(deckLink);
		if (deviceName.empty()) {
			deviceName = "(unknown)";
		}

                *count += 1;
                *available_cards = (struct device_info *)
                        realloc(*available_cards, *count * sizeof(struct device_info));
                memset(*available_cards + *count - 1, 0, sizeof(struct device_info));
                sprintf((*available_cards)[*count - 1].dev, ":device=%d", *count - 1);
                sprintf((*available_cards)[*count - 1].extra, "\"embeddedAudioAvailable\":\"t\"");
                (*available_cards)[*count - 1].repeatable = false;

		strncpy((*available_cards)[*count - 1].name, deviceName.c_str(),
				sizeof (*available_cards)[*count - 1].name - 1);

                // Release the IDeckLink instance when we've finished with it to prevent leaks
                deckLink->Release();
        }

        deckLinkIterator->Release();
        decklink_uninitialize();
}


/**
 * @brief A helper function for parsing unsigned integers out of the command line parameters. This will not allow negative numbers
 *        or numbers that are longer than 9 digits long (this stops undefined behaviour occuring). Any error case should apply a
 *        default value.
 * 
 * @param value_str    The string that is being parsed.
 * @param value_name   The name of the parameter that is being parsed
 * @param value          A pointer to a uint32 to write the parsed value into
 * @param default_value  The default value that should be applied in any of the error cases.
 */
static void parse_uint32(const char *value_str, const char *value_name, uint32_t *value, uint32_t default_value) {
        int value_len = strlen(value_str);
        if(value_len == 0) {
                LOG(LOG_LEVEL_ERROR) << MOD_NAME << "Empty string for option - " << value_name << " - Setting to default " << default_value << "\n";
                *value = default_value;
                return;
        }
        else if(value_len > 9) {
                LOG(LOG_LEVEL_ERROR) << MOD_NAME << "Inputted string too large - " << value_name << " - Setting to default " << default_value << "\n";
                *value = default_value;
                return;
        }
        int tmp_value = atoi(value_str);
        if(tmp_value < 1) {
                LOG(LOG_LEVEL_ERROR) << MOD_NAME << "Inputted resample string negative number or not a valid number - " << value_name << " - Setting to default " << default_value << "\n";
                *value = default_value;
        }
        else {
                *value = tmp_value;
        }
}

static auto parse_devices(const char *devices_str, vector<string> *cardId) {
        if (strlen(devices_str) == 0) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Empty device string!\n");
                return false;
        }
        char *save_ptr;
        char *tmp = strdup(devices_str);
        char *ptr = tmp;
        char *item;
        while ((item = strtok_r(ptr, ",", &save_ptr))) {
                cardId->push_back(item);
                ptr = NULL;
        }
        free(tmp);

        return true;
}

static bool settings_init(struct state_decklink *s, const char *fmt,
                vector<string> *cardId,
                BMDVideo3DPackingFormat *HDMI3DPacking,
                int *audio_consumer_levels,
                int *use1080psf) {
        if (strlen(fmt) == 0) {
                LOG(LOG_LEVEL_WARNING) << MOD_NAME "Card number unset, using first found (see -d decklink:help)!\n";
                return true;
        }

        auto tmp = static_cast<char *>(alloca(strlen(fmt) + 1));
        strcpy(tmp, fmt);
        char *ptr;
        char *save_ptr = nullptr;

        ptr = strtok_r(tmp, ":", &save_ptr);
        assert(ptr != nullptr);
        int i = 0;
        bool first_option_is_device = true;
        while (ptr[i] != '\0') {
                if (!isdigit(ptr[i]) && ptr[i] != ',') {
                        first_option_is_device = false;
                        break;
                }
                i++;
        }
        if (first_option_is_device) {
                log_msg(LOG_LEVEL_WARNING, MOD_NAME "Unnamed device index "
                                "deprecated. Use \"device=%s\" instead.\n", ptr);
                if (!parse_devices(ptr, cardId)) {
                        return false;
                }
                ptr = strtok_r(nullptr, ":", &save_ptr);
        }

        while (ptr != nullptr)  {
                if (strncasecmp(ptr, "device=", strlen("device=")) == 0) {
                        if (!parse_devices(ptr + strlen("device="), cardId)) {
                                return false;
                        }

                } else if (strcasecmp(ptr, "3D") == 0) {
                        s->stereo = true;
                } else if (strcasecmp(ptr, "timecode") == 0) {
                        s->emit_timecode = true;
                } else if (strcasecmp(ptr, "single-link") == 0) {
                        s->link_req = bmdLinkConfigurationSingleLink;
                } else if (strcasecmp(ptr, "dual-link") == 0) {
                        s->link_req = bmdLinkConfigurationDualLink;
                } else if (strcasecmp(ptr, "quad-link") == 0) {
                        s->link_req = bmdLinkConfigurationQuadLink;
                } else if (strstr(ptr, "profile=") == ptr) {
                        ptr += strlen("profile=");
                        if (strcmp(ptr, "keep") == 0) {
                                s->profile_req = BMD_OPT_KEEP;
                        } else {
                                s->profile_req = (BMDProfileID) bmd_read_fourcc(ptr);
                        }
                } else if (strcasecmp(ptr, "half-duplex") == 0) {
                        s->profile_req = bmdDuplexHalf;
                } else if (strcasecmp(ptr, "LevelA") == 0) {
                        s->sdi_dual_channel_level = 'A';
                } else if (strcasecmp(ptr, "LevelB") == 0) {
                        s->sdi_dual_channel_level = 'B';
                } else if (strncasecmp(ptr, "HDMI3DPacking=", strlen("HDMI3DPacking=")) == 0) {
                        char *packing = ptr + strlen("HDMI3DPacking=");
                        if (strcasecmp(packing, "SideBySideHalf") == 0) {
                                *HDMI3DPacking = bmdVideo3DPackingSidebySideHalf;
                        } else if (strcasecmp(packing, "LineByLine") == 0) {
                                *HDMI3DPacking = bmdVideo3DPackingLinebyLine;
                        } else if (strcasecmp(packing, "TopAndBottom") == 0) {
                                *HDMI3DPacking = bmdVideo3DPackingTopAndBottom;
                        } else if (strcasecmp(packing, "FramePacking") == 0) {
                                *HDMI3DPacking = bmdVideo3DPackingFramePacking;
                        } else if (strcasecmp(packing, "LeftOnly") == 0) {
                                *HDMI3DPacking = bmdVideo3DPackingRightOnly;
                        } else if (strcasecmp(packing, "RightOnly") == 0) {
                                *HDMI3DPacking = bmdVideo3DPackingLeftOnly;
                        } else {
                                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unknown HDMI 3D packing %s.\n", packing);
                                return false;
                        }
                } else if (strncasecmp(ptr, "audio_level=", strlen("audio_level=")) == 0) {
                        if (strcasecmp(ptr + strlen("audio_level="), "false") == 0) {
                                *audio_consumer_levels = 0;
                        } else {
                                *audio_consumer_levels = 1;
                        }
                } else if (strncasecmp(ptr, "conversion=",
                                        strlen("conversion=")) == 0) {
                        s->conversion_mode = (BMDVideoOutputConversionMode) bmd_read_fourcc(ptr + strlen("conversion="));
                } else if (is_prefix_of(ptr, "Use1080pNotPsF") || is_prefix_of(ptr, "Use1080PsF")) {
                        if ((*use1080psf = parse_bmd_flag(strchr(ptr, '=') + 1)) == -1) {
                                return false;
                        }
                        if (strncasecmp(ptr, "Use1080pNotPsF", strlen("Use1080pNotPsF")) == 0) { // compat, inverse
                                *use1080psf = invert_bmd_flag(*use1080psf);
                        }
                } else if (strcasecmp(ptr, "low-latency") == 0 || strcasecmp(ptr, "no-low-latency") == 0) {
                        s->low_latency = strcasecmp(ptr, "low-latency") == 0;
                } else if (strcasecmp(ptr, "quad-square") == 0 || strcasecmp(ptr, "no-quad-square") == 0) {
                        s->quad_square_division_split = strcasecmp(ptr, "quad-square") == 0;
                } else if (strncasecmp(ptr, "hdr", strlen("hdr")) == 0) {
                        s->requested_hdr_mode.EOTF = static_cast<int64_t>(HDR_EOTF::HDR); // default
                        if (strncasecmp(ptr, "hdr=", strlen("hdr=")) == 0) {
                                try {
                                        s->requested_hdr_mode.Init(ptr + strlen("hdr="));
                                } catch (ug_no_error const &e) {
                                        return false;
                                } catch (exception const &e) {
                                        LOG(LOG_LEVEL_ERROR) << MOD_NAME << "HDR mode init: " << e.what() << "\n";
                                        return false;
                                }
                        }
                } else if (strstr(ptr, "drift_fix") == ptr) {
                        s->audio_drift_fixer.m_enabled = true;
                }
                else if (strncasecmp(ptr, "maxresample=", strlen("maxresample=")) == 0) {
                        uint32_t max_resample_delta = 0;
                        parse_uint32(ptr + strlen("maxresample="), "maxresample", &max_resample_delta, MAX_RESAMPLE_DELTA_DEFAULT);
                        s->audio_drift_fixer.set_max_hz(max_resample_delta);
                        LOG(LOG_LEVEL_INFO) << MOD_NAME << "Set Max Resample Delta to be " << max_resample_delta << "Hz\n";
                } 
                else if (strncasecmp(ptr, "minresample=", strlen("minresample=")) == 0) {
                        uint32_t min_resample_delta = 0;
                        parse_uint32(ptr + strlen("minresample="), "minresample", &min_resample_delta, MIN_RESAMPLE_DELTA_DEFAULT);
                        s->audio_drift_fixer.set_min_hz(min_resample_delta);
                        LOG(LOG_LEVEL_INFO) << MOD_NAME << "Set Min Resample Delta to be " << min_resample_delta << "Hz\n";
                }else if (strncasecmp(ptr, "targetbuffer=", strlen("targetbuffer=")) == 0) {
                        uint32_t target_buffer = 0;
                        parse_uint32(ptr + strlen("targetbuffer="), "targetbuffer",&target_buffer, TARGET_BUFFER_DEFAULT);
                        s->audio_drift_fixer.set_target_buffer(target_buffer);
                        LOG(LOG_LEVEL_INFO) << MOD_NAME << "Set Target Buffer to be " << target_buffer << " samples in buffer (per channel)\n";
                } else {
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Warning: unknown options in config string.\n");
                        return false;
                }
                ptr = strtok_r(nullptr, ":", &save_ptr);
        }

        return true;
}

#define BMD_CONFIG_SET_INT(val, type, key, name) do {\
        if (val != (type) BMD_OPT_DEFAULT) {\
                HRESULT result = deckLinkConfiguration->SetInt(key, val);\
                if (result != S_OK) {\
                        log_msg(LOG_LEVEL_ERROR, "Unable to set " name "\n");\
                        goto error;\
                }\
        }\
} while (0)

static void *display_decklink_init(struct module *parent, const char *fmt, unsigned int flags)
{
        IDeckLinkIterator*                              deckLinkIterator;
        HRESULT                                         result;
        vector<string>                                  cardId;
        int                                             dnum = 0;
        IDeckLinkConfiguration*         deckLinkConfiguration = NULL;
        // for Decklink Studio which has switchable XLR - analog 3 and 4 or AES/EBU 3,4 and 5,6
        BMDAudioOutputAnalogAESSwitch audioConnection = (BMDAudioOutputAnalogAESSwitch) 0;
        BMDVideo3DPackingFormat HDMI3DPacking = (BMDVideo3DPackingFormat) 0;
        int audio_consumer_levels = -1;
        int use1080psf = BMD_OPT_DEFAULT;

        if (strcmp(fmt, "help") == 0 || strcmp(fmt, "fullhelp") == 0) {
                show_help(strcmp(fmt, "fullhelp") == 0);
                return &display_init_noerr;
        }

        if (!blackmagic_api_version_check()) {
                return NULL;
        }

        struct state_decklink *s = new state_decklink();
        s->magic = DECKLINK_MAGIC;
        s->audio_drift_fixer.set_root(get_root_module(parent));
        s->stereo = FALSE;
        s->emit_timecode = false;
        s->profile_req = BMD_OPT_DEFAULT;
        s->link_req = 0;
        s->devices_cnt = 1;
        s->low_latency = true;
        s->last_buffered_samples = 0;
        s->drift_since_last_correction = 0;
        s->audio_drift_fixer.set_summary(&(s->audio_summary));

        if (!settings_init(s, fmt, &cardId, &HDMI3DPacking, &audio_consumer_levels, &use1080psf)) {
                delete s;
                return NULL;
        }

        if (cardId.empty()) {
                cardId.emplace_back("0");
        }
        s->devices_cnt = cardId.size();

	if (s->stereo && s->devices_cnt > 1) {
                LOG(LOG_LEVEL_ERROR) << MOD_NAME "Unsupported configuration - in stereo "
                        "mode, exactly one device index must be given.\n";
                delete s;
                return NULL;
        }

        if (s->low_latency) {
                LOG(LOG_LEVEL_NOTICE) << MOD_NAME "Using low-latency mode. "
                        "In case of problems, you can try '-d decklink:no-low-latency'.\n";
        }

        // Initialize the DeckLink API
        deckLinkIterator = create_decklink_iterator(true);
        if (!deckLinkIterator)
        {
                delete s;
                return NULL;
        }

        s->state.resize(s->devices_cnt);

        // Connect to the first DeckLink instance
        IDeckLink    *deckLink;
        while (deckLinkIterator->Next(&deckLink) == S_OK)
        {
                bool found = false;
                for(int i = 0; i < s->devices_cnt; ++i) {
                        string deviceName = bmd_get_device_name(deckLink);

			if (!deviceName.empty() && deviceName == cardId[i]) {
				found = true;
			}

                        if (isdigit(cardId[i].c_str()[0]) && dnum == atoi(cardId[i].c_str())){
                                found = true;
                        }

                        if (found) {
                                s->state.at(i).deckLink = deckLink;
                        }
                }
                if(!found && deckLink != NULL)
                        deckLink->Release();
                dnum++;
        }
        deckLinkIterator->Release();
        for(int i = 0; i < s->devices_cnt; ++i) {
                if (s->state.at(i).deckLink == nullptr) {
                        LOG(LOG_LEVEL_ERROR) << "No DeckLink PCI card " << cardId[i] <<" found\n";
                        goto error;
                }
                // Print the model name of the DeckLink card
                string deviceName = bmd_get_device_name(s->state.at(i).deckLink);
                if (!deviceName.empty()) {
                        LOG(LOG_LEVEL_INFO) << MOD_NAME "Using device " << deviceName << "\n";
                }
        }

        if(flags & (DISPLAY_FLAG_AUDIO_EMBEDDED | DISPLAY_FLAG_AUDIO_AESEBU | DISPLAY_FLAG_AUDIO_ANALOG)) {
                s->play_audio = true;
                switch(flags & (DISPLAY_FLAG_AUDIO_EMBEDDED | DISPLAY_FLAG_AUDIO_AESEBU | DISPLAY_FLAG_AUDIO_ANALOG)) {
                        case DISPLAY_FLAG_AUDIO_EMBEDDED:
                                audioConnection = (BMDAudioOutputAnalogAESSwitch) 0;
                                break;
                        case DISPLAY_FLAG_AUDIO_AESEBU:
                                audioConnection = bmdAudioOutputSwitchAESEBU;
                                break;
                        case DISPLAY_FLAG_AUDIO_ANALOG:
                                audioConnection = bmdAudioOutputSwitchAnalog;
                                break;
                        default:
                                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unsupporetd audio connection.\n");
                                abort();
                }
        } else {
                s->play_audio = false;
        }
        
        if(s->emit_timecode) {
                s->timecode = new DeckLinkTimecode;
        } else {
                s->timecode = NULL;
        }
        
        for(int i = 0; i < s->devices_cnt; ++i) {
                if (s->profile_req != BMD_OPT_DEFAULT && s->profile_req != BMD_OPT_KEEP) {
                        decklink_set_duplex(s->state.at(i).deckLink, s->profile_req);
                }

		// Get IDeckLinkAttributes object
		IDeckLinkProfileAttributes *deckLinkAttributes = NULL;
		result = s->state.at(i).deckLink->QueryInterface(IID_IDeckLinkProfileAttributes, reinterpret_cast<void**>(&deckLinkAttributes));
		if (result != S_OK) {
			log_msg(LOG_LEVEL_WARNING, "Could not query device attributes.\n");
		}
                s->state.at(i).deckLinkAttributes = deckLinkAttributes;

                // Obtain the audio/video output interface (IDeckLinkOutput)
                if ((result = s->state.at(i).deckLink->QueryInterface(IID_IDeckLinkOutput, reinterpret_cast<void**>(&s->state.at(i).deckLinkOutput))) != S_OK) {
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Could not obtain the IDeckLinkOutput interface: %08x\n", (int) result);
                        goto error;
                }

                // Query the DeckLink for its configuration interface
                result = s->state.at(i).deckLink->QueryInterface(IID_IDeckLinkConfiguration, reinterpret_cast<void**>(&deckLinkConfiguration));
                s->state.at(i).deckLinkConfiguration = deckLinkConfiguration;
                if (result != S_OK)
                {
                        log_msg(LOG_LEVEL_ERROR, "Could not obtain the IDeckLinkConfiguration interface: %08x\n", (int) result);
                        goto error;
                }

                BMD_CONFIG_SET_INT(s->conversion_mode, BMDVideoOutputConversionMode, bmdDeckLinkConfigVideoOutputConversionMode, "conversion mode");

		if (use1080psf != BMD_OPT_KEEP) {
                        if (use1080psf == BMD_OPT_DEFAULT) {
                                LOG(LOG_LEVEL_WARNING) << MOD_NAME << "Setting output signal as progressive, see option \"Use1080PsF\" to use PsF or keep default.\n";
                        }
                        BMD_BOOL val = use1080psf == BMD_OPT_DEFAULT || use1080psf == BMD_OPT_FALSE ? BMD_FALSE : BMD_TRUE;
                        result = deckLinkConfiguration->SetFlag(bmdDeckLinkConfigOutput1080pAsPsF, val);
                        if (result != S_OK) {
                                LOG(LOG_LEVEL_ERROR) << MOD_NAME << "Unable to set 1080p P/PsF mode.\n";
                        }
                }

                result = deckLinkConfiguration->SetFlag(bmdDeckLinkConfigLowLatencyVideoOutput, s->low_latency);
                if (result != S_OK) {
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unable to set to low-latency mode.\n");
                        goto error;
                }

                if (s->low_latency) {
                        result = deckLinkConfiguration->SetFlag(bmdDeckLinkConfigFieldFlickerRemoval, false);
                        if (result != S_OK) {
                                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unable to set field flicker removal.\n");
                                goto error;
                        }
                }

                BMD_CONFIG_SET_INT(HDMI3DPacking, BMDVideo3DPackingFormat, bmdDeckLinkConfigHDMI3DPackingFormat, "3D packing");

                if (s->sdi_dual_channel_level != BMD_OPT_DEFAULT) {
#if BLACKMAGIC_DECKLINK_API_VERSION < ((10 << 24) | (8 << 16))
                        log_msg(LOG_LEVEL_WARNING, MOD_NAME "Compiled with old SDK - cannot set 3G-SDI level.\n");
#else
			if (deckLinkAttributes) {
				BMD_BOOL supports_level_a;
				if (deckLinkAttributes->GetFlag(BMDDeckLinkSupportsSMPTELevelAOutput, &supports_level_a) != S_OK) {
					log_msg(LOG_LEVEL_WARNING, MOD_NAME "Could figure out if device supports Level A 3G-SDI.\n");
				} else {
					if (s->sdi_dual_channel_level == 'A' && supports_level_a == BMD_FALSE) {
						log_msg(LOG_LEVEL_WARNING, MOD_NAME "Device does not support Level A 3G-SDI!\n");
					}
				}
			}
                        HRESULT res = deckLinkConfiguration->SetFlag(bmdDeckLinkConfigSMPTELevelAOutput, s->sdi_dual_channel_level == 'A');
                        if(res != S_OK) {
                                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unable set output 3G-SDI level.\n");
                        }
#endif
                }

                if (s->requested_hdr_mode.EOTF != static_cast<int64_t>(HDR_EOTF::NONE)) {
                        BMD_BOOL hdr_supp = BMD_FALSE;
                        if (s->state.at(i).deckLinkAttributes == nullptr || s->state.at(i).deckLinkAttributes->GetFlag(BMDDeckLinkSupportsHDRMetadata, &hdr_supp) != S_OK) {
                                LOG(LOG_LEVEL_WARNING) << MOD_NAME << "HDR requested, but unable to validate HDR support. Will try to pass it anyway which may result in blank image if not supported - remove the option if so.\n";
                        } else {
                                if (hdr_supp != BMD_TRUE) {
                                        LOG(LOG_LEVEL_ERROR) << MOD_NAME << "HDR requested, but card doesn't support that.\n";
                                        goto error;
                                }
                        }

                        BMD_BOOL rec2020_supp = BMD_FALSE;
                        if (s->state.at(i).deckLinkAttributes == nullptr || s->state.at(i).deckLinkAttributes->GetFlag(BMDDeckLinkSupportsColorspaceMetadata, &rec2020_supp) != S_OK) {
                                LOG(LOG_LEVEL_WARNING) << MOD_NAME << "Cannot check Rec. 2020 color space metadata support.\n";
                        } else {
                                if (rec2020_supp != BMD_TRUE) {
                                        LOG(LOG_LEVEL_WARNING) << MOD_NAME << "Rec. 2020 color space metadata not supported.\n";
                                }
                        }
                }

                if (s->play_audio && i == 0) {
                        /* Actually no action is required to set audio connection because Blackmagic card plays audio through all its outputs (AES/SDI/analog) ....
                         */
                        printf(MOD_NAME "Audio output set to: ");
                        switch(audioConnection) {
                                case bmdAudioOutputSwitchAESEBU:
                                        printf("AES/EBU");
                                        break;
                                case bmdAudioOutputSwitchAnalog:
                                        printf("analog");
                                        break;
                                default:
                                        printf("default");
                                        break;
                        }
                        printf(".\n");
                         /*
                          * .... one exception is a card that has switchable cables between AES/EBU and analog. (But this applies only for channels 3 and above.)
                         */
                        if (audioConnection != 0) { // we will set switchable AESEBU or analog
                                result = deckLinkConfiguration->SetInt(bmdDeckLinkConfigAudioOutputAESAnalogSwitch,
                                                audioConnection);
                                if(result == S_OK) { // has switchable channels
                                        log_msg(LOG_LEVEL_INFO, MOD_NAME "Card with switchable audio channels detected. Switched to correct format.\n");
                                } else if(result == E_NOTIMPL) {
                                        // normal case - without switchable channels
                                } else {
                                        log_msg(LOG_LEVEL_WARNING, MOD_NAME "Unable to switch audio output for channels 3 or above although \n"
                                                        "card shall support it. Check if it is ok. Continuing anyway.\n");
                                }
                        }

                        if (audio_consumer_levels != -1) {
                                result = deckLinkConfiguration->SetFlag(bmdDeckLinkConfigAnalogAudioConsumerLevels,
                                                audio_consumer_levels == 1 ? true : false);
                                if(result != S_OK) {
                                        log_msg(LOG_LEVEL_WARNING, MOD_NAME "Unable set output audio consumer levels.\n");
                                }
                        }
                }

                s->state.at(i).delegate = new PlaybackDelegate();
                // Provide this class as a delegate to the audio and video output interfaces
                if (!s->low_latency) {
                        s->state.at(i).deckLinkOutput->SetScheduledFrameCompletionCallback(s->state.at(i).delegate);
                }
                //s->state.at(i).deckLinkOutput->DisableAudioOutput();
        }

        s->frames = 0;
        s->initialized_audio = s->initialized_video = false;

        return (void *)s;
error:
        display_decklink_done(s);
        return NULL;
}

static void display_decklink_run(void *state)
{
        UNUSED(state);
}

#define RELEASE_IF_NOT_NULL(x) if (x != nullptr) x->Release();
static void display_decklink_done(void *state)
{
        debug_msg("display_decklink_done\n"); /* TOREMOVE */
        struct state_decklink *s = (struct state_decklink *)state;

        assert (s != NULL);

        for (int i = 0; i < s->devices_cnt; ++i)
        {
                if (s->initialized_video) {
                        if (!s->low_latency) {
                                CALL_AND_CHECK(s->state.at(i).deckLinkOutput->StopScheduledPlayback (0, nullptr, 0), "StopScheduledPlayback");
                        }

                        CALL_AND_CHECK(s->state.at(i).deckLinkOutput->DisableVideoOutput(), "DisableVideoOutput");
                }

                if (s->initialized_audio) {
                        if (i == 0) {
                                CALL_AND_CHECK(s->state.at(i).deckLinkOutput->DisableAudioOutput(), "DisableAudiioOutput");
                        }
                }

                RELEASE_IF_NOT_NULL(s->state.at(i).deckLinkAttributes);
                RELEASE_IF_NOT_NULL(s->state.at(i).deckLinkConfiguration);
                RELEASE_IF_NOT_NULL(s->state.at(i).deckLinkOutput);
                RELEASE_IF_NOT_NULL(s->state.at(i).deckLink);

                delete s->state.at(i).delegate;
        }

        while (!s->buffer_pool.frame_queue.empty()) {
                auto tmp = s->buffer_pool.frame_queue.front();
                s->buffer_pool.frame_queue.pop();
                delete tmp;
        }
        delete s->timecode;

        delete s;

        decklink_uninitialize();
}

/**
 * This function returns true if any display mode and any output supports the
 * codec. The codec, however, may not be supported with actual video mode.
 *
 * @todo For UltraStudio Pro DoesSupportVideoMode returns E_FAIL on not supported
 * pixel formats instead of setting supported to false.
 */
static bool decklink_display_supports_codec(IDeckLinkOutput *deckLinkOutput, BMDPixelFormat pf) {
        IDeckLinkDisplayModeIterator     *displayModeIterator;
        IDeckLinkDisplayMode*             deckLinkDisplayMode;

        if (FAILED(deckLinkOutput->GetDisplayModeIterator(&displayModeIterator))) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Fatal: cannot create display mode iterator.\n");
                return false;
        }

        while (displayModeIterator->Next(&deckLinkDisplayMode) == S_OK) {
                BMD_BOOL supported;
                HRESULT res = deckLinkOutput->DoesSupportVideoMode(bmdVideoConnectionUnspecified, deckLinkDisplayMode->GetDisplayMode(), pf, bmdNoVideoOutputConversion, bmdSupportedVideoModeDefault, nullptr, &supported);
                deckLinkDisplayMode->Release();
                if (res != S_OK) {
                        CALL_AND_CHECK(res, "DoesSupportVideoMode");
                        continue;
                }
                if (supported) {
                        displayModeIterator->Release();
                        return true;
                }
        }
        displayModeIterator->Release();

        return false;
}

static int display_decklink_get_property(void *state, int property, void *val, size_t *len)
{
        struct state_decklink *s = (struct state_decklink *)state;
        vector<codec_t> codecs(uv_to_bmd_codec_map.size());
        int rgb_shift[] = {16, 8, 0};
        interlacing_t supported_il_modes[] = {PROGRESSIVE, INTERLACED_MERGED, SEGMENTED_FRAME};
        int count = 0;
        for (auto & c : uv_to_bmd_codec_map) {
                if (decklink_display_supports_codec(s->state[0].deckLinkOutput, c.second)) {
                        codecs[count++] = c.first;
                }
        }
        
        switch (property) {
                case DISPLAY_PROPERTY_CODECS:
                        if(sizeof(codec_t) * count <= *len) {
                                memcpy(val, codecs.data(), sizeof(codec_t) * count);
                                *len = sizeof(codec_t) * count;
                        } else {
                                return FALSE;
                        }
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
                case DISPLAY_PROPERTY_VIDEO_MODE:
                        if(s->devices_cnt == 1 && !s->stereo)
                                *(int *) val = DISPLAY_PROPERTY_VIDEO_MERGED;
                        else
                                *(int *) val = DISPLAY_PROPERTY_VIDEO_SEPARATE_TILES;
                        break;
                case DISPLAY_PROPERTY_SUPPORTED_IL_MODES:
                        if(sizeof(supported_il_modes) <= *len) {
                                memcpy(val, supported_il_modes, sizeof(supported_il_modes));
                        } else {
                                return FALSE;
                        }
                        *len = sizeof(supported_il_modes);
                        break;
                case DISPLAY_PROPERTY_AUDIO_FORMAT:
                        {
                                assert(*len == sizeof(struct audio_desc));
                                struct audio_desc *desc = (struct audio_desc *) val;
                                desc->sample_rate = 48000;
                                if (desc->ch_count <= 2) {
                                        desc->ch_count = 2;
                                } else if (desc->ch_count > 2 && desc->ch_count <= 8) {
                                        desc->ch_count = 8;
                                } else {
                                        desc->ch_count = 16;
                                }
                                desc->codec = AC_PCM;
                                desc->bps = desc->bps < 3 ? 2 : 4;
                        }
                        break;
                default:
                        return FALSE;
        }
        return TRUE;
}

/*
 * AUDIO
 */
static void display_decklink_put_audio_frame(void *state, struct audio_frame *frame)
{
        struct state_decklink *s = (struct state_decklink *)state;
        unsigned int sample_frame_count = frame->data_len / (frame->bps * frame->ch_count);

        s->audio_summary.record_audio_time_diff();
        
        assert(s->play_audio);

        uint32_t sampleFramesWritten;

        auto t0 = chrono::high_resolution_clock::now();
        
        uint32_t buffered = 0;
        s->state[0].deckLinkOutput->GetBufferedAudioSampleFrameCount(&buffered);
        s->audio_summary.calculate_missing(buffered, sample_frame_count);
        if (buffered == 0) {
                LOG(LOG_LEVEL_WARNING) << MOD_NAME << "audio buffer underflow!\n";
                s->audio_summary.increment_buffer_underflow();
        }
        
        if (!s->audio_drift_fixer.update(buffered)) {
                log_msg(LOG_LEVEL_WARNING, MOD_NAME "update drift early exit.\n");
                return;
        }

        if (s->low_latency) {
                HRESULT res = s->state[0].deckLinkOutput->WriteAudioSamplesSync(frame->data, sample_frame_count,
                                &sampleFramesWritten);
                if (FAILED(res)) {
                        log_msg(LOG_LEVEL_WARNING, MOD_NAME "WriteAudioSamplesSync failed.\n");
                        return;
                }
        } else {
                s->state[0].deckLinkOutput->ScheduleAudioSamples(frame->data, sample_frame_count, 0,
                                0, &sampleFramesWritten);
        }
        if (sampleFramesWritten != sample_frame_count) {
                LOG(LOG_LEVEL_WARNING) << MOD_NAME << "audio buffer overflow! " << sample_frame_count
                                                << " samples written, " << sampleFramesWritten << " written, " 
                                                << sample_frame_count - sampleFramesWritten<<" diff, "<<buffered<< " buffer size.\n";
                s->audio_summary.increment_buffer_overflow();
        }
        LOG(LOG_LEVEL_VERBOSE) << MOD_NAME "putf audio - lasted " << setprecision(2) << chrono::duration_cast<chrono::duration<double>>(chrono::high_resolution_clock::now() - t0).count() * 1000.0 << " ms.\n";
        s->audio_summary.increment_audio_frames_played();
        s->audio_summary.set_buffer_average(s->audio_drift_fixer.get_buffer_avg());
        s->audio_summary.report();
        s->audio_summary.mark_audio_time_end();
}

static int display_decklink_reconfigure_audio(void *state, int quant_samples, int channels,
                int sample_rate) {
        struct state_decklink *s = (struct state_decklink *)state;
        BMDAudioSampleType sample_type;

        unique_lock<mutex> lk(s->reconfiguration_lock);

        assert(s->play_audio);

        if (s->initialized_audio) {
                CALL_AND_CHECK(s->state[0].deckLinkOutput->DisableAudioOutput(),
                                "DisableAudioOutput");
                s->initialized_audio = false;
        }

        if (channels != 2 && channels != 8 &&
                        channels != 16) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "requested channel count isn't supported: "
                        "%d\n", channels);
                return FALSE;
        }
        
        if((quant_samples != 16 && quant_samples != 32) ||
                        sample_rate != 48000) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "audio format isn't supported: "
                        "samples: %d, sample rate: %d\n",
                        quant_samples, sample_rate);
                return FALSE;
        }
        switch(quant_samples) {
                case 16:
                        sample_type = bmdAudioSampleType16bitInteger;
                        break;
                case 32:
                        sample_type = bmdAudioSampleType32bitInteger;
                        break;
                default:
                        return FALSE;
        }
                        
        EXIT_IF_FAILED(s->state[0].deckLinkOutput->EnableAudioOutput(bmdAudioSampleRate48kHz,
                        sample_type,
                        channels,
                        bmdAudioOutputStreamContinuous),
                "EnableAudioOutput");

        if (!s->low_latency) {
                // This will most certainly fail because it is started with in video
                // reconfigure. However, this doesn't seem to bother, anyway.
                CALL_AND_CHECK(s->state[0].deckLinkOutput->StartScheduledPlayback(0, s->frameRateScale, s->frameRateDuration), "StartScheduledPlayback (audio)");
        }

        s->aud_desc = { quant_samples / 8, sample_rate, channels, AC_PCM };

        s->initialized_audio = true;
        
        return TRUE;

error:
        s->initialized_audio = false;
        return FALSE;
}

#ifndef WIN32
static bool operator==(const REFIID & first, const REFIID & second){
    return (first.byte0 == second.byte0) &&
        (first.byte1 == second.byte1) &&
        (first.byte2 == second.byte2) &&
        (first.byte3 == second.byte3) &&
        (first.byte4 == second.byte4) &&
        (first.byte5 == second.byte5) &&
        (first.byte6 == second.byte6) &&
        (first.byte7 == second.byte7) &&
        (first.byte8 == second.byte8) &&
        (first.byte9 == second.byte9) &&
        (first.byte10 == second.byte10) &&
        (first.byte11 == second.byte11) &&
        (first.byte12 == second.byte12) &&
        (first.byte13 == second.byte13) &&
        (first.byte14 == second.byte14) &&
        (first.byte15 == second.byte15);
}
#endif

HRESULT DeckLinkFrame::QueryInterface(REFIID iid, LPVOID *ppv)
{
#ifdef _WIN32
        IID                     iunknown = IID_IUnknown;
#else
        CFUUIDBytes             iunknown = CFUUIDGetUUIDBytes(IUnknownUUID);
#endif
        HRESULT                 result          = S_OK;

        if (ppv == nullptr) {
                return E_INVALIDARG;
        }

        // Initialise the return result
        *ppv = nullptr;

        LOG(LOG_LEVEL_DEBUG) << MOD_NAME << "DeckLinkFrame QueryInterface " << iid << "\n";
        if (iid == iunknown) {
                *ppv = this;
                AddRef();
        } else if (iid == IID_IDeckLinkVideoFrame) {
                *ppv = static_cast<IDeckLinkVideoFrame*>(this);
                AddRef();
        } else if (iid == IID_IDeckLinkVideoFrameMetadataExtensions) {
                if (m_metadata.EOTF == static_cast<int64_t>(HDR_EOTF::NONE)) {
                        result = E_NOINTERFACE;
                } else {
                        *ppv = static_cast<IDeckLinkVideoFrameMetadataExtensions*>(this);
                        AddRef();
                }
        } else {
                result = E_NOINTERFACE;
        }

        return result;

        return E_NOINTERFACE;
}

ULONG DeckLinkFrame::AddRef()
{
        return ++ref;
}

ULONG DeckLinkFrame::Release()
{
        if (--ref == 0) {
                lock_guard<mutex> lg(buffer_pool.lock);
                buffer_pool.frame_queue.push(this);
        }
	return ref;
}

DeckLinkFrame::DeckLinkFrame(long w, long h, long rb, BMDPixelFormat pf, buffer_pool_t & bp, HDRMetadata const & hdr_metadata)
	: width(w), height(h), rawBytes(rb), pixelFormat(pf), data(new char[rb * h]), timecode(NULL), ref(1l),
        buffer_pool(bp)
{
        clear_video_buffer(reinterpret_cast<unsigned char *>(data.get()), rawBytes, rawBytes, height,
                        pf == bmdFormat8BitYUV ? UYVY : (pf == bmdFormat10BitYUV ? v210 : RGBA));
        m_metadata = hdr_metadata;
}

DeckLinkFrame *DeckLinkFrame::Create(long width, long height, long rawBytes, BMDPixelFormat pixelFormat, buffer_pool_t & buffer_pool, const HDRMetadata & hdr_metadata)
{
        return new DeckLinkFrame(width, height, rawBytes, pixelFormat, buffer_pool, hdr_metadata);
}

DeckLinkFrame::~DeckLinkFrame() 
{
}

long DeckLinkFrame::GetWidth ()
{
        return width;
}

long DeckLinkFrame::GetHeight ()
{
        return height;
}

long DeckLinkFrame::GetRowBytes ()
{
        return rawBytes;
}

BMDPixelFormat DeckLinkFrame::GetPixelFormat ()
{
        return pixelFormat;
}

BMDFrameFlags DeckLinkFrame::GetFlags ()
{
        return m_metadata.EOTF == static_cast<int64_t>(HDR_EOTF::NONE) ? bmdFrameFlagDefault : bmdFrameContainsHDRMetadata;
}

HRESULT DeckLinkFrame::GetBytes (/* out */ void **buffer)
{
        *buffer = static_cast<void *>(data.get());
        return S_OK;
}

HRESULT DeckLinkFrame::GetTimecode (/* in */ BMDTimecodeFormat, /* out */ IDeckLinkTimecode **timecode)
{
        *timecode = dynamic_cast<IDeckLinkTimecode *>(this->timecode);
        return S_OK;
}

HRESULT DeckLinkFrame::GetAncillaryData (/* out */ IDeckLinkVideoFrameAncillary **)
{
	return S_FALSE;
}

/* IDeckLinkMutableVideoFrame */
HRESULT DeckLinkFrame::SetFlags (/* in */ BMDFrameFlags)
{
        return E_FAIL;
}

HRESULT DeckLinkFrame::SetTimecode (/* in */ BMDTimecodeFormat, /* in */ IDeckLinkTimecode *timecode)
{
        if(this->timecode)
                this->timecode->Release();
        this->timecode = timecode;
        return S_OK;
}

HRESULT DeckLinkFrame::SetTimecodeFromComponents (/* in */ BMDTimecodeFormat, /* in */ uint8_t, /* in */ uint8_t, /* in */ uint8_t, /* in */ uint8_t, /* in */ BMDTimecodeFlags)
{
        return E_FAIL;
}

HRESULT DeckLinkFrame::SetAncillaryData (/* in */ IDeckLinkVideoFrameAncillary *)
{
        return E_FAIL;
}

HRESULT DeckLinkFrame::SetTimecodeUserBits (/* in */ BMDTimecodeFormat, /* in */ BMDTimecodeUserBits)
{
        return E_FAIL;
}

void HDRMetadata::Init(const string &fmt) {
        auto opts = unique_ptr<char []>(new char [fmt.size() + 1]);
        strcpy(opts.get(), fmt.c_str());
        char *save_ptr = nullptr;
        char *mode_c = strtok_r(opts.get(), ",", &save_ptr);
        assert(mode_c != nullptr);
        string mode = mode_c;
        std::for_each(std::begin(mode), std::end(mode), [](char& c) {
                        c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
                        });
        if (mode == "SDR"s) {
                EOTF = static_cast<int64_t>(HDR_EOTF::SDR);
        } else if (mode == "HDR"s) {
                EOTF = static_cast<int64_t>(HDR_EOTF::HDR);
        } else if (mode == "PQ"s) {
                EOTF = static_cast<int64_t>(HDR_EOTF::PQ);
        } else if (mode == "HLG"s) {
                EOTF = static_cast<int64_t>(HDR_EOTF::HLG);
        } else if (mode == "HELP"s) {
                cout << MOD_NAME << "HDR syntax:\n";
                cout << "\tHDR[=<eotf>|int[,{<k>=<v>}*]\n";
                cout << "\t\t<eotf> may be one of SDR, HDR, PQ, HLG or int 0-7\n";
                cout << "\t\tFurther options may be specification of HDR values, accepted keys are (values are floats):\n";
                cout << "\t\t\t- maxDisplayMasteringLuminance\n";
                cout << "\t\t\t- minDisplayMasteringLuminance\n";
                cout << "\t\t\t- maxCLL\n";
                cout << "\t\t\t- maxFALL\n";
                throw ug_no_error{};
        } else {
                EOTF = stoi(mode);
                if (EOTF < 0 || EOTF > 7) {
                        throw out_of_range("Value outside [0..7]");
                }
        }

        char *other_opt = nullptr;
        while ((other_opt = strtok_r(nullptr, ",", &save_ptr)) != nullptr) {
                if (strstr(other_opt, "maxDisplayMasteringLuminance=") != nullptr) {
                        maxDisplayMasteringLuminance = stod(other_opt + strlen("maxDisplayMasteringLuminance="));
                } else if (strstr(other_opt, "minDisplayMasteringLuminance=") != nullptr) {
                        minDisplayMasteringLuminance = stod(other_opt + strlen("minDisplayMasteringLuminance="));
                } else if (strstr(other_opt, "maxCLL=") != nullptr) {
                        maxCLL = stod(other_opt + strlen("maxCLL="));
                } else if (strstr(other_opt, "maxFALL=") != nullptr) {
                        maxFALL = stod(other_opt + strlen("maxFALL="));
                } else {
                        throw invalid_argument("Unrecognized HDR attribute "s + other_opt);
                }
        }
}

static inline void debug_print_metadata_id(const char *fn_name, BMDDeckLinkFrameMetadataID metadataID) {
        if (log_level < LOG_LEVEL_DEBUG2) {
                return;
        }
        array<char, sizeof metadataID + 1> fourcc{};
        copy(reinterpret_cast<char *>(&metadataID), reinterpret_cast<char *>(&metadataID) + sizeof metadataID, fourcc.data());
        LOG(LOG_LEVEL_DEBUG2) << MOD_NAME << "DecklLinkFrame " << fn_name << ": " << fourcc.data() << "\n";
}

// IDeckLinkVideoFrameMetadataExtensions interface
HRESULT DeckLinkFrame::GetInt(BMDDeckLinkFrameMetadataID metadataID, int64_t* value)
{
        debug_print_metadata_id(static_cast<const char *>(__func__), metadataID);
        switch (metadataID)
        {
                case bmdDeckLinkFrameMetadataHDRElectroOpticalTransferFunc:
                        *value = m_metadata.EOTF;
                        return S_OK;
                case bmdDeckLinkFrameMetadataColorspace:
                        // Colorspace is fixed for this sample
                        *value = bmdColorspaceRec2020;
                        return S_OK;
                default:
                        value = nullptr;
                        return E_INVALIDARG;
        }
}

HRESULT DeckLinkFrame::GetFloat(BMDDeckLinkFrameMetadataID metadataID, double* value)
{
        debug_print_metadata_id(static_cast<const char *>(__func__), metadataID);
        switch (metadataID)
        {
                case bmdDeckLinkFrameMetadataHDRDisplayPrimariesRedX:
                        *value = m_metadata.referencePrimaries.RedX;
                        return S_OK;
                case bmdDeckLinkFrameMetadataHDRDisplayPrimariesRedY:
                        *value = m_metadata.referencePrimaries.RedY;
                        return S_OK;
                case bmdDeckLinkFrameMetadataHDRDisplayPrimariesGreenX:
                        *value = m_metadata.referencePrimaries.GreenX;
                        return S_OK;
                case bmdDeckLinkFrameMetadataHDRDisplayPrimariesGreenY:
                        *value = m_metadata.referencePrimaries.GreenY;
                        return S_OK;
                case bmdDeckLinkFrameMetadataHDRDisplayPrimariesBlueX:
                        *value = m_metadata.referencePrimaries.BlueX;
                        return S_OK;
                case bmdDeckLinkFrameMetadataHDRDisplayPrimariesBlueY:
                        *value = m_metadata.referencePrimaries.BlueY;
                        return S_OK;
                case bmdDeckLinkFrameMetadataHDRWhitePointX:
                        *value = m_metadata.referencePrimaries.WhiteX;
                        return S_OK;
                case bmdDeckLinkFrameMetadataHDRWhitePointY:
                        *value = m_metadata.referencePrimaries.WhiteY;
                        return S_OK;
                case bmdDeckLinkFrameMetadataHDRMaxDisplayMasteringLuminance:
                        *value = m_metadata.maxDisplayMasteringLuminance;
                        return S_OK;
                case bmdDeckLinkFrameMetadataHDRMinDisplayMasteringLuminance:
                        *value = m_metadata.minDisplayMasteringLuminance;
                        return S_OK;
                case bmdDeckLinkFrameMetadataHDRMaximumContentLightLevel:
                        *value = m_metadata.maxCLL;
                        return S_OK;
                case bmdDeckLinkFrameMetadataHDRMaximumFrameAverageLightLevel:
                        *value = m_metadata.maxFALL;
                        return S_OK;
                default:
                        value = nullptr;
                        return E_INVALIDARG;
        }
}

HRESULT DeckLinkFrame::GetFlag(BMDDeckLinkFrameMetadataID metadataID, BMD_BOOL* value)
{
        debug_print_metadata_id(static_cast<const char *>(__func__), metadataID);
        // Not expecting GetFlag
        *value = BMD_TRUE;
        return E_INVALIDARG;
}

HRESULT DeckLinkFrame::GetString(BMDDeckLinkFrameMetadataID metadataID, BMD_STR* value)
{
        debug_print_metadata_id(static_cast<const char *>(__func__), metadataID);
        // Not expecting GetString
        *value = nullptr;
        return E_INVALIDARG;
}

HRESULT DeckLinkFrame::GetBytes(BMDDeckLinkFrameMetadataID metadataID, void* /* buffer */, uint32_t* bufferSize)
{
        debug_print_metadata_id(static_cast<const char *>(__func__), metadataID);
        *bufferSize = 0;
        return E_INVALIDARG;
}

// 3D frame
DeckLink3DFrame::DeckLink3DFrame(long w, long h, long rb, BMDPixelFormat pf, buffer_pool_t & buffer_pool, HDRMetadata const & hdr_metadata)
        : DeckLinkFrame(w, h, rb, pf, buffer_pool, hdr_metadata), rightEye(DeckLinkFrame::Create(w, h, rb, pf, buffer_pool, hdr_metadata))
{
}

DeckLink3DFrame *DeckLink3DFrame::Create(long width, long height, long rawBytes, BMDPixelFormat pixelFormat, buffer_pool_t & buffer_pool, HDRMetadata const & hdr_metadata)
{
        DeckLink3DFrame *frame = new DeckLink3DFrame(width, height, rawBytes, pixelFormat, buffer_pool, hdr_metadata);
        return frame;
}

DeckLink3DFrame::~DeckLink3DFrame()
{
}

ULONG DeckLink3DFrame::AddRef()
{
        return DeckLinkFrame::AddRef();
}

ULONG DeckLink3DFrame::Release()
{
        return DeckLinkFrame::Release();
}

HRESULT DeckLink3DFrame::QueryInterface(REFIID id, void **data)
{
        LOG(LOG_LEVEL_DEBUG) << MOD_NAME << "DecklLink3DFrame QueryInterface " << id << "\n";
        if(id == IID_IDeckLinkVideoFrame3DExtensions)
        {
                this->AddRef();
                *data = dynamic_cast<IDeckLinkVideoFrame3DExtensions *>(this);
                return S_OK;
        }
        return DeckLinkFrame::QueryInterface(id, data);
}

BMDVideo3DPackingFormat DeckLink3DFrame::Get3DPackingFormat()
{
        return bmdVideo3DPackingLeftOnly;
}

HRESULT DeckLink3DFrame::GetFrameForRightEye(IDeckLinkVideoFrame ** frame) 
{
        *frame = rightEye.get();
        rightEye->AddRef();
        return S_OK;
}

/* function from DeckLink SDK sample DeviceList */

static void print_output_modes (IDeckLink* deckLink)
{
        IDeckLinkOutput*                        deckLinkOutput = NULL;
        IDeckLinkDisplayModeIterator*           displayModeIterator = NULL;
        IDeckLinkDisplayMode*                   displayMode = NULL;
        HRESULT                                 result;
        int                                     displayModeNumber = 0;

        // Query the DeckLink for its configuration interface
        result = deckLink->QueryInterface(IID_IDeckLinkOutput, (void**)&deckLinkOutput);
        if (result != S_OK)
        {
                fprintf(stderr, "Could not obtain the IDeckLinkOutput interface - result = %08x\n", (int) result);
                if (result == E_NOINTERFACE) {
                        printf("Device doesn't support video playback.\n");
                }
                goto bail;
        }

        // Obtain an IDeckLinkDisplayModeIterator to enumerate the display modes supported on output
        result = deckLinkOutput->GetDisplayModeIterator(&displayModeIterator);
        if (result != S_OK)
        {
                fprintf(stderr, "Could not obtain the video output display mode iterator - result = %08x\n", (int) result);
                goto bail;
        }

        // List all supported output display modes
        printf("\tdisplay modes:\n");
        while (displayModeIterator->Next(&displayMode) == S_OK)
        {
                BMD_STR                  displayModeString = NULL;

                result = displayMode->GetName(&displayModeString);

                if (result == S_OK)
                {
                        char *displayModeCString = get_cstr_from_bmd_api_str(displayModeString);
                        int                             modeWidth;
                        int                             modeHeight;
                        BMDDisplayModeFlags             flags;
                        BMDTimeValue    frameRateDuration;
                        BMDTimeScale    frameRateScale;

                        // Obtain the display mode's properties
                        flags = displayMode->GetFlags();
                        modeWidth = displayMode->GetWidth();
                        modeHeight = displayMode->GetHeight();
                        displayMode->GetFrameRate(&frameRateDuration, &frameRateScale);
                        printf("\t\t%2d) %-20s  %d x %d \t %2.2f FPS%s\n",displayModeNumber, displayModeCString,
                                        modeWidth, modeHeight, (float) ((double)frameRateScale / (double)frameRateDuration),
                                        (flags & bmdDisplayModeSupports3D ? "\t (supports 3D)" : ""));
                        release_bmd_api_str(displayModeString);
                        free(displayModeCString);
                }

                // Release the IDeckLinkDisplayMode object to prevent a leak
                displayMode->Release();

                displayModeNumber++;
        }

bail:
        // Ensure that the interfaces we obtained are released to prevent a memory leak
        if (displayModeIterator != NULL)
                displayModeIterator->Release();

        if (deckLinkOutput != NULL)
                deckLinkOutput->Release();
}

static const struct video_display_info display_decklink_info = {
        display_decklink_probe,
        display_decklink_init,
        display_decklink_run,
        display_decklink_done,
        display_decklink_getf,
        display_decklink_putf,
        display_decklink_reconfigure_video,
        display_decklink_get_property,
        display_decklink_put_audio_frame,
        display_decklink_reconfigure_audio,
        DISPLAY_DOESNT_NEED_MAINLOOP,
};

REGISTER_MODULE(decklink, &display_decklink_info, LIBRARY_CLASS_VIDEO_DISPLAY, VIDEO_DISPLAY_ABI_VERSION);

/* vim: set expandtab sw=8: */
