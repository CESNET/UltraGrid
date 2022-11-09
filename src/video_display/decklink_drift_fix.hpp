/**
 * @file   video_display/decklink_drift_fix.hpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 * @author Andrew Walker    <andrew.walker@sohonet.com>
 */
/*
 * Copyright (c) 2021-2022 CESNET, z. s. p. o.
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

#ifndef SRC_VIDEO_DISPLAY_DECKLINK_DRIFT_FIX_HPP_69DBC8A9_974D_46C5_833D_5A77CF35E034
#define SRC_VIDEO_DISPLAY_DECKLINK_DRIFT_FIX_HPP_69DBC8A9_974D_46C5_833D_5A77CF35E034

#include <string>

#include "utils/color_out.h"

#define MAX_RESAMPLE_DELTA_DEFAULT 30
#define MIN_RESAMPLE_DELTA_DEFAULT 1
#define TARGET_BUFFER_DEFAULT 2700

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
                if (this->size() >= (ptrdiff_t) this->period ){
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

class DecklinkAudioSummary {
public:
        /**
         * @brief This will detail out the longer running stats of the Decklink. It should be called on every audio frame
         *        but will only print out the report once every 30 seconds.
         */
        void report() {
                std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
                if(std::chrono::duration_cast<std::chrono::seconds>(now - this->last_summary).count() > 10) {                
                        LOG(LOG_LEVEL_INFO) << SUNDERLINE("Decklink stats (cumulative)")
                                        << " - Total Audio Frames Played: "
                                        << SBOLD(this->frames_played)
                                        << " / Missing Audio Frames: "
                                        << SBOLD(this->frames_missed)
                                        << " / Buffer Underflows: "
                                        << SBOLD(this->buffer_underflow)
                                        << " / Buffer Overflows: "
                                        << SBOLD(this->buffer_overflow)
                                        << " / Resample (Higher Hz): "
                                        << SBOLD(this->resample_high)
                                        << " / Resample (Lower Hz): "
                                        << SBOLD(this->resample_low)
                                        << " / Average Buffer: "
                                        << SBOLD(this->buffer_average)
                                        << " / Average Added Frames: "
                                        << SBOLD(this->avg_added_frames.avg())
                                        << " / Max time diff audio (ms): "
                                        << SBOLD(this->audio_time_diff_max)
                                        << " / Min time diff audio (ms): "
                                        << SBOLD(this->audio_time_diff_min)
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
        bool m_enabled = false;

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
        void update(uint32_t buffered_count, uint32_t sample_frame_count, uint32_t sampleFramesWritten) {
                if (!this->m_enabled) {
                        return;
                }

                audio_summary.record_audio_time_diff();
                audio_summary.calculate_missing(buffered_count, sample_frame_count);
                if (buffered_count == 0) {
                        audio_summary.increment_buffer_underflow();
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
                                this->audio_summary.increment_resample_low();
                        } else if(average_buffer_depth < target_buffer_fill - this->neg_jitter) {
                                 // The buffer is too small, so we need to resample up to generate some additional frames
                                resample_hz = (int)this->scale_buffer_delta(target_buffer_fill - average_buffer_depth - this->neg_jitter);
                                dst_frame_rate = (bmdAudioSampleRate48kHz + resample_hz) * BASE;
                                this->audio_summary.increment_resample_high();
                        } else {
                                dst_frame_rate = (bmdAudioSampleRate48kHz) * BASE;
                        }       
                }

                LOG(LOG_LEVEL_DEBUG) << MOD_NAME << " UPDATE playing speed " <<  average_buffer_depth << " vs " << buffered_count << " " << average_delta.avg() << " average_velocity " << resample_hz << " resample_hz\n";

   
                if (dst_frame_rate != 0) {
                        auto *m = new msg_universal((std::string(MSG_UNIVERSAL_TAG_AUDIO_DECODER) + std::to_string(dst_frame_rate << ADEC_CH_RATE_SHIFT | BASE)).c_str());
                        LOG(LOG_LEVEL_VERBOSE) << MOD_NAME "Sending resample request " << dst_frame_rate << "/" << BASE << "\n";
                        assert(m_root != nullptr);
                        auto *response = send_message_sync(m_root, "audio.receiver.decoder", reinterpret_cast<message *>(m), 100, SEND_MESSAGE_FLAG_NO_STORE);
                        if (!RESPONSE_SUCCESSFUL(response_get_status(response))) {
                                LOG(LOG_LEVEL_WARNING) << MOD_NAME "Unable to send resample message: " << response_get_text(response) << " (" << response_get_status(response) << ")\n";
                        }
                        free_response(response);
                }


                if (sampleFramesWritten != sample_frame_count) {
                        audio_summary.increment_buffer_overflow();
                }
                audio_summary.increment_audio_frames_played();
                audio_summary.set_buffer_average(get_buffer_avg());
                audio_summary.report();
                audio_summary.mark_audio_time_end();
        }

private:
        static constexpr unsigned long BASE = (1U<<8U);
        struct module *m_root = nullptr;

        MovingAverage average_buffer_samples = 250;
        MovingAverage average_delta = 25;

        uint32_t target_buffer_fill = TARGET_BUFFER_DEFAULT;
        uint32_t previous_buffer = 0;

        // The min and max Hz changes we can resample between
        uint32_t min_hz = MIN_RESAMPLE_DELTA_DEFAULT;
        uint32_t max_hz = MAX_RESAMPLE_DELTA_DEFAULT;
        // The min and max values to scale between
        uint32_t min_buffer = 100;
        uint32_t max_buffer = 600;
        // Calculate the jitter so that we're within an acceptable range
        uint32_t pos_jitter = 5;
        uint32_t neg_jitter = 5;
        // Currently unused but might form a part of a more dynamic
        // solution to finding good jitter values in the future. @todo
        [[maybe_unused]] uint32_t max_avg = 3650;
        [[maybe_unused]] uint32_t min_avg = 1800;

        // Store a audio_summary of resampling
        DecklinkAudioSummary audio_summary{};

        static const uint32_t POS_JITTER_DEFAULT = 600;
        static const uint32_t NEG_JITTER_DEFAULT = 600;
};

#endif // defined SRC_VIDEO_DISPLAY_DECKLINK_DRIFT_FIX_HPP_69DBC8A9_974D_46C5_833D_5A77CF35E034
