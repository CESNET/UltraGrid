/**
 * @file   audio/playback/mixer.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2016-2023 CESNET z.s.p.o.
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

#include "audio/audio_capture.h"
#include "audio/audio_playback.h"
#include "audio/codec.h"
#include "audio/types.h"
#include "debug.h"
#include "lib_common.h"
#include "module.h"
#include "rtp/rtp.h"
#include "transmit.h"
#include "utils/audio_buffer.h"
#include "utils/thread.h"
#include <chrono>
#include <iostream>
#include <map>
#include <mutex>
#include <thread>
#include <vector>

#define SAMPLE_RATE 48000
#define BPS     2 /// @todo 4?
#define CHANNELS 1
#define FRAMES_PER_SEC 25
static_assert(SAMPLE_RATE % FRAMES_PER_SEC == 0, "Sample rate not divisible by frames per sec!");
#define SAMPLES_PER_FRAME (SAMPLE_RATE / FRAMES_PER_SEC)

#define PARTICIPANT_TIMEOUT_S 60
typedef int16_t sample_type_source;
typedef int32_t sample_type_mixed;
static_assert(sizeof(sample_type_source) == BPS, "sample_type source doesn't match BPS");
static_assert(sizeof(sample_type_mixed) > sizeof(sample_type_source), "sample_type_mixed is not wider than sample_type_source");

using namespace std;
using namespace std::chrono;

class sockaddr_storage_less {
public:
        bool operator() (const sockaddr_storage & x, const sockaddr_storage & y) const {
                if (x.ss_family != y.ss_family) {
                        return x.ss_family < y.ss_family;
                }

                if (x.ss_family == AF_INET) {
                        auto sin_x = reinterpret_cast<const sockaddr_in &>(x);
                        auto sin_y = reinterpret_cast<const sockaddr_in &>(y);

                        if (sin_x.sin_addr.s_addr != sin_y.sin_addr.s_addr) {
                                return sin_x.sin_addr.s_addr < sin_y.sin_addr.s_addr;
                        }
                        return sin_x.sin_port < sin_y.sin_port;
                } else if (x.ss_family == AF_INET6) {
                        auto sin_x = reinterpret_cast<const sockaddr_in6 &>(x);
                        auto sin_y = reinterpret_cast<const sockaddr_in6 &>(y);

                        for (int i = 0; i < 16; ++i) {
                                if (sin_x.sin6_addr.s6_addr[i] != sin_y.sin6_addr.s6_addr[i]) {
                                        return sin_x.sin6_addr.s6_addr[i] < sin_y.sin6_addr.s6_addr[i];
                                }
                        }

                        return sin_x.sin6_port < sin_y.sin6_port;
                }
                abort();
        }
};

static void mixer_dummy_rtp_callback(struct rtp *session [[gnu::unused]], rtp_event * e [[gnu::unused]]) {
}

struct am_participant {
        am_participant(struct socket_udp_local *l, struct sockaddr_storage *ss, string const & audio_codec) {
                assert(l != nullptr && ss != nullptr);
                m_buffer = audio_buffer_init(SAMPLE_RATE, BPS, CHANNELS, get_commandline_param("low-latency-audio") ? 50 : 5);
                assert(m_buffer != NULL);
                struct sockaddr *sa = (struct sockaddr *) ss;
                assert(ss->ss_family == AF_INET || ss->ss_family == AF_INET6);
                socklen_t len = ss->ss_family == AF_INET ? sizeof(struct sockaddr_in) : sizeof(struct sockaddr_in6);

                m_network_device = rtp_init_with_udp_socket(l, sa, len, mixer_dummy_rtp_callback);
                assert(m_network_device != NULL);
                m_tx_session = tx_init(NULL, 1500, TX_MEDIA_AUDIO, NULL, NULL, RATE_UNLIMITED);
                assert(m_tx_session != NULL);

                m_audio_coder = audio_codec_init_cfg(audio_codec.c_str(), AUDIO_CODER);
                if (!m_audio_coder) {
                        LOG(LOG_LEVEL_ERROR) << "Audio coder init failed!\n";
                        throw 1;
                }
        }
        ~am_participant() {
                if (m_tx_session) {
                        module_done(CAST_MODULE(m_tx_session));
                }
                if (m_network_device) {
                        rtp_done(m_network_device);
                }
                if (m_buffer) {
                        audio_buffer_destroy(m_buffer);
                }
                if (m_audio_coder) {
                        audio_codec_done(m_audio_coder);
                }
	}
	am_participant& operator=(am_participant&& other) {
		m_audio_coder = std::move(other.m_audio_coder);
		m_buffer = std::move(other.m_buffer);
		m_network_device = std::move(other.m_network_device);
		m_tx_session = std::move(other.m_tx_session);
		last_seen = std::move(other.last_seen);
		other.m_audio_coder = nullptr;
		other.m_buffer = nullptr;
		other.m_tx_session = nullptr;
		other.m_network_device = nullptr;
		return *this;
	}
        am_participant(am_participant && other) {
                *this = std::move(other);
        }
        struct audio_codec_state *m_audio_coder;
        struct audio_buffer *m_buffer;
        struct rtp *m_network_device;
        struct tx *m_tx_session;
        chrono::steady_clock::time_point last_seen;
};

template<typename source_t, typename intermediate_t>
class generic_mix_algo {
public:
        virtual ~generic_mix_algo() {}
        virtual intermediate_t add_to_mix(intermediate_t dst, source_t sample) {
                return dst + sample;
        }

        virtual intermediate_t get_mixed_without_source_sample(intermediate_t mix, source_t source_sample) {
                return mix - source_sample;
        }

        virtual intermediate_t normalize(intermediate_t sample) = 0;
};

/**
 * In this mixer, no normalization takes place. After mixing and substracting each
 * participant signal, values are clamped (there is no point doing it prior that -
 * non-normalized mixed value can be out-of-bounds while resulting value with
 * substracted with substracted source may be ok.
 */
template<typename source_t, typename intermediate_t>
class linear_mix_algo : public generic_mix_algo<source_t, intermediate_t> {
public:
        intermediate_t normalize(intermediate_t sample) override {
                // clamp the value since linear mixer doesn't normalize values
                return min<intermediate_t>(max<intermediate_t>(sample, numeric_limits<source_t>::min()), numeric_limits<source_t>::max());
        }
};

/**
 * Logarithmic mixing according to:
 * https://www.voegler.eu/pub/audio/digital-audio-mixing-and-normalization.html
 * Copy (as the original link doesn't seem to be present any more) can be found here:
 * http://www.voidcn.com/blog/caohongfei881/article/p-3815311.html
 * Threshold is 0.5.
 */
template<typename source_t, typename intermediate_t>
class logarithmic_mix_algo : public generic_mix_algo<source_t, intermediate_t> {
public:
        static constexpr double t = 0.5;
        static constexpr double alpha = 5.71144;
        intermediate_t normalize(intermediate_t sample) override {
		if (sample >= numeric_limits<source_t>::min() / 2 &&
				sample <= numeric_limits<source_t>::max() / 2) {
			return sample;
		} else {
                        double sample_norm = (double) sample / numeric_limits<source_t>::max();
                        double ret = sample_norm / fabs(sample_norm) * (t + (1.0 - t) * log(1.0 + alpha * (fabs(sample_norm) - t) / (2 - t)) / log(1.0 + alpha)) * numeric_limits<source_t>::max();
                        return ret;
                }
        }
};

struct state_audio_mixer final {
        state_audio_mixer(const char *cfg) {
                if (cfg) {
                        shared_ptr<char> tmp(strdup(cfg), free);
                        char *item, *save_ptr;
                        char *copy = tmp.get();

                        while ((item = strtok_r(copy, ":", &save_ptr))) {
                                if (strncmp(item, "codec=", strlen("codec=")) == 0) {
                                        audio_codec = item + strlen("codec=");
                                } else if (strncmp(item, "algo=", strlen("algo=")) == 0) {
                                        string algo = item + strlen("algo=");
                                        if (algo == "linear") {
                                                mixing_algorithm = decltype(mixing_algorithm)(new linear_mix_algo<sample_type_source, sample_type_mixed>());
                                        } else if (algo == "logarithmic") {
                                                mixing_algorithm = decltype(mixing_algorithm)(new logarithmic_mix_algo<sample_type_source, sample_type_mixed>());
                                        } else {
                                                LOG(LOG_LEVEL_ERROR) << "Unknown mixing algorithm: " << algo << "\n";
                                                throw 1;
                                        }
                                } else {
                                        LOG(LOG_LEVEL_ERROR) << "Unknown option: " << item << "\n";
                                        throw 1;
                                }
                                copy = nullptr;
                        }
                }

                struct audio_codec_state *audio_coder =
                        audio_codec_init_cfg(audio_codec.c_str(), AUDIO_CODER);
                if (!audio_coder) {
                        LOG(LOG_LEVEL_ERROR) << "Audio coder init failed!\n";
                        throw 1;
                } else {
                        audio_codec_done(audio_coder);
                }

                thread_id = thread(&state_audio_mixer::worker, this);
        }
        ~state_audio_mixer() {
                thread_id.join();
        }
        bool should_exit = false;
        state_audio_mixer(state_audio_mixer const&)            = delete;
        state_audio_mixer& operator=(state_audio_mixer const&) = delete;
        void worker();

        map<sockaddr_storage, am_participant, sockaddr_storage_less> participants;
        mutex participants_lock;

        struct socket_udp_local *recv_socket{};
        string audio_codec{"PCM"};
private:
        thread thread_id;
        unique_ptr<generic_mix_algo<sample_type_source, sample_type_mixed>> mixing_algorithm{new linear_mix_algo<sample_type_source, sample_type_mixed>()};
};

void state_audio_mixer::worker()
{
        set_thread_name(__func__);
        chrono::steady_clock::time_point next_frame_time = chrono::steady_clock::now();

        static_assert(SAMPLES_PER_FRAME * 1000ll % SAMPLE_RATE == 0, "Sample rate is not evenly divisible by number of samples in frame");
        const chrono::milliseconds interval(SAMPLES_PER_FRAME*1000ll/SAMPLE_RATE);

        while (!should_exit) {
                this_thread::sleep_until(next_frame_time);
                next_frame_time += interval;
                auto now = chrono::steady_clock::now();

                // check if we didn't overslept much
                if (next_frame_time < now) {
                        LOG(LOG_LEVEL_WARNING) << "[Audio mixer] Next frame time in past! Setting to now.\n";
                        next_frame_time = now;
                }

                unique_lock<mutex> plk(participants_lock);
                // check timeouts
                for (auto it = participants.cbegin(); it != participants.cend(); )
                {
                        if (duration_cast<seconds>(now - it->second.last_seen).count() > PARTICIPANT_TIMEOUT_S) {
                                it = participants.erase(it);
                        } else {
                                ++it;
                        }
                }

                size_t data_len_source = SAMPLES_PER_FRAME * sizeof(sample_type_source) * CHANNELS;
                vector<sample_type_mixed> mixed(SAMPLES_PER_FRAME * CHANNELS);
                vector<audio_frame2> participant_frames(participants.size());
                int participant_index = 0;

                // mix all together
                for (auto & p : participants) {
                        participant_frames[participant_index].init(CHANNELS, AC_PCM, BPS, SAMPLE_RATE);
                        static_assert(CHANNELS == 1, "Currently only one channel is implemented here.");
                        participant_frames[participant_index].resize(0, data_len_source);
                        char *particip_data = participant_frames[participant_index].get_data(0);
                        int ret = audio_buffer_read(p.second.m_buffer, particip_data, data_len_source);
                        memset(particip_data, 0, data_len_source - ret);

                        sample_type_source *src = (sample_type_source *)(void *) particip_data;
                        auto dst = mixed.begin();
                        for (int i = 0; i < SAMPLES_PER_FRAME * CHANNELS; ++i) {
                                *dst = mixing_algorithm->add_to_mix(*dst, *src);
                                ++dst;
                                ++src;
                        }
                        participant_index++;
                }

                // substract each source signal from the mix coming to that participant
                for (auto & pb : participant_frames) {
                        auto mix = mixed.begin();
                        sample_type_source *part = (sample_type_source *)(void *) pb.get_data(0);
                        for (int i = 0; i < SAMPLES_PER_FRAME * CHANNELS; ++i) {
                                *part = mixing_algorithm->normalize(mixing_algorithm->get_mixed_without_source_sample(*mix, *part));
                                ++mix;
                                ++part;
                        }
                }

                // send
                participant_index = 0;
                for (auto & p : participants) {
                        audio_frame2 *uncompressed = &participant_frames[participant_index];
                        while (audio_frame2 compressed = audio_codec_compress(p.second.m_audio_coder, uncompressed)) {
                                audio_tx_send(p.second.m_tx_session, p.second.m_network_device, &compressed);
                                uncompressed = nullptr;
                        }

                        participant_index++;
                }
                plk.unlock();
        }
}

static void audio_play_mixer_help()
{
        printf("Usage:\n"
               "\t%s -r mixer[:codec=<codec>][:algo={linear|logarithmic}]\n"
               "\n"
               "<codec>\n"
               "\taudio codec to use\n"
               "linear\n"
               "\tlinear sum of signals (with clamping)\n"
               "logarithmic\n"
               "\tlinear sum of signals to threshold, above threshold logarithmic dynamic range compression is used\n"
               "\n"
               "Notes:\n"
               "1)\tYou do not need to specify audio participants explicitly,\n"
               "\t" PACKAGE_NAME " simply sends the the stream back to the host\n"
               "\tthat is sending to mixer. Therefore it is necessary that the\n"
               "\tparticipant uses single " PACKAGE_NAME " for both sending and\n"
               "\treceiving audio.\n"
               "2)\tUses default port for receiving, therefore if you want to use it\n"
               "\ton machine that is a part of the conference, you should use something like:\n"
               "\t\t%s -s <your_capture> -P 5004:5004:5010:5006\n"
               "\tfor the " PACKAGE_NAME " instance that is part of the conference (not mixer!)\n",
               uv_argv[0], uv_argv[0]);
}

static void audio_play_mixer_probe(struct device_info **available_devices, int *count, void (**deleter)(void *))
{
        *deleter = free;
        *available_devices = NULL;
        *count = 0;
}

static void * audio_play_mixer_init(const char *cfg)
{
        if (strcmp(cfg, "help") == 0) {
                audio_play_mixer_help();
                return INIT_NOERR;
        }
        try {
                return new state_audio_mixer{cfg};
        } catch (...) {
                return nullptr;
        }
}

static void audio_play_mixer_put_frame(void *state, const struct audio_frame *frame)
{
        struct state_audio_mixer *s = (struct state_audio_mixer *) state;

        unique_lock<mutex> lk(s->participants_lock);

        auto ss = *(struct sockaddr_storage *) frame->network_source;

        if (s->participants.find(ss) == s->participants.end()) {
                s->participants.emplace(ss, am_participant{s->recv_socket, &ss, s->audio_codec});
        }

        audio_buffer_write(s->participants.at(ss).m_buffer, frame->data, frame->data_len);
        s->participants.at(ss).last_seen = chrono::steady_clock::now();
}

static void audio_play_mixer_done(void *state)
{
        struct state_audio_mixer *s = (struct state_audio_mixer *) state;
        s->should_exit = true;

        delete s;
}

static bool audio_play_mixer_ctl(void *state, int request, void *data, size_t *len)
{
        struct state_audio_mixer *s = (struct state_audio_mixer *) state;

        switch (request) {
        case AUDIO_PLAYBACK_CTL_QUERY_FORMAT:
                if (*len >= sizeof(struct audio_desc)) {
                        struct audio_desc desc { BPS, SAMPLE_RATE, CHANNELS, AC_PCM };
                        memcpy(data, &desc, sizeof desc);
                        *len = sizeof desc;
                        return true;
                } else {
                        return false;
                }
        case AUDIO_PLAYBACK_CTL_MULTIPLE_STREAMS:
                if (*len >= sizeof(bool)) {
                        *(bool *) data = true;
                        *len = sizeof(bool);
                        return true;
                } else {
                        return false;
                }
        case AUDIO_PLAYBACK_PUT_NETWORK_DEVICE:
                if (*len != sizeof(struct rtp *)) {
                        return false;
                } else {
                        s->recv_socket = rtp_get_udp_local_socket(*(struct rtp **) data);
                        assert(s->recv_socket != NULL);
                        return true;
                }
        default:
                return false;
        }
}

static bool audio_play_mixer_reconfigure(void *state [[gnu::unused]], struct audio_desc desc)
{
        audio_desc requested{BPS, SAMPLE_RATE, CHANNELS, AC_PCM};
        assert(desc == requested);
        return true;
}

static const struct audio_playback_info aplay_mixer_info = {
        audio_play_mixer_probe,
        audio_play_mixer_init,
        audio_play_mixer_put_frame,
        audio_play_mixer_ctl,
        audio_play_mixer_reconfigure,
        audio_play_mixer_done
};

REGISTER_MODULE(mixer, &aplay_mixer_info, LIBRARY_CLASS_AUDIO_PLAYBACK, AUDIO_PLAYBACK_ABI_VERSION);

