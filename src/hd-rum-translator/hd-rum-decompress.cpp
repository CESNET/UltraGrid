#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif

#include "hd-rum-translator/hd-rum-decompress.h"
#include "hd-rum-translator/hd-rum-recompress.h"
#include "rtp/ldgm.h"

#include <map>
#include <queue>

extern "C" {
#include "debug.h"
#include "rtp/video_decoders.h" // init_decompress()
#include "rtp/rtp.h"
#include "rtp/rtp_callback.h"
#include "transmit.h"
#include "tv.h"
}

#include "video.h"
#include "video_codec.h"
#include "video_compress.h"
#include "video_decompress.h"

#define MAX_QUEUE_SIZE 1

using namespace std;

extern int mtu; // defined in hd-rum-transcode.c

struct substream_frame {
        char *data;
        int data_len;
};

struct message {
        virtual ~message() {}
};

struct poisoned_pill : public message {
};

struct packet_desc {
        int pt;
        union {
                struct video_desc video;
                struct ldgm_desc ldgm;

        };
};

struct frame : public message {
        struct packet_desc packet_desc;
        int tile_count;

        map <int, substream_frame> substreams;
        map <int, map<int, int> > packets;
        uint32_t ts;
        uint32_t ssrc;

        ~frame() {
                for(map<int, substream_frame>::iterator it = substreams.begin();
                                it != substreams.end();
                                ++it) {
                        free(it->second.data);
                }
        }
};

struct state_transcoder_decompress {
        void **output_ports;
        int output_port_count;

        void **output_inact_ports;
        int output_inact_port_count;

        struct rtp *network_device;

        struct timeval start_time;

        struct frame *network_frame;
        queue<message *> received_frame;

        pthread_mutex_t lock;
        pthread_cond_t  have_frame_cv;
        pthread_t       thread_id;

        int64_t my_last_ssrc;

        uint32_t last_packet_ts;
};

/*
 * Prototypes
 */
static void hd_rum_receive_pkt(struct rtp *session, rtp_event *e);
static void receive_packet(struct state_transcoder_decompress *s, rtp_packet *pckt_rtp);
static bool decode_header(uint32_t *hdr, struct packet_desc *desc, int *buffer_len, int *substream);
static bool decode_video_header(uint32_t *hdr, struct video_desc *desc, int *buffer_len, int *substream);
static void *worker(void *arg);

void hd_rum_decompress_add_port(void *state, void *recompress_port)
{
        struct state_transcoder_decompress *s = (struct state_transcoder_decompress *) state;

        s->output_port_count += 1;
        s->output_ports = (void **) realloc(s->output_ports,
                        s->output_port_count * sizeof(void *));
        s->output_ports[s->output_port_count - 1] =
                recompress_port;
}

void hd_rum_decompress_add_inactive_port(void *state, void *recompress_port)
{
        struct state_transcoder_decompress *s = (struct state_transcoder_decompress *) state;

        s->output_inact_port_count += 1;
        s->output_inact_ports = (void **) realloc(s->output_inact_ports,
                        s->output_inact_port_count * sizeof(void *));
        s->output_inact_ports[s->output_inact_port_count - 1] =
                recompress_port;
}

ssize_t hd_rum_decompress_write(void *state, void *buf, size_t count)
{
        struct state_transcoder_decompress *s = (struct state_transcoder_decompress *) state;

        // if there are no active output ports, simply quit
        if (s->output_port_count == 0) {
                return 0;
        }

        struct timeval curr_time;
        gettimeofday(&curr_time, NULL);
        uint32_t ts = tv_diff(curr_time, s->start_time) * 90000;
        return rtp_recv_push_data(s->network_device,
                                          (char *) buf, count, ts);
}

#define MAX_SUBSTREAMS 1024

static void *worker(void *arg)
{
        struct state_transcoder_decompress *s = (struct state_transcoder_decompress *) arg;
        struct video_desc last_desc;
        struct ldgm_desc last_ldgm_desc;

        struct state_decompress *decompress = NULL;
        struct video_frame *uncompressed_frame[2] = { NULL, NULL };
        int tmp_buffers_count = 0;
        char **tmp_buffers[2] =  { (char **) malloc(tmp_buffers_count * sizeof(char *)),
                (char **) malloc(tmp_buffers_count * sizeof(char *)) };
        bool decompress_accepts_corrupted_frame = false;
        void *ldgm_state = NULL;

        char *compressed_buffers[MAX_SUBSTREAMS];
        int compressed_len[MAX_SUBSTREAMS];
        int current_idx = 0;

        memset(&last_desc, 0, sizeof(last_desc));
        memset(&last_ldgm_desc, 0, sizeof(last_ldgm_desc));

        struct video_desc video_header;

        pthread_mutex_unlock(&s->lock);

        struct frame *last_frame = NULL;

        while(1) {
                struct frame *current_frame;

                pthread_mutex_lock(&s->lock);
                while(s->received_frame.empty()) {
                        pthread_cond_wait(&s->have_frame_cv, &s->lock);
                }

                message *recv_frame = s->received_frame.front();
                s->received_frame.pop();
                current_frame = dynamic_cast<frame *>(recv_frame);
                if(!current_frame) { // poisoned pill - s->rec_fr was not (frame *)
                        delete recv_frame;
                        pthread_mutex_unlock(&s->lock);
                        break;
                }
                pthread_mutex_unlock(&s->lock);

                if(current_frame->packet_desc.pt == PT_VIDEO_LDGM) {
                        unsigned int k = current_frame->packet_desc.ldgm.k;
                        unsigned int m = current_frame->packet_desc.ldgm.m;
                        unsigned int c = current_frame->packet_desc.ldgm.c;
                        unsigned int seed = current_frame->packet_desc.ldgm.seed;
                        if(last_ldgm_desc.k != k ||
                                        last_ldgm_desc.m != m ||
                                        last_ldgm_desc.c != c ||
                                        last_ldgm_desc.seed != seed) {
                                ldgm_decoder_destroy(ldgm_state);
                                ldgm_state = ldgm_decoder_init(k, m, c, seed);
                                last_ldgm_desc = current_frame->packet_desc.ldgm;
                        }

                        for(int i = 0; i < current_frame->tile_count; ++i) {
                                compressed_len[i] = 0;
                                char *data = current_frame->substreams[i].data;
                                ldgm_decoder_decode_map(ldgm_state, data,
                                                current_frame->substreams[i].data_len,
                                                &compressed_buffers[i], &compressed_len[i],
                                                current_frame->packets[i]);

                                int unused_substr;
                                if(!compressed_len[i]) {
                                        fprintf(stderr, "LDGM: unable to reconstruct data.\n");
                                        delete current_frame;
                                        goto next_iteration;
                                }
                                if(!decode_video_header((uint32_t *) compressed_buffers[0], &video_header,
                                                &compressed_len[i], &unused_substr)) {
                                        delete current_frame;
                                        goto next_iteration;
                                }
                                compressed_buffers[i] += sizeof(video_payload_hdr_t);
                        }
                } else {
                        for(int i = 0; i < current_frame->tile_count; ++i) {
                                compressed_buffers[i] = current_frame->substreams[i].data;
                                compressed_len[i] = current_frame->substreams[i].data_len;
                        }
                        video_header = current_frame->packet_desc.video;
                }

                // reconfiguration
                if(!video_desc_eq(last_desc, video_header)) {
                        if(last_frame) {
                                for(int i = 0; i < s->output_port_count; ++i) {
                                        recompress_wait_complete(s->output_ports[i]);
                                }
                        }
                        delete last_frame;
                        last_frame = NULL;
                        for(int i = 0; i < 2; ++i) {
                                vf_free(uncompressed_frame[i]);
                                for(int j = 0; j < tmp_buffers_count; ++j) {
                                        free(tmp_buffers[i][j]);
                                }
                        }
                        struct video_desc uncompressed_desc = video_header;
                        uncompressed_desc.color_spec = UYVY;
                        uncompressed_desc.tile_count = current_frame->tile_count;

                        for(int i = 0; i < 2; ++i)
                                uncompressed_frame[i] = vf_alloc_desc(uncompressed_desc);

                        last_desc = video_header;

                        if(is_codec_opaque(video_header.color_spec)) {
                                if(decompress)
                                        decompress_done(decompress);
                                int ret;
                                ret = init_decompress(video_header.color_spec,
                                                uncompressed_desc.color_spec, &decompress, 1);
                                if(!ret) {
                                        fprintf(stderr, "Unable to initialize decompress!\n");
                                        abort();
                                }
                                uncompressed_desc.color_spec = UYVY;
                                int linesize = vc_get_linesize(uncompressed_desc.width,
                                                uncompressed_desc.color_spec);
                                int pitch = linesize;
                                if(!decompress_reconfigure(decompress, video_header,
                                                        0, 8, 16, pitch, uncompressed_desc.color_spec)) {

                                        fprintf(stderr, "Unable to reconfigure decompress!\n");
                                        abort();
                                }

                                int res = 0;
                                size_t size = sizeof(res);
                                ret = decompress_get_property(decompress,
                                                DECOMPRESS_PROPERTY_ACCEPTS_CORRUPTED_FRAME,
                                                &res,
                                                &size);
                                if(ret && res) {
                                        decompress_accepts_corrupted_frame = true;
                                } else {
                                        decompress_accepts_corrupted_frame = false;
                                }

                                tmp_buffers_count = current_frame->tile_count;
                                for(int i = 0; i < 2; ++i) {
                                        tmp_buffers[i] = (char **)
                                                realloc(tmp_buffers[i], tmp_buffers_count * sizeof(char *));
                                        for(int j = 0; j < current_frame->tile_count; ++j) {
                                                int data_len = linesize * uncompressed_desc.height;
                                                tmp_buffers[i][j] = (char *) malloc(data_len);
                                                uncompressed_frame[i]->tiles[j].data = tmp_buffers[i][j];
                                                uncompressed_frame[i]->tiles[j].data_len = data_len;
                                        }
                                }
                        }
                }

                // we need to decompress first
                if(is_codec_opaque(video_header.color_spec)) {
                        bool corrupted = false;
                        for(map <int, substream_frame>::iterator it = current_frame->substreams.begin();
                                        it != current_frame->substreams.end();
                                        ++it) {
                                int total_data_len = 0;
                                map<int, int> &substream_packets = current_frame->packets[it->first];
                                for(map <int, int>::iterator it_packets = substream_packets.begin();
                                                it_packets != substream_packets.end();
                                                ++it_packets) {
                                        total_data_len += it_packets->second;
                                }
                                if(total_data_len != it->second.data_len) {
                                        fprintf(stderr, "Corrupted frame!\n");
                                        corrupted = true;
                                }
                        }
                        if(!corrupted || decompress_accepts_corrupted_frame) {
                                for(int i = 0; i < current_frame->tile_count; ++i) {
                                        decompress_frame(decompress,
                                                        (unsigned char *) uncompressed_frame[current_idx]->tiles[i].data,
                                                        (unsigned char *) compressed_buffers[i],
                                                        compressed_len[i],
                                                        0 // TODO: propagate correct sequence number
                                                        );
                                }
                        } else {
                                fprintf(stderr, "Frame dropped by decompress.\n");
                                delete current_frame;
                                goto next_iteration;
                        }

                } else {
                        for(int i = 0; i < current_frame->tile_count; ++i) {
                                uncompressed_frame[current_idx]->tiles[i].data =
                                        compressed_buffers[i];
                                uncompressed_frame[current_idx]->tiles[i].data_len =
                                        compressed_len[i];
                        }
                }

                // to meet RTP requirements, translators must not touch SSRC
                // Watch out not to run this during recompress transmission
                if(s->my_last_ssrc != current_frame->ssrc) {
                        for(int i = 0; i < s->output_port_count; ++i) {
                                recompress_assign_ssrc(s->output_ports[i],
                                                current_frame->ssrc);
                        }
                        s->my_last_ssrc = current_frame->ssrc;
                }

                if(last_frame) {
                        for(int i = 0; i < s->output_port_count; ++i) {
                                recompress_wait_complete(s->output_ports[i]);
                        }
                }

                for(int i = 0; i < s->output_port_count; ++i) {
                        recompress_process_async(s->output_ports[i], uncompressed_frame[current_idx]);
                }

                delete last_frame;
                last_frame = current_frame;
                current_idx = (current_idx + 1) % 2;
next_iteration:
                ;
        }

        if(last_frame) {
                for(int i = 0; i < s->output_port_count; ++i) {
                        recompress_wait_complete(s->output_ports[i]);
                }
        }
        delete last_frame;
        last_frame = NULL;

        for(int i = 0; i < 2; ++i) {
                for(int j = 0; j < tmp_buffers_count; ++j) {
                        free(tmp_buffers[i][j]);
                }
                free(tmp_buffers[i]);
        }

        if(decompress) {
                decompress_done(decompress);
        }

        return NULL;
}

void *hd_rum_decompress_init(unsigned short rx_port)
{
        struct state_transcoder_decompress *s;
        int ttl = 255;
        double rtcp_bw = 5 * 1024 * 1024;       /*  FIXME */
        bool use_ipv6 = false;

        initialize_video_decompress();

        s = new state_transcoder_decompress;
        s->network_device = rtp_init_if("localhost", (char *) NULL, rx_port, rx_port, ttl,
                        rtcp_bw, FALSE, hd_rum_receive_pkt, (uint8_t *) s,
                        use_ipv6);
        s->my_last_ssrc = -1;
        if(!s->network_device) {
                fprintf(stderr, "Unable to initialize RTP.\n");
                return NULL;
        }
        rtp_set_option(s->network_device, RTP_OPT_WEAK_VALIDATION,
                        TRUE);
        rtp_set_sdes(s->network_device, rtp_my_ssrc(s->network_device),
                        RTCP_SDES_TOOL,
                        PACKAGE_STRING, strlen(PACKAGE_STRING));

        s->network_frame = NULL;

        s->output_port_count = 0;
        s->output_ports = (void **) malloc(0);
        s->output_inact_port_count = 0;
        s->output_inact_ports = (void **) malloc(0);

        s->last_packet_ts = 0u;

        gettimeofday(&s->start_time, NULL);

        pthread_mutex_init(&s->lock, NULL);
        pthread_cond_init(&s->have_frame_cv, NULL);

        pthread_mutex_lock(&s->lock);

        if(pthread_create(&s->thread_id, NULL, worker, (void *) s) != 0) {
                fprintf(stderr, "Unable to create thread\n");
                return NULL;
        }

        return (void *) s;
}

static bool decode_video_header(uint32_t *hdr, struct video_desc *desc, int *buffer_len, int *substream)
{
        int fps_pt, fpsd, fd, fi;
        uint32_t tmp;

        tmp = ntohl(hdr[0]);
        *substream = tmp >> 22;


        *buffer_len = ntohl(hdr[2]);
        desc->width = ntohl(hdr[3]) >> 16;
        desc->height = ntohl(hdr[3]) & 0xffff;
        desc->color_spec = get_codec_from_fcc(hdr[4]);
        if(desc->color_spec == (codec_t) -1) {
                fprintf(stderr, "Unknown FourCC \"%4s\"!\n", (char *) &hdr[4]);
                return false;
        }

        tmp = ntohl(hdr[5]);
        desc->interlacing = (enum interlacing_t) (tmp >> 29);
        fps_pt = (tmp >> 19) & 0x3ff;
        fpsd = (tmp >> 15) & 0xf;
        fd = (tmp >> 14) & 0x1;
        fi = (tmp >> 13) & 0x1;

        desc->fps = compute_fps(fps_pt, fpsd, fd, fi);

        return true;
}

static bool decode_header(uint32_t *hdr, struct packet_desc *desc, int *buffer_len, int *substream)
{
        if(desc->pt == PT_VIDEO) {
                return decode_video_header(hdr, &desc->video, buffer_len, substream);
        } else { // PT_VIDEO_LDGM
                uint32_t tmp;
                tmp = ntohl(hdr[0]);

                *substream = tmp >> 22;
                *buffer_len = ntohl(hdr[2]);

                tmp = ntohl(hdr[3]);
                desc->ldgm.k = tmp >> 19;
                desc->ldgm.m = 0x1fff & (tmp >> 6);
                desc->ldgm.c = 0x3f & tmp;
                desc->ldgm.seed = ntohl(hdr[4]);

                return true;
        }
}

static void decode_packet(rtp_packet *pckt, char *frame_buffer, map<int, int> &packets)
{
        int hdr_len;
        if(pckt->pt == PT_VIDEO) {
                hdr_len = sizeof(video_payload_hdr_t);
        } else {
                hdr_len = sizeof(ldgm_video_payload_hdr_t);
        }
        int len = pckt->data_len - hdr_len;
        char *data = (char *) pckt->data + hdr_len;
        int data_pos = ntohl(((uint32_t *) pckt->data)[1]);
        memcpy(frame_buffer + data_pos, data, len);
        packets[data_pos] = len;
}

static void receive_packet(struct state_transcoder_decompress *s, rtp_packet *pckt_rtp)
{
        if(pckt_rtp->pt != PT_VIDEO && pckt_rtp->pt != PT_VIDEO_LDGM) {
                fprintf(stderr, "Other packet types than video are not supported.\n");
                free(pckt_rtp);
                return;
        }


        uint32_t *pkt_hdr = (uint32_t *) pckt_rtp->data;
        struct packet_desc desc;
        int buffer_len;
        int substream_idx;

        desc.pt = pckt_rtp->pt;
        if(!decode_header(pkt_hdr, &desc, &buffer_len, &substream_idx)) {
                return;
        }

        if(s->network_frame && pckt_rtp->ts > s->network_frame->ts) {
                // new frame starts without the previous being completed
                delete s->network_frame;
                s->network_frame = 0;
                fprintf(stderr, "Frame dropped (new arriving while old in place)!\n");
        }

        if(!s->network_frame) { // this packet starts a new frame
                if(pckt_rtp->ts != s->last_packet_ts) { // this is not part of previous frame
                        s->network_frame = new frame;
                        s->network_frame->packet_desc = desc;
                        s->network_frame->ts = pckt_rtp->ts;
                        s->last_packet_ts = pckt_rtp->ts;
                        s->network_frame->ssrc = pckt_rtp->ssrc;
                } else { // we received a late frame
                        return;
                }
        }

        if(s->network_frame->substreams.find(substream_idx) ==
                        s->network_frame->substreams.end()) {
                substream_frame substr;
                substr.data = (char *) malloc(buffer_len);
                substr.data_len = buffer_len;
                s->network_frame->substreams[substream_idx] = substr;
                s->network_frame->packets[substream_idx] = map<int, int>();
        }

        decode_packet(pckt_rtp,
                        s->network_frame->substreams[substream_idx].data,
                        s->network_frame->packets[substream_idx]
                        );

        if(pckt_rtp->m) { // fire
                pthread_mutex_lock(&s->lock);
                if(s->received_frame.size() >= MAX_QUEUE_SIZE) {
                        fprintf(stderr, "Frame dropped (queue full)!\n");
                        delete s->network_frame;
                } else {
                        s->network_frame->tile_count =
                                s->network_frame->substreams.size();
                        s->received_frame.push(s->network_frame);
                }
                s->network_frame = 0;
                pthread_cond_signal(&s->have_frame_cv);
                pthread_mutex_unlock(&s->lock);
        }

        free(pckt_rtp);
}

static void hd_rum_receive_pkt(struct rtp *session, rtp_event *e)
{
        rtp_packet *pckt_rtp = (rtp_packet *) e->data;
        struct state_transcoder_decompress *s = (struct state_transcoder_decompress *)
                rtp_get_userdata(session);

        switch (e->type) {
        case RX_RTP:
                if (pckt_rtp->data_len > 0) {   /* Only process packets that contain data... */
                        receive_packet(s, pckt_rtp);
                }
                break;
        case RX_TFRC_RX:
                /* compute TCP friendly data rate */
                break;
        case RX_RTCP_START:
                break;
        case RX_RTCP_FINISH:
                break;
        case RX_SR:
                break;
        case RX_RR:
                break;
        case RX_RR_EMPTY:
                break;
        case RX_SDES:
                break;
        case RX_APP:
                break;
        case RX_BYE:
                break;
        case SOURCE_DELETED:
                break;
        case SOURCE_CREATED:
                break;
        case RR_TIMEOUT:
                break;
        default:
                debug_msg("Unknown RTP event (type=%d)\n", e->type);
        }

}

void hd_rum_decompress_done(void *state) {
        struct state_transcoder_decompress *s = (struct state_transcoder_decompress *) state;

        pthread_mutex_lock(&s->lock);
        s->received_frame.push(new poisoned_pill);
        pthread_cond_signal(&s->have_frame_cv);
        pthread_mutex_unlock(&s->lock);

        pthread_join(s->thread_id, NULL);

        // cleanup
        for(int i = 0; i < s->output_port_count; ++i) {
                recompress_done(s->output_ports[i]);
        }
        free(s->output_ports);

        for(int i = 0; i < s->output_inact_port_count; ++i) {
                recompress_done(s->output_inact_ports[i]);
        }
        free(s->output_inact_ports);

        delete s->network_frame;

        rtp_done(s->network_device);

        pthread_mutex_destroy(&s->lock);
        pthread_cond_destroy(&s->have_frame_cv);

        delete s;
}

