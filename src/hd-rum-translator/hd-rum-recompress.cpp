#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif

#include "hd-rum-translator/hd-rum-recompress.h"

extern "C" {
#include "debug.h"
#include "rtp/rtp.h"
#include "rtp/rtp_callback.h"
#include "transmit.h"
#include "tv.h"
}

#include "module.h"

#include "video.h"
#include "video_codec.h"
#include "video_compress.h"
#include "video_decompress.h"

extern int mtu; // defined in hd-rum-transcode.c

struct state_recompress {
        int rx_port, tx_port;
        const char *host;
        struct rtp *network_device;
        struct compress_state *compress;
        char *required_compress;

        struct timeval start_time;

        struct video_frame *frame;

        struct tx *tx;

        pthread_mutex_t lock;
        pthread_cond_t  have_frame_cv;
        pthread_cond_t  frame_consumed_cv;
        pthread_t       thread_id;

        struct module   *parent;
};

/*
 * Prototypes
 */
static void *worker(void *arg);
static void recompress_rtp_callback(struct rtp *session, rtp_event *e);

static void *worker(void *arg)
{
        struct state_recompress *s = (struct state_recompress *) arg;
        struct timeval t0, t;
        int frames = 0;
        struct module sender_mod;

        int ret = compress_init(s->parent, s->required_compress, &s->compress);
        if(ret != 0) {
                fprintf(stderr, "Unable to initialize video compress: %s\n",
                                s->required_compress);
                pthread_mutex_unlock(&s->lock);
                return NULL;
        }

        module_init_default(&sender_mod);
        sender_mod.cls = MODULE_CLASS_SENDER;
        module_register(&sender_mod, s->parent);

        pthread_mutex_unlock(&s->lock);

        gettimeofday(&t0, NULL);

        while(1) {
                pthread_mutex_lock(&s->lock);
                while(!s->frame) {
                        pthread_cond_wait(&s->have_frame_cv, &s->lock);
                }

                if(s->frame->tiles[0].data_len == 0) { // poisoned pill
                        vf_free(s->frame);
                        s->frame = NULL;
                        pthread_mutex_unlock(&s->lock);
                        break;
                }
                pthread_mutex_unlock(&s->lock);

                struct message *msg;
                while((msg = check_message(&sender_mod))) {
                        struct msg_sender *data = (struct msg_sender *) msg;
                        rtp_change_dest(s->network_device,
                                        data->receiver);
                        free_message(msg);
                }

                compress_frame(s->compress, s->frame);
                struct video_frame *tx_frame =
                        compress_pop(s->compress);

                if(tx_frame) {
                        tx_send(s->tx, tx_frame, s->network_device);
                } else {
                        fprintf(stderr, "Compress failed\n");
                }

                pthread_mutex_lock(&s->lock);
                s->frame = NULL;
                pthread_cond_signal(&s->frame_consumed_cv);
                pthread_mutex_unlock(&s->lock);

                frames += 1;
                gettimeofday(&t, NULL);
                double seconds = tv_diff(t, t0);
                if(seconds > 5) {
                        double fps = frames / seconds;
                        fprintf(stdout, "[%.4x->%s:%d] %d frames in %g seconds = %g FPS\n",
                                        rtp_my_ssrc(s->network_device),
                                        s->host, s->tx_port,
                                        frames, seconds, fps);
                        t0 = t;
                        frames = 0;
                }
        }

        if(s->compress) {
                module_done(CAST_MODULE(s->compress));
                s->compress = NULL;
        }

        module_done(&sender_mod);

        return NULL;
}

static struct rtp *initialize_network(const char *host, int rx_port, int tx_port, void *udata,
                int64_t ssrc = -1) {
        int ttl = 255;
        double rtcp_bw = 5 * 1024 * 1024;       /*  FIXME */
        bool use_ipv6 = false;

        struct rtp *network_device = rtp_init(host, rx_port, tx_port, ttl,
                        rtcp_bw, FALSE, recompress_rtp_callback, (uint8_t *) udata,
                        use_ipv6, false);
        if(network_device) {
                if(ssrc != -1) {
                        int ret = rtp_set_my_ssrc(network_device, ssrc);
                        if(!ret) {
                                fprintf(stderr, "Cannot set custom SSRC.\n");
                        }

                }
                rtp_set_option(network_device, RTP_OPT_WEAK_VALIDATION,
                                TRUE);
                rtp_set_sdes(network_device, rtp_my_ssrc(network_device),
                                RTCP_SDES_TOOL,
                                PACKAGE_STRING, strlen(PACKAGE_STRING));
        }

        return network_device;
}

void *recompress_init(struct module *parent,
                const char *host, const char *compress, unsigned short rx_port,
                unsigned short tx_port, int mtu, char *fec, long packet_rate)
{
        struct state_recompress *s;

        initialize_video_decompress();

        s = (struct state_recompress *) calloc(1, sizeof(struct state_recompress));

        s->parent = parent;

        s->network_device = initialize_network(host, rx_port, tx_port, s);
        s->host = host;
        s->rx_port = rx_port;
        s->tx_port = tx_port;
        if(!s->network_device) {
                fprintf(stderr, "Unable to initialize RTP.\n");
                return NULL;
        }


        s->required_compress = strdup(compress);
        s->frame = NULL;
        const char *requested_encryption = NULL;
        s->tx = tx_init(s->parent, mtu, TX_MEDIA_VIDEO, fec, requested_encryption, packet_rate);

        gettimeofday(&s->start_time, NULL);

        pthread_mutex_init(&s->lock, NULL);
        pthread_cond_init(&s->have_frame_cv, NULL);
        pthread_cond_init(&s->frame_consumed_cv, NULL);

        pthread_mutex_lock(&s->lock);

        if(pthread_create(&s->thread_id, NULL, worker, (void *) s) != 0) {
                fprintf(stderr, "Unable to create thread\n");
                return NULL;
        }

        pthread_mutex_lock(&s->lock);
        if(s->compress == NULL) {
                pthread_join(s->thread_id, NULL);
                return NULL;
        }
        pthread_mutex_unlock(&s->lock);

        return (void *) s;
}

void recompress_process_async(void *state, struct video_frame *frame)
{
       struct state_recompress *s = (struct state_recompress *) state;

       pthread_mutex_lock(&s->lock);
       assert(s->frame == NULL);
       s->frame = frame;
       pthread_cond_signal(&s->have_frame_cv);
       pthread_mutex_unlock(&s->lock);
}

void recompress_wait_complete(void *state)
{
       struct state_recompress *s = (struct state_recompress *) state;

       pthread_mutex_lock(&s->lock);
       while(s->frame != NULL)
               pthread_cond_wait(&s->frame_consumed_cv, &s->lock);
       pthread_mutex_unlock(&s->lock);
}

void recompress_assign_ssrc(void *state, uint32_t ssrc)
{
       struct state_recompress *s = (struct state_recompress *) state;

       rtp_done(s->network_device);
       s->network_device = initialize_network(s->host, s->rx_port, s->tx_port,
                       s, ssrc);
}

static void recompress_rtp_callback(struct rtp *session, rtp_event *e)
{
        UNUSED(session);
        UNUSED(e);
}

void recompress_done(void *state)
{
        struct state_recompress *s = (struct state_recompress *) state;

        pthread_mutex_lock(&s->lock);
        assert(s->frame == NULL);
        s->frame = vf_alloc(1);
        pthread_cond_signal(&s->have_frame_cv);
        pthread_mutex_unlock(&s->lock);

        pthread_join(s->thread_id, NULL);

        module_done(CAST_MODULE(s->tx));
        rtp_done(s->network_device);
        pthread_mutex_destroy(&s->lock);
        pthread_cond_destroy(&s->have_frame_cv);
        pthread_cond_destroy(&s->frame_consumed_cv);

        free(s);
}

