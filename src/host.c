/*
 * This file contains common external definitions
 */
#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif

#include "host.h"

#include "debug.h"
#include "video_capture.h"
#include "video_display.h"

#include "utils/resource_manager.h"
#include "rtp/decoders.h"
#include "rtp/rtp.h"
#include "rtp/pbuf.h"

long packet_rate;
unsigned int cuda_device = 0;
unsigned int audio_capture_channels = 2;

unsigned int cuda_devices[MAX_CUDA_DEVICES] = { 0 };
unsigned int cuda_devices_count = 1;

uint32_t RTT = 0;               /*  this is computed by handle_rr in rtp_callback */
uint32_t hd_color_spc = 0;

int uv_argc;
char **uv_argv;

char *export_dir = NULL;
char *sage_network_device = NULL;

extern void (*vidcap_free_devices_extrn)();
extern display_type_t *(*display_get_device_details_extrn)(int i);
extern struct vidcap_type *(*vidcap_get_device_details_extrn)(int i);
extern void (*display_free_devices_extrn)(void);
extern vidcap_id_t (*vidcap_get_null_device_id_extrn)();
extern display_id_t (*display_get_null_device_id_extrn)();
extern void (*decoder_destroy_extrn)(struct state_decoder *decoder);
extern int (*vidcap_init_extrn)(vidcap_id_t id, char *fmt, unsigned int flags, struct vidcap **);
extern int (*display_init_extrn)(display_id_t id, char *fmt, unsigned int flags, struct display **);
extern int (*vidcap_get_device_count_extrn)(void);
extern int (*display_get_device_count_extrn)(void);
extern int (*vidcap_init_devices_extrn)(void);
extern int (*display_init_devices_extrn)(void);

int initialize_video_capture(const char *requested_capture,
                                               char *fmt, unsigned int flags,
                                               struct vidcap **state)
{
        struct vidcap_type *vt;
        vidcap_id_t id = 0;
        int i;

        if(!strcmp(requested_capture, "none"))
                id = vidcap_get_null_device_id_extrn();

        pthread_mutex_t *vidcap_lock = rm_acquire_shared_lock("VIDCAP_LOCK");
        pthread_mutex_lock(vidcap_lock);

        vidcap_init_devices_extrn();
        for (i = 0; i < vidcap_get_device_count_extrn(); i++) {
                vt = vidcap_get_device_details_extrn(i);
                if (strcmp(vt->name, requested_capture) == 0) {
                        id = vt->id;
                        break;
                }
        }
        if(i == vidcap_get_device_count_extrn()) {
                fprintf(stderr, "WARNING: Selected '%s' capture card "
                        "was not found.\n", requested_capture);
                return -1;
        }
        vidcap_free_devices_extrn();

        pthread_mutex_unlock(vidcap_lock);
        rm_release_shared_lock("VIDCAP_LOCK");

        return vidcap_init_extrn(id, fmt, flags, state);
}

int initialize_video_display(const char *requested_display,
                                                char *fmt, unsigned int flags,
                                                struct display **out)
{
        struct display *d;
        display_type_t *dt;
        display_id_t id = 0;
        int i;

        if(!strcmp(requested_display, "none"))
                 id = display_get_null_device_id_extrn();

        if (display_init_devices_extrn() != 0) {
                printf("Unable to initialise devices\n");
                abort();
        } else {
                debug_msg("Found %d display devices\n",
                          display_get_device_count_extrn());
        }
        for (i = 0; i < display_get_device_count_extrn(); i++) {
                dt = display_get_device_details_extrn(i);
                if (strcmp(requested_display, dt->name) == 0) {
                        id = dt->id;
                        debug_msg("Found device\n");
                        break;
                } else {
                        debug_msg("Device %s does not match %s\n", dt->name,
                                  requested_display);
                }
        }
        if(i == display_get_device_count_extrn()) {
                fprintf(stderr, "WARNING: Selected '%s' display card "
                        "was not found.\n", requested_display);
                return -1;
        }
        display_free_devices_extrn();

        int ret = display_init_extrn(id, fmt, flags, &d);
        *out = d;
        return ret;
}

void destroy_decoder(struct vcodec_state *video_decoder_state) {
        if(!video_decoder_state) {
                return;
        }

        simple_linked_list_destroy(video_decoder_state->messages);
        decoder_destroy_extrn(video_decoder_state->decoder);

        free(video_decoder_state);
}

