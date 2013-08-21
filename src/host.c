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
#include "rtp/video_decoders.h"
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
const char *sage_receiver = NULL;
volatile bool should_exit_receiver = false;

bool verbose = false;

int initialize_video_capture(struct module *parent,
                const struct vidcap_params *params,
                struct vidcap **state)
{
        struct vidcap_type *vt;
        vidcap_id_t id = 0;
        int i;

        if(!strcmp(vidcap_params_get_driver(params), "none"))
                id = vidcap_get_null_device_id();

        // locking here is because listing of the devices is not really thread safe
        pthread_mutex_t *vidcap_lock = rm_acquire_shared_lock("VIDCAP_LOCK");
        pthread_mutex_lock(vidcap_lock);

        vidcap_init_devices();
        for (i = 0; i < vidcap_get_device_count(); i++) {
                vt = vidcap_get_device_details(i);
                if (strcmp(vt->name, vidcap_params_get_driver(params)) == 0) {
                        id = vt->id;
                        break;
                }
        }
        if(i == vidcap_get_device_count()) {
                fprintf(stderr, "WARNING: Selected '%s' capture card "
                        "was not found.\n", vidcap_params_get_driver(params));
                return -1;
        }
        vidcap_free_devices();

        pthread_mutex_unlock(vidcap_lock);
        rm_release_shared_lock("VIDCAP_LOCK");

        return vidcap_init(parent, id, params, state);
}

int initialize_video_display(const char *requested_display,
                                                char *fmt, unsigned int flags,
                                                struct display **out)
{
        display_type_t *dt;
        display_id_t id = 0;
        int i;

        if(!strcmp(requested_display, "none"))
                 id = display_get_null_device_id();

        if (display_init_devices() != 0) {
                printf("Unable to initialise devices\n");
                abort();
        } else {
                debug_msg("Found %d display devices\n",
                          display_get_device_count());
        }
        for (i = 0; i < display_get_device_count(); i++) {
                dt = display_get_device_details(i);
                if (strcmp(requested_display, dt->name) == 0) {
                        id = dt->id;
                        debug_msg("Found device\n");
                        break;
                } else {
                        debug_msg("Device %s does not match %s\n", dt->name,
                                  requested_display);
                }
        }
        if(i == display_get_device_count()) {
                fprintf(stderr, "WARNING: Selected '%s' display card "
                        "was not found.\n", requested_display);
                return -1;
        }
        display_free_devices();

        return display_init(id, fmt, flags, out);
}

