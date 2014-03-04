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
#include "pdb.h"
#include "rtp/rtp.h"
#include "rtp/rtp_callback.h"
#include "rtp/video_decoders.h"
#include "rtp/pbuf.h"

long long bitrate = 0;
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

int rxtx_mode; // MODE_SENDER, MODE_RECEIVER or both

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
                const char *fmt, unsigned int flags,
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

void display_buf_increase_warning(int size)
{
        fprintf(stderr, "\n***\n"
                        "Unable to set buffer size to %d B.\n"
                        "Please set net.core.rmem_max value to %d or greater. (see also\n"
                        "https://www.sitola.cz/igrid/index.php/Setup_UltraGrid)\n"
#ifdef HAVE_MACOSX
                        "\tsysctl -w kern.ipc.maxsockbuf=%d\n"
                        "\tsysctl -w net.inet.udp.recvspace=%d\n"
#else
                        "\tsysctl -w net.core.rmem_max=%d\n"
#endif
                        "To make this persistent, add these options (key=value) to /etc/sysctl.conf\n"
                        "\n***\n\n",
                        size, size,
#ifdef HAVE_MACOSX
                        size * 4,
#endif /* HAVE_MACOSX */
                        size);

}

struct rtp **initialize_network(const char *addrs, int recv_port_base,
                int send_port_base, struct pdb *participants, bool use_ipv6,
                const char *mcast_if)
{
        struct rtp **devices = NULL;
        double rtcp_bw = 5 * 1024 * 1024;       /* FIXME */
        int ttl = 255;
        char *saveptr = NULL;
        char *addr;
        char *tmp;
        int required_connections, index;
        int recv_port = recv_port_base;
        int send_port = send_port_base;

        tmp = strdup(addrs);
        if(strtok_r(tmp, ",", &saveptr) == NULL) {
                free(tmp);
                return NULL;
        }
        else required_connections = 1;
        while(strtok_r(NULL, ",", &saveptr) != NULL)
                ++required_connections;

        free(tmp);
        tmp = strdup(addrs);

        devices = (struct rtp **)
                malloc((required_connections + 1) * sizeof(struct rtp *));

        for(index = 0, addr = strtok_r(tmp, ",", &saveptr);
                index < required_connections;
                ++index, addr = strtok_r(NULL, ",", &saveptr), recv_port += 2, send_port += 2)
        {
                /* port + 2 is reserved for audio */
                if (recv_port == recv_port_base + 2)
                        recv_port += 2;
                if (send_port == send_port_base + 2)
                        send_port += 2;

                devices[index] = rtp_init_if(addr, mcast_if, recv_port,
                                send_port, ttl, rtcp_bw, FALSE,
                                rtp_recv_callback, (uint8_t *)participants,
                                use_ipv6);
                if (devices[index] != NULL) {
                        rtp_set_option(devices[index], RTP_OPT_WEAK_VALIDATION,
                                TRUE);
                        rtp_set_sdes(devices[index], rtp_my_ssrc(devices[index]),
                                RTCP_SDES_TOOL,
                                PACKAGE_STRING, strlen(PACKAGE_STRING));

                        int size = INITIAL_VIDEO_RECV_BUFFER_SIZE;
                        int ret = rtp_set_recv_buf(devices[index], INITIAL_VIDEO_RECV_BUFFER_SIZE);
                        if(!ret) {
                                display_buf_increase_warning(size);
                        }

                        rtp_set_send_buf(devices[index], 1024 * 56);

                        pdb_add(participants, rtp_my_ssrc(devices[index]));
                }
                else {
                        int index_nest;
                        for(index_nest = 0; index_nest < index; ++index_nest) {
                                rtp_done(devices[index_nest]);
                        }
                        free(devices);
                        devices = NULL;
                }
        }
        if(devices != NULL) devices[index] = NULL;
        free(tmp);

        return devices;
}

void destroy_rtp_devices(struct rtp ** network_devices)
{
        struct rtp ** current = network_devices;
        if(!network_devices)
                return;
        while(*current != NULL) {
                rtp_done(*current++);
        }
        free(network_devices);
}

