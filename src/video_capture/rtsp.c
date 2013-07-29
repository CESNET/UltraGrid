#include "config.h"
#include "config_unix.h"
#include "config_win32.h"

#include "debug.h"
#include "host.h"
#include "tv.h"
#include "rtp/rtp.h"
#include "rtp/rtp_callback.h"
#include "rtp/rtpdec_h264.h"

#include "pdb.h"
#include "rtp/pbuf.h"

#include "video.h"
#include "video_codec.h"
#include "video_capture.h"
#include "video_capture/rtsp.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <netinet/in.h>
#include <net/if.h>
//#include "audio/audio.h"

FILE *F_video_rtsp=NULL;

struct rtsp_state {
//	char *ethX;

	char *nals;
	int nals_size;

	struct timeval t0,t;
	int frames;
	struct video_frame *frame;
	struct tile *tile;
	struct audio_frame audio;
	int width;
	int height;

	struct recieved_data *rx_data[2];
	int rx_index;
	bool new_frame;
	bool pbuf_removed;

	struct rtp *device;
	struct pdb *participants;
	struct pdb_e *cp;
	double rtcp_bw;
	int ttl;
	char *addr;
	char *mcast_if;
	struct timeval curr_time;
	struct timeval timeout;
	struct timeval prev_time;
	struct timeval start_time;
	int required_connections;
	uint32_t timestamp;

	int play_audio_frame;

	struct timeval last_audio_time;
	unsigned int grab_audio:1;

	pthread_t rtsp_thread_id; //the worker_id
//	pthread_t keepalive;
    pthread_mutex_t lock;
    pthread_cond_t worker_cv;
    volatile bool worker_waiting;
    pthread_cond_t boss_cv;
    volatile bool boss_waiting;

	volatile bool should_exit;

	CURL *curl;
	char *uri;
	int port;
};

struct rtsp_state *s_global;
void * vidcap_rtsp_thread(void *args);
char *get_ip_from_ethX(char * ethX);

void * vidcap_rtsp_thread(void *arg)
{
	struct rtsp_state *s;
	s = (struct rtsp_state *)arg;

	gettimeofday(&s->start_time, NULL);
	gettimeofday(&s->prev_time, NULL);

	while(!s->should_exit){
		gettimeofday(&s->curr_time, NULL);
		s->timestamp = tv_diff(s->curr_time, s->start_time) * 90000;

		if(tv_diff(s->curr_time, s->prev_time) >= 30){
			rtsp_get_parameters(s->curl, s->uri);
			gettimeofday(&s->prev_time, NULL);
		}

		rtp_update(s->device, s->curr_time);
		rtp_send_ctrl(s->device, s->timestamp, 0, s->curr_time);

		s->timeout.tv_sec = 0;
		s->timeout.tv_usec = 10000;

		rtp_recv_r(s->device, &s->timeout, s->timestamp);

		pdb_iter_t it;
		s->cp = pdb_iter_init(s->participants, &it);

		int ret;

		s->pbuf_removed = true;

		while (s->cp != NULL ) {
			ret = pbuf_decode(s->cp->playout_buffer, s->curr_time, decode_frame_h264, s->rx_data[s->rx_index]);
			//printf("DECODE return value: %d\n", ret);

			if (ret) {

                pthread_mutex_lock(&s->lock);
                {
                        while(s->new_frame && !s->should_exit) {
                                s->worker_waiting = true;
                                pthread_cond_wait(&s->worker_cv, &s->lock);
                                s->worker_waiting = false;
                        }
                        s->new_frame = true;

                        if(s->boss_waiting)
                                pthread_cond_signal(&s->boss_cv);
                }
                pthread_mutex_unlock(&s->lock);

			} else {
				//printf("FRAME not ready yet\n");
				//s->new_frame=false;

			}
			pbuf_remove(s->cp->playout_buffer, s->curr_time);

			s->cp = pdb_iter_next(&it);
		}

		pdb_iter_done(&it);

	}
	return NULL;
}

struct video_frame	*vidcap_rtsp_grab(void *state, struct audio_frame **audio){
	struct rtsp_state *s;
	s = (struct rtsp_state *)state;

	*audio = NULL;

    pthread_mutex_lock(&s->lock);
    {
            while(!s->new_frame) {
                    s->boss_waiting = true;
                    pthread_cond_wait(&s->boss_cv, &s->lock);
                    s->boss_waiting = false;
            }

            if(s->worker_waiting) {
                    pthread_cond_signal(&s->worker_cv);
            }
    }
    pthread_mutex_unlock(&s->lock);

	gettimeofday(&s->curr_time, NULL);

	//s->frame->tiles[0].data = s->rx_data[s->rx_index]->frame_buffer[0];
	s->frame->tiles[0].data_len = s->rx_data[s->rx_index]->buffer_len[0];

	char *data = malloc(s->rx_data[s->rx_index]->buffer_len[0] + s->nals_size);
	if(data !=NULL){
		//printf("\n[rtsp] data no NULL with nal size of %d bytes\n",s->nals_size);

		memcpy(data,s->nals,s->nals_size);
		memcpy(data+s->nals_size,s->rx_data[s->rx_index]->frame_buffer[0],s->rx_data[s->rx_index]->buffer_len[0]);

		memcpy(s->frame->tiles[0].data,data,s->rx_data[s->rx_index]->buffer_len[0] + s->nals_size);
		s->frame->tiles[0].data_len += s->nals_size;

		free(data);
	}
//	//MODUL DE CAPTURA AUDIO A FITXER PER COMPROVACIONS EN TX
//	//CAPTURA FRAMES ABANS DE DESCODIFICAR PER COMPROVAR RECEPCIÓ.
//	if (F_video_rtsp == NULL) {
//		printf("[rtsp] recording rtsp rx frame...\n");
//		F_video_rtsp = fopen("rtsp_rx_frame.mp4", "wb");
//	}
//
//	fwrite(s->frame->tiles[0].data, s->frame->tiles[0].data_len, 1,F_video_rtsp);
//	//FI CAPTURA

	s->rx_index = (s->rx_index + 1) %2;
	s->new_frame = false;

//	pbuf_remove(s->cp->playout_buffer,s->curr_time_cpy);
//	gettimeofday(&s->curr_time_cpy, NULL);

    gettimeofday(&s->t, NULL);
	double seconds = tv_diff(s->t, s->t0);
	if (seconds >= 5) {
		float fps = s->frames / seconds;
		fprintf(stderr, "[rtsp capture] %d frames in %g seconds = %g FPS\n",
				s->frames, seconds, fps);
		s->t0 = s->t;
		s->frames = 0;
	}
	s->frames++;


	return s->frame;
}

void *vidcap_rtsp_init(char *fmt, unsigned int flags){

    struct rtsp_state *s;

    s = calloc(1, sizeof(struct rtsp_state));
    if (!s)
            return NULL;
    s_global = s;

    char *save_ptr = NULL;

    gettimeofday(&s->t0, NULL);
    s->frames=0;
    s->nals = malloc(1024);

    s->device = NULL;
    s->rtcp_bw = 5 * 1024 * 1024; /* FIXME */
    s->ttl  = 255;

    s->mcast_if = NULL;
    s->required_connections = 1;

    s->timeout.tv_sec = 0;
    s->timeout.tv_usec = 10000;

    s->device = (struct rtp *) malloc((s->required_connections) * sizeof(struct rtp *));
    s->participants = pdb_init();

    s->rx_data[0] = calloc(1, sizeof(struct recieved_data));
    s->rx_data[1] = calloc(1, sizeof(struct recieved_data));
    s->rx_index=0;
    s->new_frame = false;

    s->uri = NULL;
    s->curl = NULL;

    if (fmt == NULL || strcmp(fmt, "help") == 0) {
		printf("rtsp options:\n");
		printf("\t-t rtsp:<uri>:<port>:<width>:<height>\n");
		printf("\t\t <uri> uri server without 'rtsp://' \n");
		printf("\t\t <port> receiver port \n");
		printf("\t\t <width> receiver width \n");
		printf("\t\t <height> receiver height \n");
//		printf("\t\t <ethX> receiver network interface\n");
		//show_codec_help("rtsp");
		return &vidcap_init_noerr;
	}
    char *tmp;
	tmp = strtok_r(fmt, ":", &save_ptr);
	if (!tmp) {
		fprintf(stderr, "[rtsp] Wrong format for rtsp '%s' - no rtsp server specified\n", tmp);
		free(s);
		return NULL;
	}
	s->uri = malloc(strlen(tmp) + 32);
	sprintf(s->uri, "rtsp://%s",tmp);

	tmp = strtok_r(NULL, ":", &save_ptr);
	if (!tmp) {
		fprintf(stderr, "[rtsp] Wrong format for rtsp '%s' - no receiver port specified\n", tmp);
		free(s);
		return NULL;
	}
	s->port = atoi(tmp);

	tmp = strtok_r(NULL, ":", &save_ptr);
	if (!tmp) {
		fprintf(stderr, "[rtsp] Wrong format for rtsp '%s' - no receiver width specified\n", tmp);
		free(s);
		return NULL;
	}
	s->width = atoi(tmp);

	tmp = strtok_r(NULL, ":", &save_ptr);
	if (!tmp) {
		fprintf(stderr, "[rtsp] Wrong format for rtsp '%s' - no receiver height specified\n", tmp);
		free(s);
		return NULL;
	}
	s->height = atoi(tmp);
//	tmp = strtok_r(NULL, ":", &save_ptr);
//	if (!tmp) {
//		fprintf(stderr, "[rtsp] Wrong format for rtsp '%s' - no receiver interface specified\n", tmp);
//		free(s);
//		return NULL;
//	}
//	s->ethX = malloc(strlen(tmp) + 32);
//	sprintf(s->ethX, "%s",tmp);
//	s->addr=get_ip_from_ethX(s->ethX);//"127.0.0.1";
//	printf("\n[rtsp] network interface %s selected with IP %s\n",s->ethX,s->addr);
	s->addr="127.0.0.1";

    s->frame = vf_alloc(1);
    s->tile = vf_get_tile(s->frame, 0);
    vf_get_tile(s->frame, 0)->width=s->width;
    vf_get_tile(s->frame, 0)->height=s->height;
    s->frame->fps=30;
    s->frame->color_spec=H264;  //TODO Configure from SDP
    s->frame->interlacing=PROGRESSIVE;
    s->frame->tiles[0].data = calloc(1, s->width*s->height);

	s->should_exit = false;

    s->device = rtp_init_if(s->addr, s->mcast_if, s->port, s->port, s->ttl, s->rtcp_bw, 0, rtp_recv_callback, (void *)s->participants, 0);

    if (s->device != NULL) {
        //printf("[rtsp] RTP INIT OPTIONS\n");
        if (!rtp_set_option(s->device, RTP_OPT_WEAK_VALIDATION, 1)) {
            printf("[rtsp] RTP INIT - set option\n");
            return NULL;
        }
        if (!rtp_set_sdes(s->device, rtp_my_ssrc(s->device),
                RTCP_SDES_TOOL, PACKAGE_STRING, strlen(PACKAGE_STRING))) {
            printf("[rtsp] RTP INIT - set sdes\n");
            return NULL;
        }

        int ret = rtp_set_recv_buf(s->device, INITIAL_VIDEO_RECV_BUFFER_SIZE);
        if (!ret) {
            printf("[rtsp] RTP INIT - set recv buf \nset command: sudo sysctl -w net.core.rmem_max=9123840\n");
            return NULL;
        }

        if (!rtp_set_send_buf(s->device, 1024 * 56)) {
            printf("[rtsp] RTP INIT - set send buf\n");
            return NULL;
        }
        ret=pdb_add(s->participants, rtp_my_ssrc(s->device));
        //printf("[rtsp] [PDB ADD] returned result = %d for ssrc = %x\n",ret,rtp_my_ssrc(s->devices[s->index]));
    }

    printf("[rtsp] rtp receiver init done\n");

    pthread_mutex_init(&s->lock, NULL);
    pthread_cond_init(&s->boss_cv, NULL);
    pthread_cond_init(&s->worker_cv, NULL);

    s->boss_waiting = false;
    s->worker_waiting = false;

	const struct sigaction action = { .sa_handler = (void *)&sigaction_handler };
	sigaction(SIGINT, &action, NULL);

	s->nals_size = init_rtsp(s->uri, s->port, s , s->nals);

    pthread_create(&s->rtsp_thread_id, NULL , vidcap_rtsp_thread , s);

    printf("[rtsp] rtsp capture init done\n");

    return s;
}

int init_rtsp(char* rtsp_uri, int rtsp_port,void *state, unsigned char* nals) {
	struct rtsp_state *s;
	s = (struct rtsp_state *)state;
	const char *range = "0.000-";
	int len_nals=0;
	char *base_name = NULL;
	CURLcode res;
	//printf("\n[rtsp] request %s\n", VERSION_STR);
	//printf("    Project web site: http://code.google.com/p/rtsprequest/\n");
	//printf("    Requires cURL V7.20 or greater\n\n");
	const char *url = rtsp_uri;
	char *uri = malloc(strlen(url) + 32);
	char *sdp_filename = malloc(strlen(url) + 32);
	char *control = malloc(strlen(url) + 32);
	//unsigned char nals[1500];
	//bzero(nals, 1500);
	char transport[256];
	bzero(transport, 256);
	int port = rtsp_port;
	get_sdp_filename(url, sdp_filename);
	sprintf(transport, "RTP/AVP;unicast;client_port=%d-%d", port, port+1);

	/* initialize curl */
	res = curl_global_init(CURL_GLOBAL_ALL);
	if (res == CURLE_OK) {
	    curl_version_info_data *data = curl_version_info(CURLVERSION_NOW);
	    CURL *curl;
	    fprintf(stderr, "[rtsp]    cURL V%s loaded\n", data->version);

	    /* initialize this curl session */
	    curl = curl_easy_init();
	    if (curl != NULL) {
	    	my_curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1);  //This tells curl not to use any functions that install signal handlers or cause signals to be sent to your process.
			my_curl_easy_setopt(curl, CURLOPT_VERBOSE, 0L);
			my_curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 1L);
			my_curl_easy_setopt(curl, CURLOPT_WRITEHEADER, stdout);
			my_curl_easy_setopt(curl, CURLOPT_URL, url);

			sprintf(uri, "%s", url);

			s->curl = curl;
			s->uri = uri;

			//TODO TO CHECK CONFIGURING ERRORS
			//CURLOPT_ERRORBUFFER
			//http://curl.haxx.se/libcurl/c/curl_easy_perform.html

			/* request server options */
			rtsp_options(curl, uri);
			//printf("sdp_file: %s\n", sdp_filename);
			/* request session description and write response to sdp file */
			rtsp_describe(curl, uri, sdp_filename);
			//printf("sdp_file!!!!: %s\n", sdp_filename);
			/* get media control attribute from sdp file */
			get_media_control_attribute(sdp_filename, control);

			/* setup media stream */
			//sprintf(uri, "%s/%s", url, "track1");
			sprintf(uri, "%s/%s", url, control);

			rtsp_setup(curl, uri, transport);

			/* start playing media stream */
			sprintf(uri, "%s/", url);

			rtsp_play(curl, uri, range);

			/* get start nal size attribute from sdp file */
			len_nals=get_nals(sdp_filename, nals);

			s->curl = curl;
			s->uri = uri;
			printf("[rtsp] playing video from server...\n");

	    } else {
	    	fprintf(stderr, "[rtsp] curl_easy_init() failed\n");
	    }
	    curl_global_cleanup();
	} else {
	    fprintf(stderr, "[rtsp] curl_global_init(%s) failed: %d\n",	    "CURL_GLOBAL_ALL", res);
	}
	return len_nals;
}

static void rtsp_get_parameters(CURL *curl, const char *uri)
{
    CURLcode res = CURLE_OK;
    printf("\n[rtsp] GET_PARAMETERS %s\n", uri);
    my_curl_easy_setopt(curl, CURLOPT_RTSP_STREAM_URI, uri);
    my_curl_easy_setopt(curl, CURLOPT_RTSP_REQUEST, (long)CURL_RTSPREQ_GET_PARAMETER);
    my_curl_easy_perform(curl);
}

/* send RTSP OPTIONS request */
static void rtsp_options(CURL *curl, const char *uri)
{
    CURLcode res = CURLE_OK;
    printf("\n[rtsp] OPTIONS %s\n", uri);
    my_curl_easy_setopt(curl, CURLOPT_RTSP_STREAM_URI, uri);
    my_curl_easy_setopt(curl, CURLOPT_RTSP_REQUEST, (long)CURL_RTSPREQ_OPTIONS);
    my_curl_easy_perform(curl);
}


/* send RTSP DESCRIBE request and write sdp response to a file */
static void rtsp_describe(CURL *curl, const char *uri, const char *sdp_filename)
{
    CURLcode res = CURLE_OK;
    FILE *sdp_fp = fopen(sdp_filename, "wt");
    printf("\n[rtsp] DESCRIBE %s\n", uri);
    if (sdp_fp == NULL) {
        fprintf(stderr, "Could not open '%s' for writing\n", sdp_filename);
        sdp_fp = stdout;
    }
    else {
       // printf("Writing SDP to '%s'\n", sdp_filename);
    }
    my_curl_easy_setopt(curl, CURLOPT_WRITEDATA, sdp_fp);
    my_curl_easy_setopt(curl, CURLOPT_RTSP_REQUEST, (long)CURL_RTSPREQ_DESCRIBE);
    my_curl_easy_perform(curl);
    my_curl_easy_setopt(curl, CURLOPT_WRITEDATA, stdout);
    if (sdp_fp != stdout) {
        fclose(sdp_fp);
    }
}

/* send RTSP SETUP request */
static void rtsp_setup(CURL *curl, const char *uri, const char *transport)
{
    CURLcode res = CURLE_OK;
    printf("\n[rtsp] SETUP %s\n", uri);
    printf("\t TRANSPORT %s\n", transport);
    my_curl_easy_setopt(curl, CURLOPT_RTSP_STREAM_URI, uri);
    my_curl_easy_setopt(curl, CURLOPT_RTSP_TRANSPORT, transport);
    my_curl_easy_setopt(curl, CURLOPT_RTSP_REQUEST, (long)CURL_RTSPREQ_SETUP);
    my_curl_easy_perform(curl);
}


/* send RTSP PLAY request */
static void rtsp_play(CURL *curl, const char *uri, const char *range)
{
    CURLcode res = CURLE_OK;
    printf("\n[rtsp] PLAY %s\n", uri);
    my_curl_easy_setopt(curl, CURLOPT_RTSP_STREAM_URI, uri);
    //my_curl_easy_setopt(curl, CURLOPT_RANGE, range);
    my_curl_easy_setopt(curl, CURLOPT_RTSP_REQUEST, (long)CURL_RTSPREQ_PLAY);
    my_curl_easy_perform(curl);
}


/* send RTSP TEARDOWN request */
static void rtsp_teardown(CURL *curl, const char *uri)
{
    CURLcode res = CURLE_OK;
    printf("\n[rtsp] TEARDOWN %s\n", uri);
    my_curl_easy_setopt(curl, CURLOPT_RTSP_REQUEST, (long)CURL_RTSPREQ_TEARDOWN);
    my_curl_easy_perform(curl);
}


/* convert url into an sdp filename */
static void get_sdp_filename(const char *url, char *sdp_filename)
{
    const char *s = strrchr(url, '/');
   // printf("sdp_file get: %s\n", sdp_filename);


    if (s != NULL) {
        s++;
        if (s[0] != '\0') {
            sprintf(sdp_filename, "%s.sdp", s);
        }
    }
}


/* scan sdp file for media control attribute */
static void get_media_control_attribute(const char *sdp_filename,
                                        char *control)
{
    int max_len = 1256;
    char *s = malloc(max_len);
    FILE *sdp_fp = fopen(sdp_filename, "rt");
    control[0] = '\0';
    if (sdp_fp != NULL) {
        while (fgets(s, max_len - 2, sdp_fp) != NULL) {
            sscanf(s, " a = control: %s", control);
		//printf("el control: %s\n", control);
        }

        fclose(sdp_fp);
    }
    free(s);
}
char *get_ip_from_ethX(char * ethX){
	int fd;
	struct ifreq ifr;

	fd = socket(AF_INET, SOCK_DGRAM, 0);

	/* I want to get an IPv4 IP address */
	ifr.ifr_addr.sa_family = AF_INET;

	/* I want IP address attached to "ethX" */
	strncpy(ifr.ifr_name, ethX, IFNAMSIZ - 1);

	ioctl(fd, SIOCGIFADDR, &ifr);

	close(fd);

	return inet_ntoa(((struct sockaddr_in *) &ifr.ifr_addr)->sin_addr);
}

struct vidcap_type	*vidcap_rtsp_probe(void){
    struct vidcap_type *vt;

    vt = (struct vidcap_type *)malloc(sizeof(struct vidcap_type));
    if (vt != NULL) {
            vt->id = VIDCAP_RTSP_ID;
            vt->name = "rtsp";
            vt->description = "Video capture from RTSP remote server";
    }
    return vt;
}


void vidcap_rtsp_done(void *state){
    struct rtsp_state *s = state;

    s->should_exit = true;
    pthread_join(s->rtsp_thread_id, NULL);
    //pthread_join(s->keepalive, NULL);

    rtsp_teardown(s->curl, s->uri);

	curl_easy_cleanup(s->curl);
	s->curl = NULL;

	rtp_done(s->device);

//    free(s->data);

//    free(s->audio_tone);
//    free(s->audio_silence);
    free(s);
}

/* scan sdp file for media control attribute */
static int get_nals(const char *sdp_filename, unsigned char *nals){
    int max_len = 1500 , len_nals = 0;
    char *s = malloc(max_len);
    char *sprop;
    bzero(s, max_len);
    FILE *sdp_fp = fopen(sdp_filename, "rt");
    nals[0] = '\0';
    if (sdp_fp != NULL) {
        while (fgets(s, max_len - 2, sdp_fp) != NULL) {
		sprop = strstr(s, "sprop-parameter-sets=");
		if (sprop != NULL) {
			int length;
			char *nal_aux, *nals_aux, *nal;
			nals_aux = malloc(max_len);
			nals[0]=0x00;
			nals[1]=0x00;
			nals[2]=0x00;
			nals[3]=0x01;
			len_nals = 4;
			nal_aux = strstr(sprop, "=") ;
			nal_aux++;
			nal = strtok(nal_aux, ",");
			//convert base64 to hex
			nals_aux = g_base64_decode(nal, &length);
			memcpy(nals+len_nals, nals_aux, length);
			len_nals += length;

			while ((nal = strtok(NULL, ",")) != NULL){
				//convert base64 to hex
				nals[len_nals]=0x00;
				nals[len_nals+1]=0x00;
				nals[len_nals+2]=0x00;
				nals[len_nals+3]=0x01;
				len_nals += 4;
				nals_aux = g_base64_decode(nal, &length);
				memcpy(nals+len_nals, nals_aux, length);
				len_nals += length;
				}
			}

        	}

        fclose(sdp_fp);
    }

    free(s);
    return len_nals;
}


/* forced sigaction handler when rtsp init waits... */
void sigaction_handler()
{
	vidcap_rtsp_done(s_global);
}
