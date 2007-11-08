/*
 * FILE:    auddev_alsa.c
 * PROGRAM: RAT ALSA 0.9+/final audio driver.
 * AUTHOR:  Steve Smith
 *
 * Copyright (c) 2003 University of Sydney
 * Distributed under the same terms as RAT itself.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 */

#define ALSA_PCM_NEW_HW_PARAMS_API
#define ALSA_PCM_NEW_SW_PARAMS_API
#include <alsa/asoundlib.h>

#include "config_unix.h"
#include "audio_types.h"
#include "audio_hw/linux_alsa.h"
#include "debug.h"


/*
 * Structure that keeps track of the cards we know about. Rat wants a linear
 * list of cards, so we'll give it that.
 *
 * This is filled in during the init step.
 */

typedef struct RatCardInfo_t
{
    char *name;
    int card_number;
    char *pcm_device;
} RatCardInfo;

#define MAX_RAT_CARDS 128

static RatCardInfo ratCards[MAX_RAT_CARDS];
static int nRatCards = 0;

/* Define ALSA mixer identifiers.  We use these to lookup the volume
 * controls on the mixer.  This feels like a bit of a hack, but
 * appears to be what everybody else uses. */
#define RAT_ALSA_MIXER_PCM_NAME "PCM"
#define RAT_ALSA_MIXER_CAPTURE_NAME "Capture"
#define RAT_ALSA_MIXER_LINE_NAME "Line"
#define RAT_ALSA_MIXER_MIC_NAME "Mic"
#define RAT_ALSA_MIXER_CD_NAME "CD"

/*
 * The list of input ports to choose between.  These are scanned from the
 * mixer device.
 */
#define MAX_RAT_DEVICES 32

typedef struct _port_t {
    audio_port_details_t details;
    snd_mixer_elem_t* mixer;
    int priority;
} port_t;

static port_t iports[MAX_RAT_DEVICES];
static unsigned num_iports;

/*
 * All output passes through the pcm device, so we only have a single port
 * here.
 * FIXME: There are some cards that don't have a PCM control, only a master.
 * It is assumed that these are unlikely to occur in real RAT usage.
 */
static audio_port_details_t out_port = {
    0, RAT_ALSA_MIXER_PCM_NAME
};

/*
 * Current open audio device
 */

typedef struct _pcm_stream_t {
    snd_pcm_t *handle;
    snd_pcm_uframes_t buffer_size;
    snd_pcm_uframes_t period_size;
    int channels;
} pcm_stream_t;

static struct current_t {
    int index;
    RatCardInfo *info;
    unsigned bytes_per_block;
    pcm_stream_t tx;
    pcm_stream_t rx;
    audio_port_t iport;
    snd_mixer_t *mixer;
    snd_mixer_elem_t *txgain;
    snd_mixer_elem_t *rxgain;
    audio_port_t txport;
} current;

static void clear_current()
{
    current.tx.handle = NULL;
    current.rx.handle = NULL;
    current.index = -1;
    current.info = NULL;
}


/*
 * Utility funcs
 */
static char *encodingToString[] = {
    "PCMU",
    "PCMA",
    "S8",
    "U8",
    "S16"
};

// Ah, exceptions, I hardly new thee ...
#define CHECKERR(msg) \
{ \
  if (err < 0) \
  { \
    fprintf(stderr, msg ": %s\n", snd_strerror(err)); \
    return FALSE; \
  } \
    }
#define CHECKERRCONT(msg) \
{ \
  if (err < 0) \
    fprintf(stderr, msg ": %s\n", snd_strerror(err)); \
    }
#define VCHECKERR(msg) \
{ \
  if (err < 0) \
  { \
    fprintf(stderr, msg ": %s\n", snd_strerror(err)); \
    return; \
  } \
}


static int mapformat(deve_e encoding)
{
    int format = -1;
    switch (encoding)
    {
    case DEV_PCMU:
        format = SND_PCM_FORMAT_MU_LAW;
        break;
      case DEV_PCMA:
        format = SND_PCM_FORMAT_A_LAW;
	break;
    case DEV_S8:
        format = SND_PCM_FORMAT_S8;
	break;
    case DEV_U8:
        format = SND_PCM_FORMAT_U8;
	break;
    case DEV_S16:
        format = SND_PCM_FORMAT_S16;
	break;
    }
    return format;
}

__attribute__((unused)) static char* mapstate(snd_pcm_state_t s)
{
    switch (s) {
      case SND_PCM_STATE_OPEN:
        return "SND_PCM_STATE_OPEN";
      case SND_PCM_STATE_SETUP:
        return "SND_PCM_STATE_SETUP";
      case SND_PCM_STATE_PREPARED:
        return "SND_PCM_STATE_PREPARED";
      case SND_PCM_STATE_RUNNING:
        return "SND_PCM_STATE_RUNNING";
      case SND_PCM_STATE_XRUN:
        return "SND_PCM_STATE_XRUN";
      case SND_PCM_STATE_DRAINING:
        return "SND_PCM_STATE_DRAINING";
      case SND_PCM_STATE_PAUSED:
        return "SND_PCM_STATE_PAUSED";
      case SND_PCM_STATE_SUSPENDED:
        return "SND_PCM_STATE_SUSPENDED";
      default:
        return "UNKNOWN!!!";
    }
}

static void dump_audio_format(audio_format *f)
{
    if (f == 0)
	debug_msg("    <null>\n");
    else
        debug_msg("encoding=%s sample_rate=%d bits_per_sample=%d "
                  "channels=%d bytes_per_block=%d\n",
                  encodingToString[f->encoding],
                  f->sample_rate, f->bits_per_sample,
		  f->channels, f->bytes_per_block);
}


static void  __attribute__((unused)) dump_alsa_current(snd_pcm_t *handle)
{
    int err;
    snd_output_t *out;

    err = snd_output_stdio_attach(&out, stderr, 0);
    snd_output_printf(out, "--- MY IO\n");

    err = snd_pcm_dump_setup(handle, out);

    snd_output_printf(out, "--- SW\n");
    err = snd_pcm_dump_sw_setup(handle, out);

    snd_output_printf(out, "--- HW\n");
    err = snd_pcm_dump_hw_setup(handle, out);

    snd_output_printf(out, "--- DONE\n");
    snd_output_close(out);
    }


/* *** Alsa driver implementation. *** */

#undef CHECKOPENERR
#define CHECKOPENERR(msg) \
{ \
  if (err < 0) \
  { \
    fprintf(stderr, msg ": %s\n", snd_strerror(err)); \
    snd_pcm_close(stream->handle); \
    return FALSE; \
  } \
}

static int open_stream(RatCardInfo *info, pcm_stream_t *stream,
                       snd_pcm_stream_t type, audio_format *fmt)
    {
    int err;
    size_t bsize;
    snd_pcm_uframes_t frames;
    unsigned int rrate;
    snd_pcm_hw_params_t *hw_params;
    snd_pcm_sw_params_t *sw_params;

    err = snd_pcm_open(&stream->handle, info->pcm_device,
                       type, SND_PCM_NONBLOCK);
    CHECKERR("Card open failed");

    snd_pcm_hw_params_alloca (&hw_params);

    err = snd_pcm_hw_params_any (stream->handle, hw_params);
    CHECKOPENERR("Failed to initialise HW parameters");

    err = snd_pcm_hw_params_set_access(stream->handle, hw_params,
                                       SND_PCM_ACCESS_RW_INTERLEAVED);
    CHECKOPENERR("Failed to set interleaved access");

    err = snd_pcm_hw_params_set_format (stream->handle, hw_params,
                                        mapformat(fmt->encoding));
    CHECKOPENERR("Failed to set encoding");

    err = snd_pcm_hw_params_set_channels (stream->handle, hw_params,
                                          fmt->channels);
    CHECKOPENERR("Failed to set channels");
    stream->channels = fmt->channels;

    rrate = fmt->sample_rate;
    err = snd_pcm_hw_params_set_rate_near (stream->handle, hw_params,
                                           &rrate, 0);
    CHECKOPENERR("Failed to set sample rate");
    if (rrate != fmt->sample_rate) {
        fprintf(stderr, "ALSA rate set to %d when we wanted %d\n",
                rrate, fmt->sample_rate);
        return FALSE;
    }

    // Setup the buffer size. This stuff's all in frames, BTW.  We can't
    // convert with the helper functions at this point as they require
    // a working handle, and ours isn't setup yet. We don't actually do
    // anything with these values anyway.
    bsize = snd_pcm_format_size (mapformat(fmt->encoding),
                                 fmt->sample_rate / RAT_ALSA_BUFFER_DIVISOR);
    frames = bsize;
    err = snd_pcm_hw_params_set_buffer_size_near(stream->handle, hw_params,
                                                 &frames);
    CHECKOPENERR("Failed to set buffer size");

    stream->buffer_size = frames;
    debug_msg("Buffer == %d\n", stream->buffer_size);

    frames = bsize / 2;
    err = snd_pcm_hw_params_set_period_size_near(stream->handle, hw_params,
                                                 &frames, 0);
    CHECKOPENERR("Failed to set period size");

    stream->period_size = frames;
    debug_msg("Period == %d\n", stream->period_size);

    err = snd_pcm_hw_params (stream->handle, hw_params);
    CHECKOPENERR("Failed to install HW parameters");


    // ALSA software settings
    snd_pcm_sw_params_alloca(&sw_params);
    err = snd_pcm_sw_params_current(stream->handle, sw_params);
    CHECKOPENERR("Failed to initialise SW params");

    err = snd_pcm_sw_params_set_start_threshold(stream->handle, sw_params,
                                                stream->buffer_size);
    CHECKOPENERR("Failed to set threshold value");

    err = snd_pcm_sw_params_set_avail_min(stream->handle, sw_params, 4);
    CHECKOPENERR("Failed to set min available value");

    err = snd_pcm_sw_params_set_xfer_align(stream->handle, sw_params, 1);
    CHECKOPENERR("Failed to set xfer align value");

    err = snd_pcm_sw_params(stream->handle, sw_params);
    CHECKOPENERR("Failed to set SW params");
    
    return TRUE;
}


// Get the mixer

#undef CHECKOPENERR
#define CHECKOPENERR(msg) \
{ \
  if (err < 0) \
  { \
    fprintf(stderr, msg ": %s\n", snd_strerror(err)); \
    snd_mixer_close(current.mixer); \
    return FALSE; \
  } \
}


// Open a named mixer
static int open_volume_ctl(char *name, snd_mixer_elem_t **ctl)
{
    snd_mixer_selem_id_t *sid;
    int err;
	      
    snd_mixer_selem_id_alloca(&sid);

    // FIXME? Find the appropriate mixer element.  This feels really wrong,
    // there has to be another way to do this.
    snd_mixer_selem_id_set_name (sid, name);

    *ctl = snd_mixer_find_selem(current.mixer, sid);
    err = (int)*ctl;
    CHECKOPENERR("Couldn't find mixer control element");

    if (snd_mixer_selem_has_playback_volume(*ctl)) {
        debug_msg("Got volume control %s of type PLAY\n", name);
        // FIXME: Does this always work?
        snd_mixer_selem_set_playback_volume_range (*ctl, 0, 100);

        err = snd_mixer_selem_set_playback_switch_all(*ctl, 1);
        CHECKOPENERR("Failed to switch on playback volume");

    } else if (snd_mixer_selem_has_capture_volume(*ctl)) {
        debug_msg("Got volume control %s of type CAPTURE\n", name);
        snd_mixer_selem_set_capture_volume_range (*ctl, 0, 100);

        err = snd_mixer_selem_set_capture_switch_all(*ctl, 1);
        CHECKOPENERR("Failed to switch on capture volume");

    } else {
        debug_msg("Unknown mixer type %s set\n", name);
        return FALSE;
}

    return TRUE;
}


// Open and initialise the system mixer

static int porder(const void *a, const void *b)
{
    return (((port_t *)a)->priority - ((port_t *)b)->priority);
}

static int setup_mixers()
{
    snd_mixer_elem_t *elem;
    int err;
    unsigned i;
    
    err = snd_mixer_open (&current.mixer, 0);
    CHECKERR("Failed to open the mixer");

    // FIXME: Attach the mixer to the default card.  Is this enough?
    err = snd_mixer_attach(current.mixer, "default");
    CHECKOPENERR("Failed to attach mixer");

    err = snd_mixer_selem_register(current.mixer, NULL, NULL);
    CHECKOPENERR("Failed to register mixer");
    
    err = snd_mixer_load(current.mixer);
    CHECKOPENERR("Failed to load mixer");


    // Get the playback and capture volume controls
    if ((!open_volume_ctl(RAT_ALSA_MIXER_PCM_NAME, &current.txgain)) ||
        (!open_volume_ctl(RAT_ALSA_MIXER_CAPTURE_NAME, &current.rxgain)))
	    return FALSE;	

    num_iports = 0;

    // We now scan the mixer for recording controls.  We're interested in
    // the controls that are part of a group (i.e. radio-button types) that
    // can be flipped between.
    for (elem = snd_mixer_first_elem (current.mixer);
         elem && (num_iports < MAX_RAT_DEVICES);
         elem = snd_mixer_elem_next (elem))
    {
        if (snd_mixer_selem_is_active (elem) &&
            snd_mixer_selem_has_capture_switch(elem) &&
            snd_mixer_selem_has_capture_switch_exclusive(elem))
	{
            // FIXME: It's theoretically possible that there would be more
            // than one capture group, but RAT isn't really equipped to handle
            // the case so we'll just ignore it for now.
            int gid = snd_mixer_selem_get_capture_group(elem);

            const char *name = snd_mixer_selem_get_name(elem);
       
            debug_msg("Got CAPTURE element '%s' of group %d\n", name, gid);

            snprintf(iports[num_iports].details.name, AUDIO_PORT_NAME_LENGTH,
                     "%s", name);
            iports[num_iports].mixer = elem;

            // The principle of least-surprise means that we should present
            // the ports in the same order as the other drivers.  As we're
            // more flexible about retrieving the mixer ports we need to
            // attempt to reorder the list, so we assign a priority and
            // sort the list at the end.
            if (strstr(name, RAT_ALSA_MIXER_MIC_NAME) == name) {
                iports[num_iports].priority = 100;
            } else if (strstr(name, RAT_ALSA_MIXER_LINE_NAME) == name) {
                iports[num_iports].priority = 50;
            } else if (strstr(name, RAT_ALSA_MIXER_CD_NAME) == name) {
                iports[num_iports].priority = 30;
            } else {
                iports[num_iports].priority = 10;
    }


            num_iports++;

        }
      
    }

    qsort(iports, num_iports, sizeof(port_t), porder);

    // Now it's sorted we need to set the port ID to the index, allowing us
    // a fast lookup for port-IDs
    for (i=0; i<num_iports; i++) {
        iports[i].details.port = i;
    }


    return TRUE;
}


int alsa_audio_open(audio_desc_t ad, audio_format *infmt, audio_format *outfmt)
{
    int err;

    debug_msg("Audio open ad == %d\n", ad);
    debug_msg("Input format:\n");
    dump_audio_format(infmt);
    debug_msg("Output format:\n");
    dump_audio_format(outfmt);
    
    if (current.tx.handle != NULL) {
        fprintf(stderr, "Attempt to open a device while another is open\n");
        return FALSE;
    }
    current.index = ad;
    current.info = ratCards + ad;
    current.bytes_per_block = infmt->bytes_per_block;

    if (!open_stream(current.info, &current.tx,
                     SND_PCM_STREAM_PLAYBACK, outfmt)) {
        fprintf(stderr, "Failed to open device for playback\n");
        return FALSE;
    }
    if (!open_stream(current.info, &current.rx,
                     SND_PCM_STREAM_CAPTURE, infmt)) {
        fprintf(stderr, "Failed to open device for capture\n");
        return FALSE;
    }

    setup_mixers();

    err = snd_pcm_prepare(current.tx.handle);
    CHECKERR("Failed to prepare playback");

    err = snd_pcm_start(current.rx.handle);
    CHECKERR("Failed to start PCM capture");

    return TRUE;
}

/*
 * Shutdown.
 */
void alsa_audio_close(audio_desc_t ad)
{
    int err;
    
    if (current.index != ad) {
        fprintf(stderr, "Index to close (%d) doesn't match current(%d)\n",
                ad, current.index);
        return;
    }

    debug_msg("Closing device \"%s\"\n", current.info->name);

    err = snd_pcm_close(current.tx.handle);
    CHECKERRCONT("Error closing playback PCM");

    err = snd_pcm_close(current.rx.handle);
    CHECKERRCONT("Error closing capture PCM");

    // Open a mixer for each of the ports
    err = snd_mixer_close(current.mixer);
    CHECKERRCONT("Error closing mixer");

    clear_current();
}

/*
 * Flush input buffer.
 */
void alsa_audio_drain(audio_desc_t ad __attribute__((unused)))
{
    int err;

    debug_msg("audio_drain\n");
    err = snd_pcm_drain(current.rx.handle);
    VCHECKERR("Problem draining input");
    }
    


/*
 * Set record gain.
 */
void alsa_audio_set_igain(audio_desc_t ad, int gain)
{
    int err;
    debug_msg("Set igain %d %d\n", ad, gain);

    err = snd_mixer_selem_set_capture_volume_all(current.rxgain, gain);
    VCHECKERR("Couldn't set capture volume");
}


/*
 * Get capture gain.
 */
int alsa_audio_get_igain(audio_desc_t ad)
{
    long igain;
    int err;
    debug_msg("Get igain %d\n", ad);

    err = snd_mixer_selem_get_capture_volume(current.rxgain,
                                             SND_MIXER_SCHN_MONO, &igain);
    CHECKERR("Failed to get capture volume");

    return (int)igain;
}

int alsa_audio_duplex(audio_desc_t ad __attribute__((unused)))
{
    return TRUE; // FIXME: ALSA always duplex?
}

/*
 * Set play gain.
 */
void alsa_audio_set_ogain(audio_desc_t ad, int vol)
{
    int err;

    debug_msg("Set igain %d %d\n", ad, vol);

    err = snd_mixer_selem_set_playback_volume_all(current.txgain, vol);
    VCHECKERR("Couldn't set mixer playback volume");

    err = snd_mixer_selem_set_playback_switch_all(current.txgain, 1);
    VCHECKERR("Failed to switch on playback volume");
}

/*
 * Get play gain.
 */
int
alsa_audio_get_ogain(audio_desc_t ad)
{
    long ogain;
    int err;

    debug_msg("Get igain %d\n", ad);
    err = snd_mixer_selem_get_playback_volume(current.txgain,
                                             SND_MIXER_SCHN_MONO, &ogain);
    CHECKERR("Failed to get capture volume");

    return (int)ogain;
}

/*
 * Record audio data.
 */

int alsa_audio_read(audio_desc_t ad __attribute__((unused)),
                u_char *buf, int bytes)
	{
    snd_pcm_sframes_t frames = snd_pcm_bytes_to_frames(current.rx.handle, bytes);
    snd_pcm_sframes_t fread;
    int err;

    fread = snd_pcm_readi(current.rx.handle, buf, frames);

    if (fread >= 0) {
        // Normal case
        fread = snd_pcm_frames_to_bytes(current.rx.handle, fread);
        debug_msg("Read %d bytes\n", fread);
        return fread;
    }

    // Something happened
    switch (fread)
	{
	case -EAGAIN:
        // Normal when non-blocking
	    return 0;

	case -EPIPE:
        debug_msg("Got capture XRUN\n");
        err = snd_pcm_prepare(current.rx.handle);
        CHECKERR("Can't recover from capture overrun");
        return FALSE;

      case -ESTRPIPE:
        debug_msg("Got capture ESTRPIPE\n");
        while ((err = snd_pcm_resume(current.rx.handle)) == -EAGAIN)
            sleep(1);       /* wait until the suspend flag is released */
        if (err < 0) {
            err = snd_pcm_prepare(current.rx.handle);
            CHECKERR("Can't recovery from capture suspend");
		}
        return FALSE;

	default:
        debug_msg("Write failed status=%d: %s\n", snd_strerror(fread));
	return 0;
    }
}

/*
 * Playback audio data.
 */


int alsa_audio_write(audio_desc_t ad __attribute__((unused)),
                     u_char *buf, int bytes)
{
    int fwritten, err;
    snd_pcm_sframes_t frames =
        snd_pcm_bytes_to_frames(current.tx.handle,bytes);

    debug_msg("Audio write %d\n", bytes);

    fwritten = snd_pcm_writei(current.tx.handle, buf, frames);
    if (fwritten >= 0) {
        // Normal case
        fwritten = snd_pcm_frames_to_bytes(current.tx.handle, fwritten);
        debug_msg("Wrote %d bytes\n", fwritten);
        return fwritten;
    }

    // Something happened
    switch (fwritten)
    {
      case -EAGAIN:
        // Normal when non-blocking
        return FALSE;

      case -EPIPE:
        debug_msg("Got transmit XRUN\n");
        err = snd_pcm_prepare(current.tx.handle);
        err = snd_pcm_writei(current.tx.handle, buf, frames);
        CHECKERR("Can't recover from transmit overrun");
        return TRUE;

      case -ESTRPIPE:
        debug_msg("Got transmit ESTRPIPE\n");
        while ((err = snd_pcm_resume(current.tx.handle)) == -EAGAIN)
            sleep(1);       /* wait until the suspend flag is released */
        if (err < 0) {
            err = snd_pcm_prepare(current.tx.handle);
            CHECKERR("Can't recovery from transmit suspend");
    }
        return FALSE;

      default:
        debug_msg("Write failed status=%d: %s\n", snd_strerror(fwritten));
        return FALSE;
    }


    fwritten = snd_pcm_frames_to_bytes(current.tx.handle, fwritten);
    debug_msg("Audio wrote %d\n", fwritten);
    return fwritten;
}


/*
 * Set options on audio device to be non-blocking.
 */
void alsa_audio_non_block(audio_desc_t ad __attribute__((unused)))
    {
    int err;
    debug_msg("Set nonblocking\n");

    err = snd_pcm_nonblock(current.tx.handle, TRUE);
    VCHECKERR("Error setting TX non-blocking");

    err = snd_pcm_nonblock(current.rx.handle, TRUE);
    VCHECKERR("Error setting RX non-blocking");
    }

    /*
 * Set options on audio device to be blocking.
    */
void alsa_audio_block(audio_desc_t ad)
    {
    int err;
    debug_msg("[%d] set blocking\n", ad);
    if ((err = snd_pcm_nonblock(current.tx.handle, FALSE)) < 0) {
        fprintf (stderr, "Cannot set blocking: %s\n",
                 snd_strerror (err));
    }
}



/*
 * Output port controls.  In our case there is only one output port, the
 * PCM control, so this is a dummy.
 */
void
alsa_audio_oport_set(audio_desc_t ad, audio_port_t port)
{
    debug_msg("oport_set %d %d\n", ad, port);
}
audio_port_t
alsa_audio_oport_get(audio_desc_t ad)
{
    debug_msg("oport_get %d\n", ad);
    return 0;
}

int
alsa_audio_oport_count(audio_desc_t ad)
{
    debug_msg("Get oport count for %d\n", ad);
    return 1;
}

const audio_port_details_t* alsa_audio_oport_details(audio_desc_t ad, int idx)
{
	debug_msg("oport details ad=%d idx=%d\n", ad, idx);
    return &out_port;
}

/*
 * Set input port.
 */
void
alsa_audio_iport_set(audio_desc_t ad, audio_port_t port)
{
    int err = 0;
    audio_port_t i;
    debug_msg("iport_set %d %d\n", ad, port);
    current.iport = port;

    for (i=0; i < num_iports; i++) {
        err = snd_mixer_selem_set_capture_switch_all(
            iports[i].mixer, (i==port));
    }
    VCHECKERR("Failed to set record switch");
}


/*
 * Get input port.
 */
audio_port_t
alsa_audio_iport_get(audio_desc_t ad)
{
    debug_msg("iport_get %d\n", ad);
    return current.iport;
}

int
alsa_audio_iport_count(audio_desc_t ad)
{
    debug_msg("Get iport count for %d (=%d)\n", ad, num_iports);
    return num_iports;
}

const audio_port_details_t* alsa_audio_iport_details(audio_desc_t ad, int idx)
{
	debug_msg("iport details ad=%d idx=%d\n", ad, idx);
    return &iports[idx].details;
}


/*
 * For external purposes this function returns non-zero
 * if audio is ready.
 */
int alsa_audio_is_ready(audio_desc_t ad __attribute__((unused)))
{
    snd_pcm_status_t *status;
    snd_pcm_uframes_t avail;
    int err;

    snd_pcm_status_alloca(&status);
    err = snd_pcm_status(current.rx.handle, status);
    CHECKERR("Can't get status of rx");

    avail = snd_pcm_frames_to_bytes(current.rx.handle,
                                    snd_pcm_status_get_avail(status));
    debug_msg("Audio ready == %d\n", avail);
    return (avail >= current.bytes_per_block);

}


void alsa_audio_wait_for(audio_desc_t ad __attribute__((unused)), int delay_ms)
	{
    debug_msg("Audio wait %d\n", delay_ms);
    snd_pcm_wait(current.rx.handle, delay_ms);
	}


char* alsa_get_device_name(audio_desc_t idx)
	{
    debug_msg("Get name for card %d: \"%s\"\n", idx, ratCards[idx].name);
    return ratCards[idx].name;
}


int alsa_audio_init()
    {
    int fd;
    char buf[4096];
    char *version;
    size_t count;
    int result =  FALSE;

    // Based on xmms-alsa

    fd = open("/proc/asound/version", O_RDONLY, 0);
    if (fd < 0) {
        result = FALSE;
	}  

    count = read(fd, buf, sizeof(buf) - 1);
    buf[count] = 0;
    close(fd);

    debug_msg("ALSA version identifier == %s\n", buf);

    version = strstr(buf, " Version ");

    if (version == NULL) {
        result = FALSE;
    }
	    
    version += 9; /* strlen(" Version ") */

    /* The successor to 0.9 might be 0.10, not 1.0.... */
    if (strcmp(version, "0.9") > 0  ||  isdigit(version[3])) {
        result = TRUE;
    } else {
        result = FALSE;
	    }

    debug_msg("ALSA init result == %d\n", result);

    clear_current();
    return result;
	    }

int alsa_get_device_count()
	    {
    snd_ctl_t *ctl_handle;
    snd_ctl_card_info_t *ctl_info;
    int err, cindex = 0;
    char card[128];
    RatCardInfo *ratCard;

    debug_msg("ALSA get device count\n");
	    
    do {
        sprintf (card , "hw:%d", cindex);
        err = snd_ctl_open (&ctl_handle, card, 0);


        if (err == 0) {
            // Grab the card info
            ratCard = ratCards + cindex;
            ratCard->card_number = cindex;
            sprintf (card , "plughw:%d", cindex);
            ratCard->pcm_device = strdup(card);

            snd_ctl_card_info_alloca(&ctl_info);
            if ((err = snd_ctl_card_info (ctl_handle, ctl_info) < 0)) {
                fprintf(stderr, "Card query failed: %s\n", snd_strerror(err));
                return 0;
    }
            snprintf(card, sizeof(card), "ALSA %d: %s", cindex,
                     snd_ctl_card_info_get_name (ctl_info));
            ratCard->name = strdup (card);
            debug_msg("  Got card %s\n", ratCard->name);
    
            snd_ctl_close(ctl_handle);
}
        cindex++;

    } while (err == 0);

    nRatCards = cindex - 1;
    debug_msg("Got %d devices\n", nRatCards);

    return nRatCards;
}

int alsa_audio_supports(audio_desc_t ad, audio_format *fmt)
{
    snd_pcm_hw_params_t *hw_params;
    unsigned int rmin, rmax, cmin, cmax;
    int err, dir;

    debug_msg("Got \"ALSA supports\" for %d\n", ad);
    dump_audio_format(fmt);


    snd_pcm_hw_params_alloca (&hw_params);
    err = snd_pcm_hw_params_any (current.tx.handle, hw_params);

    err = snd_pcm_hw_params_get_rate_min (hw_params, &rmin, &dir);
    CHECKERR("Failed to get min rate");
    err = snd_pcm_hw_params_get_rate_max (hw_params, &rmax, &dir);
    CHECKERR("Failed to get max rate");

    err = snd_pcm_hw_params_get_channels_min (hw_params, &cmin);
    CHECKERR("Failed to get min channels");
    err = snd_pcm_hw_params_get_channels_max (hw_params, &cmax);
    CHECKERR("Failed to get max channels");

    if ((fmt->sample_rate >= rmin) && (fmt->sample_rate <= rmax) &&
        (fmt->channels >= (int)cmin) && (fmt->channels <= (int)cmax))
{
        debug_msg("Config is supported\n");
        return TRUE;
    }
    return FALSE;
}

