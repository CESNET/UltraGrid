/**
 * @file   video_capture/shm.c
 * @author Martin Pulec     <martin.pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2019-2020 CESNET, z. s. p. o.
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
#endif // HAVE_CONFIG_H
#include "config_unix.h"
#include "config_win32.h"

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#endif
#include <assert.h>
#include <stdatomic.h>
#include <stdbool.h>

#include "compat/platform_ipc.h"
#include "debug.h"
#include "lib_common.h"
#include "utils/color_out.h"
#include "utils/misc.h"
#include "video.h"
#include "video_capture.h"
#include "video_capture/shm.h"

// substitute RGBA not to collide with UG RGBA
#define RGBA VR_RGBA
#ifdef HAVE_VRG_H
#include <vrgstream.h>
#else
#include "vrgstream-fallback.h"
#endif
#undef RGBA

#define BUFFERS 2
#define MAX_BUF_LEN (7680 * 2160 * 3 / 2)
#define KEY "UltraGrid-SHM"
#define SHM_VERSION 8
#define MAGIC to_fourcc('V', 'C', 'C', 'U')
#define MOD_NAME "[shm] "
#define UG_CUDA_IPC_HANDLE_SIZE 64 // originally definde by CUDA

#if defined HAVE_CUDA && (defined __cplusplus || defined static_assert)
static_assert(UG_CUDA_IPC_HANDLE_SIZE == CUDA_IPC_HANDLE_SIZE, "CUDA IPC handle doesn't match expected size!");
#endif

static void shm_dispose_frame(struct video_frame *f);

struct shm_frame {
        atomic_bool buffer_free;
        int width, height;
        char cuda_ipc_mem_handle[UG_CUDA_IPC_HANDLE_SIZE];
        char data[MAX_BUF_LEN];

        // follow auxilliary data used by UltraGrid for dispose
        int frame_idx; ///< index inside shm::frames
        void *parent; ///< pointer to struct shm
};

struct shm {
        int version; ///< this struct version, must be SHM_VERSION
        atomic_bool initialized;
        atomic_bool ug_exited;
        bool use_gpu;
        struct RenderPacket pkt; ///@todo make this more IPC safe
        atomic_int write_head; ///< next position of written frame
        atomic_int read_head; ///< next position of frame read
        struct shm_frame frames[BUFFERS];
};

#define READY_TO_CONSUME_FRAME 0
#define FRAME_READY 1
#define SHOULD_EXIT_LOCK 2

struct state_vidcap_shm {
        uint32_t magic;
        struct video_frame *f[BUFFERS];
        struct shm *shm;
        platform_ipc_shm_t shm_id;
        platform_ipc_sem_t sem_id[3]; ///< 3 semaphores: [ready_to_consume_frame, frame_ready, should_exit_lock]
        bool use_gpu;
        struct module *parent;
        bool should_exit;
        bool force; ///< forces re-creation of shm/sem even if already exist

        struct RenderPacket render_pkt;
        pthread_mutex_t render_pkt_lock;
};

static void vidcap_shm_cleanup(struct state_vidcap_shm *s);

static void show_help() {
        printf("Usage:\n");
        color_out(COLOR_OUT_BOLD, "\t-t cuda|shm[:force]\n");
        printf("\t\tCaptures the frame either to shared memory on GPU (CUDA buffer) or in RAM (shared memory)\n");
        printf("options:\n");
        color_out(COLOR_OUT_BOLD, "\tforce\n");
        printf("\t\tRemoves already existing IPC primitives (Linux) if already exist.\n");
}

static void vidcap_shm_should_exit(void *arg) {
        struct state_vidcap_shm *s = (struct state_vidcap_shm *) arg;
        assert(s->magic == MAGIC);
        s->should_exit = true;

        platform_ipc_sem_post(s->sem_id[FRAME_READY]);
}

#define CUDA_CHECK(cmd) do { cudaError_t err = cmd; if (err != cudaSuccess) { log_msg(LOG_LEVEL_ERROR, "%s: %s\n", #cmd, cudaGetErrorString(err)); goto error; } } while(0)
static int vidcap_shm_init(struct vidcap_params *params, void **state)
{
        if (vidcap_params_get_flags(params) & VIDCAP_FLAG_AUDIO_ANY) {
                return VIDCAP_INIT_AUDIO_NOT_SUPPOTED;
        }

        const char *fmt = vidcap_params_get_fmt(params);
        if (strcmp(fmt, "help") == 0) {
                show_help();
                return VIDCAP_INIT_NOERR;
        }

        struct state_vidcap_shm *s = calloc(1, sizeof(struct state_vidcap_shm));
        if (!s) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Cannot allocate.\n");
                return VIDCAP_INIT_FAIL;
        }
        s->magic = MAGIC;
        s->shm_id = PLATFORM_IPC_ERR;
        for (int i = 0; i < 3; ++i) {
                s->sem_id[i] = PLATFORM_IPC_ERR;
        }
        if (strcmp(vidcap_params_get_name(params), "cuda") == 0) {
                s->use_gpu = true;
        }
        if (strcmp(fmt, "force") == 0) {
                s->force = true;
        }
        s->parent = vidcap_params_get_parent(params);

        struct video_desc desc = { .width = 0,
                .height = 0,
                .fps = 30,
                .tile_count = 1,
                .color_spec = s->use_gpu ? CUDA_I420 : I420,
                .interlacing = PROGRESSIVE,
        };

        size_t size = sizeof(struct shm);
        if ((s->shm_id = platform_ipc_shm_create(KEY, size)) == PLATFORM_IPC_ERR) {
                if (errno == EEXIST) {
                        if (s->force) {
                                s->shm_id = platform_ipc_shm_open(KEY, size);
                                platform_ipc_shm_done(s->shm_id, true);
                                s->shm_id = platform_ipc_shm_create(KEY, size);
                        } else {
                                printf("You may also use ':force' parameter to force removing already existing IPC handles.\n");
                        }
                }
                if (s->shm_id == PLATFORM_IPC_ERR) {
                        goto error;
                }
        }
        s->shm = platform_ipc_shm_attach(s->shm_id, size);
        if (s->shm == (void *) PLATFORM_IPC_ERR) {
                perror("shmat");
                goto error;
        }
        s->shm->version = SHM_VERSION;
        s->shm->ug_exited = 0;
        s->shm->use_gpu = s->use_gpu;
        s->shm->pkt.frame = -1;
        for (int i = 0; i < 3; ++i) {
                s->sem_id[i] = platform_ipc_sem_create(KEY, i + 1);
                if (s->sem_id[i] == PLATFORM_IPC_ERR) {
                        if (s->force && errno == EEXIST) {
                                s->sem_id[i] = platform_ipc_sem_open(KEY, i + 1);
                                platform_ipc_sem_done(s->sem_id[i], true);
                                s->sem_id[i] = platform_ipc_sem_create(KEY, i + 1);
                        }
                        if (s->sem_id[i] == PLATFORM_IPC_ERR) {
                                goto error;
                        }
                }
        }

        for (int i = 0; i < BUFFERS; ++i) {
                s->f[i] = vf_alloc_desc(desc);
                s->shm->frames[i].buffer_free = true;
                s->shm->frames[i].parent = s;
                s->shm->frames[i].frame_idx = i;
                if (s->use_gpu) {
#ifdef HAVE_CUDA
                        CUDA_CHECK(cudaMalloc((void **) &s->f[i]->tiles[0].data, MAX_BUF_LEN));
                        CUDA_CHECK(cudaIpcGetMemHandle ((cudaIpcMemHandle_t *) &s->shm->frames[i].cuda_ipc_mem_handle, s->f[i]->tiles[0].data));
#endif
                } else {
                        s->f[i]->tiles[0].data = s->shm->frames[i].data;
                }
                s->f[i]->callbacks.dispose = shm_dispose_frame;
                s->f[i]->callbacks.dispose_udata = &s->shm->frames[i];
        }

        // initialize semaphores
        for (int i = 0; i < BUFFERS; ++i) {
                platform_ipc_sem_post(s->sem_id[READY_TO_CONSUME_FRAME]);
        }
        platform_ipc_sem_post(s->sem_id[SHOULD_EXIT_LOCK]);

        register_should_exit_callback(s->parent, vidcap_shm_should_exit, s);
        pthread_mutex_init(&s->render_pkt_lock, NULL);
        s->render_pkt.frame = -1;

        s->shm->initialized = true;

        *state = s;
        return VIDCAP_INIT_OK;

error:
        vidcap_shm_cleanup(s);
        return VIDCAP_INIT_FAIL;
}

static void vidcap_shm_cleanup(struct state_vidcap_shm *s)
{
        for (int i = 0; i < BUFFERS; ++i) {
                if (s->use_gpu) {
#ifdef HAVE_CUDA
                        cudaFree(s->f[i]->tiles[0].data);
#endif
                }
                vf_free(s->f[i]);
        }

        // We are deleting the IPC primitives here. Calls on such primitives will
        // then fail inside the plugin. If it is waiting on the semaphore 0, the
        // call gets interrupted.
        if (s->shm_id != PLATFORM_IPC_ERR) {
                platform_ipc_shm_done(s->shm_id, true);
        }
        for (int i = 0; i < 3; ++i) {
                if (s->sem_id[i] != PLATFORM_IPC_ERR) {
                        platform_ipc_sem_done(s->sem_id[i], true);
                }
        }
        pthread_mutex_destroy(&s->render_pkt_lock);
        free(s);
}

static void vidcap_shm_done(void *state)
{
        struct state_vidcap_shm *s = (struct state_vidcap_shm *) state;
        assert(s->magic == MAGIC);

        s->shm->ug_exited = true;

        platform_ipc_sem_post(s->sem_id[READY_TO_CONSUME_FRAME]);

        platform_ipc_sem_wait(s->sem_id[SHOULD_EXIT_LOCK]); // wait for consumer to finish
        platform_ipc_sem_post(s->sem_id[SHOULD_EXIT_LOCK]); // in case of wrong race condition

        vidcap_shm_cleanup(s);
}

static void shm_dispose_frame(struct video_frame *f) {
        struct shm_frame *frame = (struct shm_frame *) f->callbacks.dispose_udata;
        struct state_vidcap_shm *s = (struct state_vidcap_shm *) frame->parent;
        assert(!frame->buffer_free);
        frame->buffer_free = true;

        // we are ready to take another frame
        platform_ipc_sem_post(s->sem_id[READY_TO_CONSUME_FRAME]);
}


static struct video_frame *vidcap_shm_grab(void *state, struct audio_frame **audio)
{
        struct state_vidcap_shm *s = (struct state_vidcap_shm *) state;
        assert(s->magic == MAGIC);

        if (s->should_exit) {
                return NULL;
        }

        // wait for frame
        if (!platform_ipc_sem_wait(s->sem_id[FRAME_READY])) {
                return NULL;
        }

        if (s->should_exit) {
                return NULL;
        }

        s->f[s->shm->read_head]->tiles[0].width = s->shm->frames[s->shm->read_head].width;
        s->f[s->shm->read_head]->tiles[0].height = s->shm->frames[s->shm->read_head].height;
        s->f[s->shm->read_head]->tiles[0].data_len = vc_get_datalen(s->f[s->shm->read_head]->tiles[0].width, s->f[s->shm->read_head]->tiles[0].height, s->f[s->shm->read_head]->color_spec);
        assert(!s->shm->frames[s->shm->read_head].buffer_free);

        pthread_mutex_lock(&s->render_pkt_lock);
        if (s->render_pkt.frame != (unsigned int) -1) {
                memcpy(&s->shm->pkt, &s->render_pkt, sizeof(s->render_pkt));
        }
        pthread_mutex_unlock(&s->render_pkt_lock);

        *audio = NULL;
        struct video_frame *out = s->f[s->shm->read_head];
        s->shm->read_head = (s->shm->read_head + 1) % BUFFERS;
        return out;
}

static struct vidcap_type *vidcap_shm_probe(bool verbose, void (**deleter)(void *))
{
        UNUSED(verbose);
        struct vidcap_type *vt;

        vt = (struct vidcap_type *) calloc(1, sizeof(struct vidcap_type));
        if (vt == NULL) {
                return NULL;
        }

        vt->name = "shm";
        vt->description = "SHM virtual capture card";
        *deleter = free;

        return vt;
}

void vidcap_shm_set_view(void *state, struct RenderPacket *pkt) {
        struct state_vidcap_shm *s = state;
        assert(s->magic == MAGIC);

        pthread_mutex_lock(&s->render_pkt_lock);
        memcpy(&s->render_pkt, pkt, sizeof *pkt);
        pthread_mutex_unlock(&s->render_pkt_lock);
}

static const struct video_capture_info vidcap_shm_info = {
        vidcap_shm_probe,
        vidcap_shm_init,
        vidcap_shm_done,
        vidcap_shm_grab,
        true
};

#ifdef HAVE_CUDA
REGISTER_MODULE(cuda, &vidcap_shm_info, LIBRARY_CLASS_VIDEO_CAPTURE, VIDEO_CAPTURE_ABI_VERSION);
#endif
REGISTER_MODULE(shm, &vidcap_shm_info, LIBRARY_CLASS_VIDEO_CAPTURE, VIDEO_CAPTURE_ABI_VERSION);

