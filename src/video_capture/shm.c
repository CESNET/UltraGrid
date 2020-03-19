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

#include <cuda_runtime.h>
#include <stdbool.h>
#include <sys/sem.h>
#include <sys/shm.h>

#include "debug.h"
#include "lib_common.h"
#include "utils/color_out.h"
#include "utils/misc.h"
#include "video.h"
#include "video_capture.h"
#include "video_capture/shm.h"
#define RGBA VR_RGBA
#include "vrgstream.h"
#undef RGBA

#define MAX_BUF_LEN (7680 * 2160 / 3 * 2)
#define SEM_KEY "/UltraGridSem"
#define SHM_KEY "/UltraGridSHM"
#define SHM_VERSION 3
#define MAGIC to_fourcc('V', 'C', 'C', 'U')
#define MOD_NAME "[shm] "

struct shm {
        int version; ///< this struct version, must be SHM_VERSION
        int width, height;
        cudaIpcMemHandle_t d_ptr;
        int ug_exited;
        int use_gpu;
        struct RenderPacket pkt;
        char data[];
};

struct state_vidcap_shm {
        uint32_t magic;
        struct video_frame *f;
        struct shm *shm;
	int shm_id;
	int sem_id;
        bool use_gpu;
        struct module *parent;
        bool should_exit;

        struct RenderPacket render_pkt;
        pthread_mutex_t render_pkt_lock;
};

static void show_help() {
        printf("Usage:\n");
        color_out(COLOR_OUT_BOLD, "\t-t cuda|shm\n");
        printf("\t\tCaptures the frame either to shared memory on GPU (CUDA buffer) or in RAM (shared memory)\n");
}

static void vidcap_shm_should_exit(void *arg) {
        struct state_vidcap_shm *s = (struct state_vidcap_shm *) arg;
        assert(s->magic == MAGIC);
        s->should_exit = true;

        struct sembuf op;
        op.sem_num = 1;
        op.sem_op = 1;
        op.sem_flg = 0;
        if (semop(s->sem_id, &op, 1) < 0) {
                perror("semop");
                abort();
        }
}

#define CUDA_CHECK(cmd) do { cudaError_t err = cmd; if (err != cudaSuccess) { log_msg(LOG_LEVEL_ERROR, "%s: %s\n", #cmd, cudaGetErrorString(err)); goto error; } } while(0)
static int vidcap_shm_init(struct vidcap_params *params, void **state)
{
        if (vidcap_params_get_flags(params) & VIDCAP_FLAG_AUDIO_ANY) {
                return VIDCAP_INIT_AUDIO_NOT_SUPPOTED;
        }

        if (strcmp(vidcap_params_get_fmt(params), "help") == 0) {
                show_help();
                return VIDCAP_INIT_NOERR;
        }

        struct state_vidcap_shm *s = calloc(1, sizeof(struct state_vidcap_shm));
        s->magic = MAGIC;
        if (strcmp(vidcap_params_get_name(params), "cuda") == 0) {
                s->use_gpu = true;
        }
        s->parent = vidcap_params_get_parent(params);

        struct video_desc desc = { .width = 0,
                .height = 0,
                .fps = 30,
                .tile_count = 1,
                .color_spec = s->use_gpu ? CUDA_I420 : I420,
                .interlacing = PROGRESSIVE,
        };

        s->f = vf_alloc_desc(desc);

        size_t size;
        if (s->use_gpu) {
                size = sizeof(struct shm);
        } else {
                size = offsetof(struct shm, data[MAX_BUF_LEN]);
        }
        key_t key = ftok(SHM_KEY, 1);
        if ((s->shm_id = shmget(key, size, IPC_CREAT | 0666)) == -1) {
                if (errno != EEXIST) {
                        perror("shmget");
                        if (errno == EINVAL) {
                                log_msg(LOG_LEVEL_INFO, MOD_NAME "Try to remove it with \"ipcrm\" (see \"ipcs\"), "
                                                "key %lld.\n", (long long) key);
                        }
                        goto error;
                }
        }
        s->shm = shmat(s->shm_id, NULL, 0);
        if (s->shm == (void *) -1) {
                perror("shmat");
                goto error;
        }
        s->shm->version = SHM_VERSION;
        s->shm->ug_exited = 0;
        s->shm->use_gpu = s->use_gpu;
        s->shm->pkt.frame = -1;
        s->sem_id = semget(ftok(SEM_KEY, 1), 2, IPC_CREAT | 0666);
        if (s->sem_id == -1) {
                if (errno != EEXIST) {
                        perror("semget");
                        goto error;
                }
        }

        if (s->use_gpu) {
#ifdef HAVE_CUDA
                CUDA_CHECK(cudaMalloc((void **) &s->f->tiles[0].data, MAX_BUF_LEN));
                CUDA_CHECK(cudaIpcGetMemHandle (&s->shm->d_ptr, s->f->tiles[0].data));
#endif
        } else {
                s->f->tiles[0].data = s->shm->data;
        }

        struct sembuf op;
        op.sem_num = 0;
        op.sem_op = 1;
        op.sem_flg = 0;
        if (semop(s->sem_id, &op, 1) < 0) {
                perror("semop");
                goto error;
        }

        register_should_exit_callback(s->parent, vidcap_shm_should_exit, s);
        pthread_mutex_init(&s->render_pkt_lock, NULL);
        s->render_pkt.frame = -1;

        *state = s;
        return VIDCAP_INIT_OK;

error:
        free(s);
        return VIDCAP_INIT_FAIL;
}

static void vidcap_shm_done(void *state)
{
        struct state_vidcap_shm *s = (struct state_vidcap_shm *) state;
        assert(s->magic == MAGIC);

        s->shm->ug_exited = true;

        struct sembuf op;
        op.sem_num = 0;
        op.sem_op = -1;
        op.sem_flg = 0;
        if (semop(s->sem_id, &op, 1) < 0) {
                perror("semop");
        }
        op.sem_op = 1;
        if (semop(s->sem_id, &op, 1) < 0) {
                perror("semop");
        }

        if (s->use_gpu) {
                cudaFree(s->f->tiles[0].data);
        }
        vf_free(s->f);

        // We are deleting the IPC primitives here. Calls on such primitives will
        // then fail inside the plugin. If it is waiting on the semaphore 0, the
        // call gets interrupted.
	if (shmctl(s->shm_id, IPC_RMID , 0) == -1) {
		perror("shmctl");
	}
	if (semctl(s->sem_id, IPC_RMID , 0) == -1) {
		perror("semctl");
	}
        pthread_mutex_destroy(&s->render_pkt_lock);
        free(s);
}

static struct video_frame *vidcap_shm_grab(void *state, struct audio_frame **audio)
{
        struct state_vidcap_shm *s = (struct state_vidcap_shm *) state;
        assert(s->magic == MAGIC);

        // wait for frame
        struct sembuf op;
        op.sem_num = 1;
        op.sem_op = -1;
        op.sem_flg = 0;
        if (semop(s->sem_id, &op, 1) < 0) {
                perror("semop");
                *audio = NULL;
                return NULL;
        }

        if (s->should_exit) {
                return NULL;
        }

        s->f->tiles[0].width = s->shm->width;
        s->f->tiles[0].height = s->shm->height;

        pthread_mutex_lock(&s->render_pkt_lock);
        if (s->render_pkt.frame != (unsigned int) -1) {
                memcpy(&s->shm->pkt, &s->render_pkt, sizeof(s->render_pkt));
        }
        pthread_mutex_unlock(&s->render_pkt_lock);

        return s->f;
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

