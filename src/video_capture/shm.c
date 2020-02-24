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

#define MAX_BUF_LEN (7680 * 2160 / 3 * 2)
#define SEM_KEY "/UltraGridSem"
#define SHM_KEY "/UltraGridSHM"
#define MAGIC to_fourcc('V', 'C', 'C', 'U')
#define MOD_NAME "[shm] "

struct shm {
        int width, height;
        cudaIpcMemHandle_t d_ptr;
        int ug_exited;
        int use_gpu;
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
};

static void show_help() {
        printf("Usage:\n");
        printf("-t cuda|shm\n");
}

static void vidcap_shm_should_exit(void *arg) {
        struct state_vidcap_shm *s = (struct state_vidcap_shm *) arg;
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

#define CUDA_CHECK(cmd) do { cudaError_t err = cmd; if (err != cudaSuccess) { fprintf(stderr, "%s\n", cudaGetErrorString(err)); goto error; } } while(0)
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
        s->shm->ug_exited = 0;
        s->shm->use_gpu = s->use_gpu;
        s->sem_id = semget(ftok(SEM_KEY, 1), 2, IPC_CREAT | 0666);
        if (s->sem_id == -1) {
                if (errno != EEXIST) {
                        perror("semget");
                        goto error;
                }
        }

        if (s->use_gpu) {
                CUDA_CHECK(cudaMalloc((void **) &s->f->tiles[0].data, MAX_BUF_LEN));
                CUDA_CHECK(cudaIpcGetMemHandle (&s->shm->d_ptr, s->f->tiles[0].data));
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

static const struct video_capture_info vidcap_shm_info = {
        vidcap_shm_probe,
        vidcap_shm_init,
        vidcap_shm_done,
        vidcap_shm_grab,
        true
};

REGISTER_MODULE(cuda, &vidcap_shm_info, LIBRARY_CLASS_VIDEO_CAPTURE, VIDEO_CAPTURE_ABI_VERSION);
REGISTER_MODULE(shm, &vidcap_shm_info, LIBRARY_CLASS_VIDEO_CAPTURE, VIDEO_CAPTURE_ABI_VERSION);

