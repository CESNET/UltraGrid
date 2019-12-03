/**
 * @file   video_capture/cuda.c
 * @author Martin Pulec     <martin.pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2019 CESNET, z. s. p. o.
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
#include <sys/sem.h>
#include <sys/shm.h>

#include "debug.h"
#include "lib_common.h"
#include "utils/color_out.h"
#include "utils/misc.h"
#include "video.h"
#include "video_capture.h"

#define MS100_IN_NS (100ULL * 1000 * 1000)
#define MAX_BUF_LEN (7680 * 2160 / 3 * 2)
#define SEM_KEY "/UltraGridSem"
#define SHM_KEY "/UltraGridSHM"
#define MAGIC to_fourcc('V', 'C', 'C', 'U')

struct shm {
        int width, height;
        cudaIpcMemHandle_t *d_ptr;
        int ug_exited;
};

struct state_vidcap_cuda {
        uint32_t magic;
        struct video_frame *f;
        struct shm *shm;
	int shm_id;
	int sem_id;
};

static int vidcap_cuda_init(struct vidcap_params *params, void **state)
{
        if (vidcap_params_get_flags(params) & VIDCAP_FLAG_AUDIO_ANY) {
                return VIDCAP_INIT_AUDIO_NOT_SUPPOTED;
        }
        struct state_vidcap_cuda *s = calloc(1, sizeof(struct state_vidcap_cuda));
        s->magic = MAGIC;

        struct video_desc desc = { .width = 0,
                .height = 0,
               .fps = 30,
               .tile_count = 1,
               .color_spec = CUDA_I420,
               .interlacing = PROGRESSIVE,
        };

        s->f = vf_alloc_desc(desc);

        cudaError_t err = cudaMalloc((void **) &s->f->tiles[0].data, MAX_BUF_LEN);
        if (err != cudaSuccess) {
                fprintf(stderr, "%s\n", cudaGetErrorString(cudaGetLastError()));
                goto error;
        }

        int id = shmget(ftok(SHM_KEY, 1), sizeof(struct shm), IPC_CREAT | 0666);
        if (id == -1) {
                perror("shmget");
                goto error;
        }
	if (shmctl(id, IPC_RMID , 0) == -1) {
		perror("shmctl");
	}
        if ((s->shm_id = shmget(ftok(SHM_KEY, 1), sizeof(struct shm), IPC_CREAT | 0666)) == -1) {
                perror("shmget");
                goto error;
        }
        s->shm = shmat(s->shm_id, NULL, 0);
        if (s->shm == (void *) -1) {
                perror("shmat");
                goto error;
        }
        err = cudaIpcGetMemHandle (s->shm->d_ptr, s->f->tiles[0].data);
        if (err != cudaSuccess) {
                fprintf(stderr, "%s\n", cudaGetErrorString(cudaGetLastError()));
                goto error;
        }
        s->shm->ug_exited = 0;
        s->sem_id = semget(ftok(SEM_KEY, 1), 2, IPC_CREAT | 0666);
        if (s->sem_id == -1) {
                perror("semget");
                goto error;
        }

        struct sembuf op;
        op.sem_num = 0;
        op.sem_op = 1;
        op.sem_flg = 0;
        if (semop(s->sem_id, &op, 1) < 0) {
                perror("semop");
                goto error;
        }

        *state = s;
        return VIDCAP_INIT_OK;

error:
        free(s);
        return VIDCAP_INIT_FAIL;
}

static void vidcap_cuda_done(void *state)
{
        struct state_vidcap_cuda *s = (struct state_vidcap_cuda *) state;
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

        cudaFree(s->f->tiles[0].data);
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

static struct video_frame *vidcap_cuda_grab(void *state, struct audio_frame **audio)
{
        struct state_vidcap_cuda *s = (struct state_vidcap_cuda *) state;
        assert(s->magic == MAGIC);

        // wait for frame
        struct sembuf op;
        op.sem_num = 1;
        op.sem_op = -1;
        op.sem_flg = 0;
        struct timespec timeout;
        timeout.tv_sec = 0;
        timeout.tv_nsec = MS100_IN_NS;
        if (semtimedop(s->sem_id, &op, 1, &timeout) < 0) {
                perror("semtimedop");
                *audio = NULL;
                return NULL;
        }

        s->f->tiles[0].width = s->shm->width;
        s->f->tiles[0].height = s->shm->height;

        return s->f;
}

static struct vidcap_type *vidcap_cuda_probe(bool verbose, void (**deleter)(void *))
{
        UNUSED(verbose);
        struct vidcap_type *vt;

        vt = (struct vidcap_type *) calloc(1, sizeof(struct vidcap_type));
        if (vt == NULL) {
                return NULL;
        }

        vt->name = "cuda";
        vt->description = "CUDA virtual capture card";
        *deleter = free;

        return vt;
}

static const struct video_capture_info vidcap_cuda_info = {
        vidcap_cuda_probe,
        vidcap_cuda_init,
        vidcap_cuda_done,
        vidcap_cuda_grab,
        true
};

REGISTER_MODULE(cuda, &vidcap_cuda_info, LIBRARY_CLASS_VIDEO_CAPTURE, VIDEO_CAPTURE_ABI_VERSION);

