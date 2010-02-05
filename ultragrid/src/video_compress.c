/*
 * FILE:    video_compress.c
 * AUTHORS: Martin Benes     <martinbenesh@gmail.com>
 *          Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 *          Petr Holub       <hopet@ics.muni.cz>
 *          Milos Liska      <xliska@fi.muni.cz>
 *          Jiri Matela      <matela@ics.muni.cz>
 *          Dalibor Matura   <255899@mail.muni.cz>
 *          Ian Wesley-Smith <iwsmith@cct.lsu.edu>
 *
 * Copyright (c) 2005-2010 CESNET z.s.p.o.
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
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 * 
 *      This product includes software developed by CESNET z.s.p.o.
 * 
 * 4. Neither the name of the CESNET nor the names of its contributors may be
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
 *
 */

#include "host.h"
#include "config.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <assert.h>
#ifdef HAVE_MACOSX
#include <malloc/malloc.h>
#else /* HAVE_MACOSX */
#include <malloc.h>
#endif /* HAVE_MACOSX */
#include <string.h>
#include <unistd.h>
#include "video_types.h"
#include "video_compress.h"
#include "libdxt.h"

#ifndef HAVE_MACOSX
#define uint64_t 	unsigned long
#endif /* HAVE_MACOSX */

/* NOTE: These threads busy wait, so at *most* set this to one less than the
 * total number of cores on your system (Also 3 threads will work)! Also, if
 * you see tearing in the rendered image try increasing the number of threads
 * by 1 (For a dual dual-core Opteron 285 3 threads work great). 
 * -- iwsmith <iwsmith@cct.lsu.ed> 9 August 2007
 */
#define NUM_THREADS 3
#define HD_HEIGHT 1080
#define HD_WIDTH 1920

/* Ok, we are going to decompose the problem into 2^n pieces (generally
 * 2, 4, or 8). We will most likely need to do the following:
 * 1. Convert 10-bit -> 8bit
 * 2. Convert YUV 4:2:2 -> 8 bit RGB
 * 3. Compress 8 bit RGB -> DXT
 */

struct video_compress {
	unsigned char *buffer[NUM_THREADS];
	unsigned char *output_data;
	unsigned char *out;
	pthread_mutex_t lock;
	int thread_count,len[NUM_THREADS],go[NUM_THREADS];
	pthread_t thread_ids[NUM_THREADS];
};

inline void compress_copyline64(unsigned char *dst, unsigned char *src, int len);
inline void compress_copyline128(unsigned char *d, unsigned char *s, int len);
void compress_deinterlace(unsigned char *buffer);
void compress_data(void *args, struct video_frame * tx);

struct video_compress * initialize_video_compression(void)
{
	/* This function does the following:
	 * 1. Allocate memory for buffers 
	 * 2. Spawn compressor threads
	 */
	int x;
	struct video_compress *compress;

	compress=calloc(1,sizeof(struct video_compress));
	for(x=0;x<NUM_THREADS;x++) {
		compress->buffer[x]=(unsigned char*)malloc(1920*1080*4/NUM_THREADS);
	}
#ifdef HAVE_MACOSX
	compress->output_data=(unsigned char*)malloc(1920*1080*4);
	compress->out=(unsigned char*)malloc(1920*1080*4);
#else	
	/*
	 *  memalign doesn't exist on Mac OS. malloc always returns 16  
	 *  bytes aligned memory
	 *
         *  see: http://www.mythtv.org/pipermail/mythtv-dev/2006-January/044309.html
	 */	 
	compress->output_data=(unsigned char*)memalign(16,1920*1080*4);
	compress->out=(unsigned char*)memalign(16,1920*1080*4);
#endif /* HAVE_MACOSX */
	memset(compress->output_data,0,1920*1080*4);
	memset(compress->out,0,1920*1080*4/8);
	compress->thread_count=0;
	if(pthread_mutex_init(&(compress->lock),NULL)){
		perror("Error initializing mutex!");
		exit(x);
	}

	pthread_mutex_lock(&(compress->lock));

	for(x=0;x<NUM_THREADS;x++){
		if(pthread_create(&(compress->thread_ids[x]), NULL, (void *)compress_thread, (void *)compress)) {
			perror("Unable to create compressor thread!");
			exit(x);
		}

		compress->go[x]=0;
		compress->len[x]=0;
	} 
	pthread_mutex_unlock(&(compress->lock));
	return compress;	
}


#if !(HAVE_MACOSX || HAVE_32B_LINUX)

inline void compress_copyline128(unsigned char *d, unsigned char *s, int len)
{
        register unsigned char *_d=d,*_s=s;

        while(--len >= 0) {

                asm ("movd %0, %%xmm4\n": : "r" (0xffffff));

                asm volatile ("movdqa (%0), %%xmm0\n"
                        "movdqa 16(%0), %%xmm5\n"
                        "movdqa %%xmm0, %%xmm1\n"
                        "movdqa %%xmm0, %%xmm2\n"
                        "movdqa %%xmm0, %%xmm3\n"
                        "pand  %%xmm4, %%xmm0\n"
                        "movdqa %%xmm5, %%xmm6\n"
                        "movdqa %%xmm5, %%xmm7\n"
                        "movdqa %%xmm5, %%xmm8\n"
                        "pand  %%xmm4, %%xmm5\n"
                        "pslldq $4, %%xmm4\n"
                        "pand  %%xmm4, %%xmm1\n"
                        "pand  %%xmm4, %%xmm6\n"
                        "pslldq $4, %%xmm4\n"
                        "psrldq $1, %%xmm1\n"
                        "psrldq $1, %%xmm6\n"
                        "pand  %%xmm4, %%xmm2\n"
                        "pand  %%xmm4, %%xmm7\n"
                        "pslldq $4, %%xmm4\n"
                        "psrldq $2, %%xmm2\n"
                        "psrldq $2, %%xmm7\n"
                        "pand  %%xmm4, %%xmm3\n"
                        "pand  %%xmm4, %%xmm8\n"
                        "por %%xmm1, %%xmm0\n"
                        "psrldq $3, %%xmm3\n"
                        "psrldq $3, %%xmm8\n"
                        "por %%xmm2, %%xmm0\n"
                        "por %%xmm6, %%xmm5\n"
                        "por %%xmm3, %%xmm0\n"
                        "por %%xmm7, %%xmm5\n"
                        "movdq2q %%xmm0, %%mm0\n"
                        "por %%xmm8, %%xmm5\n"
                        "movdqa %%xmm5, %%xmm1\n"
                        "pslldq $12, %%xmm5\n"
                        "psrldq $4, %%xmm1\n"
                        "por %%xmm5, %%xmm0\n"
                        "psrldq $8, %%xmm0\n"
                        "movq %%mm0, (%1)\n"
                        "movdq2q %%xmm0, %%mm1\n"
                        "movdq2q %%xmm1, %%mm2\n"
                        "movq %%mm1, 8(%1)\n"
                        "movq %%mm2, 16(%1)\n"
                        :
                        : "r" (_s), "r" (_d));
                _s += 32;
                _d += 24;
        }
}

#endif /* !(HAVE_MACOSX || HAVE_32B_LINUX) */

inline void compress_copyline64(unsigned char *dst, unsigned char *src, int len)
{
        register uint64_t *d, *s;

        register uint64_t a1,a2,a3,a4;

        d = (uint64_t *)dst;
        s = (uint64_t *)src;

        while(len-- > 0) {
                a1 = *(s++);
                a2 = *(s++);
                a3 = *(s++);
                a4 = *(s++);

                a1 = (a1 & 0xffffff) | ((a1 >> 8) & 0xffffff000000);
                a2 = (a2 & 0xffffff) | ((a2 >> 8) & 0xffffff000000);
                a3 = (a3 & 0xffffff) | ((a3 >> 8) & 0xffffff000000);
                a4 = (a4 & 0xffffff) | ((a4 >> 8) & 0xffffff000000);

                *(d++) = a1 | (a2 << 48);       /* 0xa2|a2|a1|a1|a1|a1|a1|a1 */
                *(d++) = (a2 >> 16)|(a3 << 32); /* 0xa3|a3|a3|a3|a2|a2|a2|a2 */
                *(d++) = (a3 >> 32)|(a4 << 16); /* 0xa4|a4|a4|a4|a4|a4|a3|a3 */
        }
}


/* linear blend deinterlace */
void compress_deinterlace(unsigned char *buffer)
{
        int i,j;
        long pitch = 1920*2;
        register long pitch2 = pitch*2;
        unsigned char *bline1, *bline2, *bline3;
        register unsigned char *line1, *line2, *line3;

        bline1 = buffer;
        bline2 = buffer + pitch;
        bline3 = buffer + 3*pitch;
        for(i=0; i < 1920*2; i+=16) {
                /* preload first two lines */
                asm volatile(
                             "movdqa (%0), %%xmm0\n"
                             "movdqa (%1), %%xmm1\n"
                             :
                             : "r" ((unsigned long *)bline1),
                               "r" ((unsigned long *)bline2));
                line1 = bline2;
                line2 = bline2 + pitch;
                line3 = bline3;
                for(j=0; j < 1076; j+=2) {
                        asm  volatile(
                              "movdqa (%1), %%xmm2\n"
                              "pavgb %%xmm2, %%xmm0\n"
                              "pavgb %%xmm1, %%xmm0\n"
                              "movdqa (%2), %%xmm1\n"
                              "movdqa %%xmm0, (%0)\n"
                              "pavgb %%xmm1, %%xmm0\n"
                              "pavgb %%xmm2, %%xmm0\n"
                              "movdqa %%xmm0, (%1)\n"
                              :
                              :"r" ((unsigned long *)line1),
                               "r" ((unsigned long *)line2),
                               "r" ((unsigned long *)line3)
                              );
                        line1 += pitch2;
                        line2 += pitch2;
                        line3 += pitch2;
                }
                bline1 += 16;
                bline2 += 16;
                bline3 += 16;
        }
}

void compress_data(void *args, struct video_frame * tx)
{
	/* This thread will be called from main.c and handle the compress_threads */
	struct video_compress * compress= (struct video_compress *)args;
	int x,total=0;
	int i;
	unsigned char *line1,*line2;

	line1=(unsigned char *)tx->data;
	line2=compress->output_data;
	/* First 10->8 bit conversion */
	if (bitdepth == 10) {
		for(x=0;x<HD_HEIGHT;x+=2) {
#if (HAVE_MACOSX || HAVE_32B_LINUX)
                    compress_copyline64(line2, line1, 5120/32);
		    compress_copyline64(line2+3840, line1+5120*540, 5120/32);
								
#else /* (HAVE_MACOSX || HAVE_32B_LINUX) */
		    compress_copyline128(line2, line1, 5120/32);
		    compress_copyline128(line2+3840, line1+5120*540, 5120/32);
#endif /* (HAVE_MACOSX || HAVE_32B_LINUX) */
		    line1 += 5120;
		    line2 += 2*3840;
		}
	} else {
		if (progressive == 1) {
			memcpy(line2, line1, hd_size_x*hd_size_y*hd_color_bpp);
		} else {
			for(i=0; i<1080; i+=2) {
				memcpy(line2, line1, hd_size_x*hd_color_bpp);
				memcpy(line2+hd_size_x*hd_color_bpp, line1+hd_size_x*hd_color_bpp*540, hd_size_x*hd_color_bpp);
				line1 += hd_size_x*hd_color_bpp;
				line2 += 2*hd_size_x*hd_color_bpp;
			}
		}
	}

	compress_deinterlace(compress->output_data);

	for(x=0;x<NUM_THREADS;x++) {
		compress->go[x]=1;
	}

	while(total!=1036800){
		//This is just getting silly...
		total=0;
		for(x=0;x<NUM_THREADS;x++) {
			total+=compress->len[x];
		}
	}

	tx->data=(char *)compress->out;
	tx->colour_mode=DXT_1080;
	tx->data_len=total;
}
	

static void compress_thread(void *args)
{
	struct video_compress * compress= (struct video_compress *)args;
	int myId,myEnd,myStart,range,x;
	unsigned char *retv, *input;

	pthread_mutex_lock(&(compress->lock));
	myId=compress->thread_count;
	compress->thread_count++;
	pthread_mutex_unlock(&(compress->lock));
	range=1920*1080/NUM_THREADS;
	myStart=myId*range;
	myEnd=(myId+1)*range;
	fprintf(stderr, "Thread %d online, handling elements %d - %d\n",myId,myStart,myEnd-1);


	while(1) {
		while(compress->go[myId]==0) {
			//Busywait
		}
		retv=compress->buffer[myId];
		input=(compress->output_data)+(myId*1920*1080*2/NUM_THREADS);
		/* Repack the data to YUV 4:4:4 Format */
		for(x=0;x<range;x+=2) {
			retv[4*x]=input[2*x+1];		//Y1
			retv[4*x+1]=input[2*x];		//U1
			retv[4*x+2]=input[2*x+2];	//V1
			retv[4*x+3]=255;		//Alpha

			retv[4*x+4]=input[2*x+3];	//Y2
			retv[4*x+5]=input[2*x];		//U1
			retv[4*x+6]=input[2*x+2];	//V1
			retv[4*x+7]=255;		//Alpha
		}
		compress->len[myId]=DirectDXT1(retv,(compress->out)+myId*1036800/(NUM_THREADS),1920,1080/NUM_THREADS);
		compress->go[myId]=0;

	}
}

void compress_exit(void *args)
{
	struct video_compress * compress= (struct video_compress *)args;
	int x;

	for(x=0;x<compress->thread_count;x++){
		pthread_kill(compress->thread_ids[x],SIGKILL);
	}

	free(compress->buffer[0]);
	free(compress->buffer[1]);
	free(compress->buffer[2]);
	free(compress->buffer[3]);
}
	
	
