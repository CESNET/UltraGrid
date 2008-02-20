#include "config.h"
#include "config_unix.h"
#include "config_win32.h"

#include "debug.h"
#include "video_display.h"
#include "video_display/sage.h"


#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glext.h>

#include <X11/Xlib.h>
#include <sys/time.h>
#include <assert.h>

#include <semaphore.h>
#include <signal.h>
#include <pthread.h>

#include "video_display/sage_wrapper.h"
#include <host.h>

#define HD_WIDTH 	1920
#define HD_HEIGHT 	1080
#define MAGIC_SAGE	DISPLAY_SAGE_ID


struct state_sdl {
	GLubyte			*buffers[2];
	GLubyte			*outBuffer;
	GLubyte			*yuvBuffer;
	int			 image_display, image_network;
	/* Thread related information follows... */
	pthread_t		 thread_id;
	pthread_mutex_t		 lock;
	pthread_cond_t		 boss_cv;
	pthread_cond_t		 worker_cv;
	sem_t			 semaphore;
	int			 work_to_do;
	int			 boss_waiting;
	int			 worker_waiting;
	/* For debugging... */
	uint32_t		 magic;	
};

int *_0080;
int *_00ff;

//FIXME mplayeri pouzivaji jednu hodnotu typu uint64_t
//Cb = U, Cr = V
int *_YUV_Coef;
#define RED_v   "0*16"    //+ 1.596
#define GREEN_u "1*16"    //- 0.391
#define GREEN_v "2*16"    //- 0.813
#define BLUE_u  "3*16"    //+ 2.018
#define Y_COEF  "4*16"    //+ 1.164 

/** Prototyping */
inline void sage_copyline64(unsigned char *dst, unsigned char *src, int len);
inline void sage_copyline128(unsigned char *dst, unsigned char *src, int len);
inline void swapBytes(unsigned char *buf, int len);
void sage_deinterlace(unsigned char *buffer);
void sage_gl_resize_window(int width,int height);
void yuv2rgba(unsigned char *buffer, unsigned char *rgbBuf);


int display_sage_handle_events(void)
{
#if 0
	SDL_Event	sdl_event;
	while (SDL_PollEvent(&sdl_event)) {
		switch (sdl_event.type) {
			case SDL_KEYDOWN:
			case SDL_KEYUP:
				if (!strcmp(SDL_GetKeyName(sdl_event.key.keysym.sym), "q")) {
					kill(0, SIGINT);
				}
				break;

			default:
				break;
		}
	}
#endif
	return 0;

}

/* TODO:
 * dalo by se optimalizovat s pouzitim vice registru. Napr. prolozit ten kod jeste jednou stejnou instanci
 * s vyuzitim dalsich 8 registru. Melo by se kontrolovat, zda je sirka vstupu delitelna 32 a zda je
 * vstup i vystup zarovnan na 16 bytu. */
void yuv2rgba(unsigned char *buffer, unsigned char *rgbBuf)
{
    int i = 0;
    int j = 0;
    for (i = 0; i < HD_WIDTH*HD_HEIGHT*2; i+=32)
    {
        if (bitdepth == 8) {
            j++; // interlace
            if (j == 121) buffer += 2069760;
            if (j == 241) { buffer -=2073600; j = 1; }
        }

    asm (
        "psubq %%xmm5, %%xmm5\n"
        "movdqa (%0), %%xmm0\n"     // U0 Y0 V0 Y1 U1 Y2 V1 Y3 U2 Y4 V2 Y5 U3 Y6 V3 Y7
        "movdqa 16(%0), %%xmm1\n"   // S0 X0 T0 X1 S1 X2 T1 X3 S2 X4 T2 X5 S3 X6 T3 X7
	"movdqa (%2), %%xmm2\n"
        /** prepare Y*/
        "movdqa %%xmm0, %%xmm3\n"       //kopie
        "movdqa %%xmm1, %%xmm4\n"       
        "pand %%xmm2, %%xmm3\n"    // 00 Y0 00 Y1 00 Y2 00 Y3 00 Y4 00 Y5 00 Y6 00 Y7
        "pand %%xmm2, %%xmm4\n"    // 00 X0 00 X1 00 X2 00 X3 00 X4 00 X5 00 X6 00 X7
        "psrldq $1, %%xmm3\n"      // Y0 00 Y1 00 Y2 00 Y3 00 Y4 00 Y5 00 Y6 00 Y7 00
        "psrldq $1, %%xmm4\n"      // X0 00 X1 00 X2 00 X3 00 X4 00 X5 00 X6 00 X7 00
        "packuswb %%xmm4, %%xmm3\n"// Y0 Y1 Y2 Y3 Y4 Y5 Y6 Y7 X0 X1 X2 X3 X4 X5 X6 X7
        /** prepare u*/
        "psrldq $1, %%xmm2\n"        //prepare U mask
        "punpcklwd %%xmm5, %%xmm2\n" //xmm2: 0x000000ff000000ff000000ff000000ff
        "movdqa %%xmm0, %%xmm5\n"    //kopie puvodnich registru
        "movdqa %%xmm1, %%xmm6\n"       
        "pand %%xmm2, %%xmm5\n"      // U0 00 00 00 U1 00 00 00 U2 00 00 00 U3 00 00 00
        "pand %%xmm2, %%xmm6\n"      // S0 00 00 00 S1 00 00 00 S2 00 00 00 S3 00 00 00
        "packssdw %%xmm6, %%xmm5\n"  // U0 00 U1 00 U2 00 U3 00 S0 00 S1 00 S2 00 S3 00
        "pslldq $1, %%xmm5\n"        // 00 U0 00 U1 00 U2 00 U3 00 S0 00 S1 00 S2 00 S3
        /** prepare v*/
        "pslldq $2, %%xmm2\n"     //xmm2: 0x0000ff000000ff000000ff000000ff00
        "movdqa %%xmm0, %%xmm6\n" //kopie puvodnich registru
        "movdqa %%xmm1, %%xmm7\n"       
        "pand %%xmm2, %%xmm6\n"   // 00 00 V0 00 00 00 V1 00 00 00 V2 00 00 00 V3 00
        "pand %%xmm2, %%xmm7\n"   // 00 00 T0 00 00 00 T1 00 00 00 T2 00 00 00 T3 00
        "psrldq $2, %%xmm6\n"     //shift <<
        "psrldq $2, %%xmm7\n" 
        "packssdw %%xmm7, %%xmm6\n" // 00 V0 00 V1 00 V2 00 V3 00 T0 00 T1 00 T2 00 T3
	"psllw  $8, %%xmm6\n"      //  V0 00 V1 00 V2 00 V3 00 T0 00 T1 00 T2 00 T3 00

        "movdqa (%3), %%xmm2\n"  
        "psubw %%xmm2, %%xmm5\n" /* u -= 128 */ 
        "psubw %%xmm2, %%xmm6\n" /* v -= 128 */
	"psrlw $3, %%xmm2\n"     /* 128 -> 16 (0x8000 -> 0x1000) */

        "movdqa %%xmm5, %%xmm0\n" //mov u -> xmm0
        "movdqa %%xmm6, %%xmm1\n" //mov v -> xmm1

        "pmulhw "GREEN_u"(%4), %%xmm0\n" /*Mul Green u coef -> Green u*/
        "pmulhw "GREEN_v"(%4), %%xmm1\n" /*Mul Green v coef -> Green v*/

        "pmulhw "BLUE_u"(%4), %%xmm5\n" /*Mul Blue u coef -> Blue u*/
        "pmulhw "RED_v"(%4), %%xmm6\n" /*Mul Red v coef -> Red v*/

        "paddsw %%xmm1, %%xmm0\n"  /*Green u + Green v -> CGreen */

        /** process luma*/
        "movdqa (%2), %%xmm4\n"   //mov 00ff* mask to  xmm2
        "movdqa %%xmm3, %%xmm1\n" //copy Y0 Y1 Y2 Y3 Y4 Y5 Y6 Y7 X0 X1 X2 X3 X4 X5 X6 X7
	"pand %%xmm4, %%xmm3\n"   // get Y odd Y1 00 Y3 00 Y5 00 Y7 00 X1 00 X3 00 X5 00 X7 00
	"psrldq $1, %%xmm4\n"
	"pand %%xmm4, %%xmm1\n"   
	"psllw $8, %%xmm1\n"      //get Y even  Y0 00 Y2 00 Y4 00 Y6 00 X0 00 X2 00 X4 00 X6 00

	"psubusw %%xmm2, %%xmm1\n" // Y -= 16
	"psubusw %%xmm2, %%xmm3\n" // Y -= 16

	"pmulhuw "Y_COEF"(%4), %%xmm1\n"  // Y = (Y-16)*1.164
	"pmulhuw "Y_COEF"(%4), %%xmm3\n"  // Y = (Y-16)*1.164

        // xmm0 -> green, xmm5 -> blue, xmm6 -> red
        // xmm1 -> Y even, xmm3 -> Y odd
        "movdqa %%xmm0, %%xmm2\n" // copy green
        "movdqa %%xmm5, %%xmm4\n" // copy blue
        "movdqa %%xmm6, %%xmm7\n" // copy red

        "paddsw %%xmm1, %%xmm5\n" // Y even + blue
        "paddsw %%xmm3, %%xmm4\n" // Y odd + blue

        "paddsw %%xmm1, %%xmm6\n" // Y even + red
        "paddsw %%xmm3, %%xmm7\n" // Y odd + red

	"paddsw %%xmm1, %%xmm0\n" // Y even - green
	"paddsw %%xmm3, %%xmm2\n" // Y odd - green

        /*Limit RGB even to 0..255*/
        "packuswb %%xmm0, %%xmm0\n" /* B6 B4 B2 B0  B6 B4 B2 B0 */
        "packuswb %%xmm5, %%xmm5\n" /* R6 R4 R2 R0  R6 R4 R2 R0 */
        "packuswb %%xmm6, %%xmm6\n" /* G6 G4 G2 G0  G6 G4 G2 G0 */

        /*Limit RGB odd to 0..255*/
        "packuswb %%xmm2, %%xmm2\n" /* B7 B5 B3 B1  B7 B5 B3 B1 */
        "packuswb %%xmm4, %%xmm4\n" /* R7 R5 R3 R1  R7 R5 R3 R1 */
        "packuswb %%xmm7, %%xmm7\n" /* G7 G5 G3 G1  G7 G5 G3 G1 */

        /*Interleave RGB even and odd */
        "punpcklbw %%xmm7, %%xmm6\n" //RED
        "punpcklbw %%xmm2, %%xmm0\n" //GREEN
        "punpcklbw %%xmm4, %%xmm5\n" //BLUE

        "movdqa %%xmm6, %%xmm1\n" //copy R
        "movdqa %%xmm5, %%xmm2\n" //copy B
        "pxor %%xmm4, %%xmm4\n"
        /*unpack high qword*/
        "punpckhbw %%xmm0, %%xmm5\n" //bg
        "punpckhbw %%xmm4, %%xmm6\n" //r0
        "movdqa %%xmm5, %%xmm7\n"
        "punpckhwd %%xmm6, %%xmm5\n" //bgr0
        "punpcklwd %%xmm6, %%xmm7\n" //bgr0 low
        /*unpack low qword*/
        "punpcklbw %%xmm0, %%xmm2\n" //bg
        "punpcklbw %%xmm4, %%xmm1\n" //r0
        "movdqa %%xmm2, %%xmm6\n"
        "punpckhwd %%xmm1, %%xmm2\n" //bgr0
        "punpcklwd %%xmm1, %%xmm6\n" //bgr0 low

        /* save */
        "movdqa %%xmm6, (%1)\n"
        "movdqa %%xmm2, 16(%1)\n"
        "movdqa %%xmm7, 32(%1)\n"
        "movdqa %%xmm5, 48(%1)\n"

        :
        : "r" ((unsigned long *) buffer),
          "r" ((unsigned long *) rgbBuf),
          "r" ((unsigned long *) _00ff),
          "r" ((unsigned long *) _0080),
          "r" ((unsigned long *) _YUV_Coef)
    );

    buffer += 32;
    rgbBuf += 64;
    }
}

/* linear blend sage_deinterlace */
void sage_deinterlace(unsigned char *buffer)
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

inline void swapBytes(unsigned char *buf, int len)
{
    register unsigned char b1;
    int i =0;

    do {
        b1 = buf[i];
        buf[i] = buf[i+1];
        buf[i+1] = b1;
    } while ((i += 2)<len);

//    i = open("/tmp/dump", O_CREAT|O_WRONLY,0644);
//    write(i, buf, len);
//    close(i);
}

inline void sage_copyline64(unsigned char *dst, unsigned char *src, int len)
{
    register unsigned long *d, *s;
    register unsigned long a1,a2,a3,a4;

    d = (unsigned long *)dst;
    s = (unsigned long *)src;

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

/* convert 10bits Cb Y Cr A Y Cb Y A to 8bits Cb Y Cr Y Cb Y */

#ifndef HAVE_MACOSX

inline void sage_copyline128(unsigned char *d, unsigned char *s, int len)
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

#endif /* HAVE_MACOSX */


static void* display_thread_sage(void *arg)
{
	struct state_sdl *s = (struct state_sdl *) arg;
	int i;

	while (1) {
		GLubyte *line1, *line2;
		//display_sage_handle_events();

		sem_wait(&s->semaphore);

		assert(s->outBuffer != NULL);
        
        if (bitdepth == 10)  {
            line1 = s->buffers[s->image_display];
            line2 = s->yuvBuffer;
            
            for(i=0; i<1080; i+=2) {
#ifdef HAVE_MACOSX
                sage_copyline64(line2, line1, 5120/32);
                sage_copyline64(line2+3840, line1+5120*540, 5120/32);
#else /* HAVE_MACOSX */
                sage_copyline128(line2, line1, 5120/32);
                sage_copyline128(line2+3840, line1+5120*540, 5120/32);
#endif /* HAVE_MACOSX */
                line1 += 5120;
                line2 += 2*3840;
            }
            yuv2rgba(s->yuvBuffer, s->outBuffer);
        } 
        else  {
            yuv2rgba(s->buffers[s->image_display], s->outBuffer);
        }
//        swapBytes(s->outBuffer, HD_WIDTH*HD_HEIGHT*2);
//		int i = open("/tmp/testcard_image.rgba_c", O_WRONLY|O_CREAT, 0644);
//		write(i,s->buffers[s->image_display], HD_WIDTH*HD_HEIGHT*2);
//		close(i);
        sage_swapBuffer();
        s->outBuffer = sage_getBuffer();
	}
	return NULL;
}


void * display_sage_init(void)
{
	struct state_sdl	*s;

	s = (struct state_sdl *) malloc(sizeof(struct state_sdl));
	s->magic   = MAGIC_SAGE;

	//iopl(3);

	debug_msg("Window initialized %p\n", s);

    /** yuv2rgb constants init */
    posix_memalign((void *)&_YUV_Coef, 16, 80+32);
    _YUV_Coef[0] = _YUV_Coef[1] = _YUV_Coef[2] = _YUV_Coef[3] = 0x01990199;// RED_v 1.596
    _YUV_Coef[4] = _YUV_Coef[5] = _YUV_Coef[6] = _YUV_Coef[7] = 0xff9cff9c;// GREEN_u 0.391
    _YUV_Coef[8] = _YUV_Coef[9] = _YUV_Coef[10] = _YUV_Coef[11] = 0xff30ff30; // GREEN_v 0.813
    _YUV_Coef[12] = _YUV_Coef[13] = _YUV_Coef[14] = _YUV_Coef[15] = 0x02050205; // BLUE_u  2.018
    _YUV_Coef[16] = _YUV_Coef[17] = _YUV_Coef[18] = _YUV_Coef[19] = 0x012a012a; // Y 1.164 
    _00ff = & _YUV_Coef[20];
    _0080 = & _YUV_Coef[24];
    _00ff[0] = _00ff[1] = _00ff[2] = _00ff[3] = 0xff00ff00;
    _0080[0] = _0080[1] = _0080[2] = _0080[3] = 0x80008000;

	s->buffers[0] = malloc(HD_WIDTH*HD_HEIGHT*3);
	s->buffers[1] = malloc(HD_WIDTH*HD_HEIGHT*3);
	s->yuvBuffer = malloc(HD_WIDTH*HD_HEIGHT*2);
    s->image_network=0;
    s->image_display=1;

	asm("emms\n");

    /* sage init */
    //FIXME sem se musi propasovat ty spravne parametry argc argv
    int appID;
//	if (argc < 2)
	    appID = 0;
//	else
//        appID = atoi(argv[1]);
	   
	int nodeID;
//	if (argc < 3)
        nodeID = 1;
//	else
//       nodeID = atoi(argv[2]);
    initSage(appID, nodeID);
    s->outBuffer = sage_getBuffer();

    /* thread init */
	pthread_mutex_init(&s->lock, NULL);
	pthread_cond_init(&s->boss_cv, NULL);
	pthread_cond_init(&s->worker_cv, NULL);
	sem_init(&s->semaphore, 0, 0);
	s->work_to_do     = FALSE;
	s->boss_waiting   = FALSE;
	s->worker_waiting = TRUE;
	if (pthread_create(&(s->thread_id), NULL, display_thread_sage, (void *) s) != 0) {
		perror("Unable to create display thread\n");
		return NULL;
	}

	return (void *)s;
}

void display_sage_done(void *state)
{
	struct state_sdl *s = (struct state_sdl *) state;

	assert(s->magic == MAGIC_SAGE);
    sage_shutdown();
}

char * display_sage_getf(void *state)
{
	struct state_sdl *s = (struct state_sdl *) state;
	assert(s->magic == MAGIC_SAGE);
	return (char *)s->buffers[s->image_network];
}

int display_sage_putf(void *state, char *frame)
{
	int		 tmp;
	struct state_sdl *s = (struct state_sdl *) state;

	assert(s->magic == MAGIC_SAGE);
	UNUSED(frame);

	/* ...and give it more to do... */
	tmp = s->image_display;
	s->image_display = s->image_network;
	s->image_network = tmp;
	s->work_to_do    = TRUE;

	/* ...and signal the worker */
	sem_post(&s->semaphore);
	sem_getvalue(&s->semaphore, &tmp);
	if(tmp > 1) 
		printf("frame drop!\n");
	return 0;
}

display_colour_t display_sage_colour(void *state)
{
	struct state_sdl *s = (struct state_sdl *) state;
	assert(s->magic == MAGIC_SAGE);
	return DC_YUV;
}

display_type_t * display_sage_probe(void)
{
    display_type_t          *dt;
    display_format_t        *dformat;

    dformat = malloc(4 * sizeof(display_format_t));
    dformat[0].size        = DS_176x144;
    dformat[0].colour_mode = DC_YUV;
    dformat[0].num_images  = 1;
    dformat[1].size        = DS_352x288;
    dformat[1].colour_mode = DC_YUV;
    dformat[1].num_images  = 1;
    dformat[2].size        = DS_702x576;
    dformat[2].colour_mode = DC_YUV;
    dformat[2].num_images  = 1;
    dformat[3].size        = DS_1280x720;
    dformat[3].colour_mode = DC_YUV;
    dformat[3].num_images  = 1;

    dt = malloc(sizeof(display_type_t));
    if (dt != NULL) {
        dt->id	        = DISPLAY_SAGE_ID;
        dt->name        = "sage";
        dt->description = "SAGE";
        dt->formats     = dformat;
        dt->num_formats = 4;
    }
    return dt;
}
