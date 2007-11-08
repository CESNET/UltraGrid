/*
 * Copyright 1992 by Jutta Degener and Carsten Bormann, Technische
 * Universitaet Berlin.  See the accompanying file "COPYRIGHT" for
 * details.  THERE IS ABSOLUTELY NO WARRANTY FOR THIS SOFTWARE.
 *
 * $Id: gsm_impl.h,v 1.1 2007/11/08 09:48:59 hopet Exp $
 */

/*$Header: /cvs/collab/ultragrid/ultragrid/src/audio_codec/Attic/gsm_impl.h,v 1.1 2007/11/08 09:48:59 hopet Exp $*/

#ifndef	_GSM_P_H_
#define	_GSM_P_H_

#define	HAS_STDLIB_H	1		/* /usr/include/stdlib.h	*/
#define	HAS_FCNTL_H	1		/* /usr/include/fcntl.h		*/
#define	HAS_FSTAT 	1		/* fstat syscall		*/
#define	HAS_FCHMOD 	1		/* fchmod syscall		*/
#define	HAS_FCHOWN 	1		/* fchown syscall		*/
#define	HAS_STRING_H 	1		/* /usr/include/string.h 	*/
#define	HAS_UNISTD_H	1		/* /usr/include/unistd.h	*/
#define	HAS_UTIME	1		/* POSIX utime(path, times)	*/
#define	HAS_UTIME_H	1		/* UTIME header file		*/

typedef short			word;		/* 16 bit signed int	*/
typedef int			longword;	/* 32 bit signed int	*/

typedef unsigned short		uword;		/* unsigned word	*/
typedef unsigned int		ulongword;	/* unsigned longword	*/

struct gsm_state {
	word		dp0[ 280 ];
	word		z1;		/* preprocessing.c, Offset_com. */
	longword	L_z2;		/*                  Offset_com. */
	int		mp;		/*                  Preemphasis	*/
	word		u[8];		/* short_term_aly_filter.c	*/
	word		LARpp[2][8]; 	/*                              */
	word		j;		/*                              */
	word            ltp_cut;        /* long_term.c, LTP crosscorr.  */
	word		nrp; /* 40 */	/* long_term.c, synthesis	*/
	word		v[9];		/* short_term.c, synthesis	*/
	word		msr;		/* decoder.c,	Postprocessing	*/
	char		verbose;	/* only used if !NDEBUG		*/
	char		fast;		/* only used if FAST		*/
};


#define	MIN_WORD	((-32767)-1)
#define	MAX_WORD	( 32767)

#define	MIN_LONGWORD	((-2147483647)-1)
#define	MAX_LONGWORD	( 2147483647)

#ifdef	SASR		/* >> is a signed arithmetic shift right */
#undef	SASR
#define	SASR(x, by)	((x) >> (by))
#endif	/* SASR */


/*
 *	Prototypes from add.c
 */
extern word	gsm_mult 	(word a, word b);
extern longword gsm_L_mult 	(word a, word b);
extern word	gsm_mult_r	(word a, word b);

extern word	gsm_div  	(word num, word denum);

extern word	gsm_add 	( word a, word b );
extern longword gsm_L_add 	( longword a, longword b );

extern word	gsm_sub 	(word a, word b);
extern longword gsm_L_sub 	(longword a, longword b);

extern word	gsm_abs 	(word a);

extern word	gsm_norm 	( longword a );

extern longword gsm_L_asl  	(longword a, int n);
extern word	gsm_asl 	(word a, int n);

extern longword gsm_L_asr  	(longword a, int n);
extern word	gsm_asr  	(word a, int n);


# define GSM_MULT_R(a, b)	gsm_mult_r(a, b)
# define GSM_MULT(a, b)		gsm_mult(a, b)
# define GSM_L_MULT(a, b)	gsm_L_mult(a, b)

# define GSM_L_ADD(a, b)	gsm_L_add(a, b)
# define GSM_ADD(a, b)		gsm_add(a, b)
# define GSM_SUB(a, b)		gsm_sub(a, b)

# define GSM_ABS(a)		gsm_abs(a)


/*
 *  More prototypes from implementations..
 */
extern void Gsm_Coder (
		struct gsm_state	* S,
		word	* s,	/* [0..159] samples		IN	*/
		word	* LARc,	/* [0..7] LAR coefficients	OUT	*/
		word	* Nc,	/* [0..3] LTP lag		OUT 	*/
		word	* bc,	/* [0..3] coded LTP gain	OUT 	*/
		word	* Mc,	/* [0..3] RPE grid selection	OUT     */
		word	* xmaxc,/* [0..3] Coded maximum amplitude OUT	*/
		word	* xMc	/* [13*4] normalized RPE samples OUT	*/);

extern void Gsm_Long_Term_Predictor (		/* 4x for 160 samples */
		struct gsm_state * S,
		word	* d,	/* [0..39]   residual signal	IN	*/
		word	* dp,	/* [-120..-1] d'		IN	*/
		word	* e,	/* [0..40] 			OUT	*/
		word	* dpp,	/* [0..40] 			OUT	*/
		word	* Nc,	/* correlation lag		OUT	*/
		word	* bc	/* gain factor			OUT	*/);

extern void Gsm_LPC_Analysis (
		struct gsm_state * S,
		word * s,	 /* 0..159 signals	IN/OUT	*/
	        word * LARc);   /* 0..7   LARc's	OUT	*/

extern void Gsm_Preprocess (
		struct gsm_state * S,
		word * s, word * so);

extern void Gsm_Encoding (
		struct gsm_state * S,
		word	* e,	
		word	* ep,	
		word	* xmaxc,
		word	* Mc,	
		word	* xMc);

extern void Gsm_Short_Term_Analysis_Filter (
		struct gsm_state * S,
		word	* LARc,	/* coded log area ratio [0..7]  IN	*/
		word	* d	/* st res. signal [0..159]	IN/OUT	*/);

extern void Gsm_Decoder (
		struct gsm_state * S,
		word	* LARcr,	/* [0..7]		IN	*/
		word	* Ncr,		/* [0..3] 		IN 	*/
		word	* bcr,		/* [0..3]		IN	*/
		word	* Mcr,		/* [0..3] 		IN 	*/
		word	* xmaxcr,	/* [0..3]		IN 	*/
		word	* xMcr,		/* [0..13*4]		IN	*/
		word	* s);		/* [0..159]		OUT 	*/

extern void Gsm_Decoding (
		struct gsm_state * S,
		word 	xmaxcr,
		word	Mcr,
		word	* xMcr,  	/* [0..12]		IN	*/
		word	* erp); 	/* [0..39]		OUT 	*/

extern void Gsm_Long_Term_Synthesis_Filtering (
		struct gsm_state* S,
		word	Ncr,
		word	bcr,
		word	* erp,		/* [0..39]		  IN 	*/
		word	* drp); 	/* [-120..-1] IN, [0..40] OUT 	*/

void Gsm_RPE_Decoding (
		word xmaxcr,
		word Mcr,
		word * xMcr,  /* [0..12], 3 bits             IN      */
		word * erp); /* [0..39]                     OUT     */

void Gsm_RPE_Encoding (
		word    * e,            /* -5..-1][0..39][40..44     IN/OUT  */
		word    * xmaxc,        /*                              OUT */
		word    * Mc,           /*                              OUT */
		word    * xMc);        /* [0..12]                      OUT */

extern void Gsm_Short_Term_Synthesis_Filter (
		struct gsm_state * S,
		word	* LARcr, 	/* log area ratios [0..7]  IN	*/
		word	* drp,		/* received d [0...39]	   IN	*/
		word	* s);		/* signal   s [0..159]	  OUT	*/

extern void Gsm_Update_of_reconstructed_short_time_residual_signal (
		word	* dpp,		/* [0...39]	IN	*/
		word	* ep,		/* [0...39]	IN	*/
		word	* dp);		/* [-120...-1]  IN/OUT 	*/

/*
 *  Tables from table.c
 */
#ifndef	GSM_TABLE_C

extern word gsm_A[8], gsm_B[8], gsm_MIC[8], gsm_MAC[8];
extern word gsm_INVA[8];
extern word gsm_DLB[4], gsm_QLB[4];
extern word gsm_H[11];
extern word gsm_NRFAC[8];
extern word gsm_FAC[8];

#endif	/* GSM_TABLE_C */

/*
 *  Debugging
 */
#ifdef NDEBUG

#	define	gsm_debug_words(a, b, c, d)		/* nil */
#	define	gsm_debug_longwords(a, b, c, d)		/* nil */
#	define	gsm_debug_word(a, b)			/* nil */
#	define	gsm_debug_longword(a, b)		/* nil */

#else	/* !NDEBUG => DEBUG */

	extern void  gsm_debug_words     (char * name, int, int, word *);
	extern void  gsm_debug_longwords (char * name, int, int, longword *);
	extern void  gsm_debug_longword  (char * name, longword);
	extern void  gsm_debug_word      (char * name, word);

#endif /* !NDEBUG */

typedef struct gsm_state * 	gsm;
typedef short		   	gsm_signal;		/* signed 16 bit */
typedef unsigned char		gsm_byte;
typedef gsm_byte 		gsm_frame[33];		/* 33 * 8 bits	 */

#define GSM_FRAMESIZE   33
#define	GSM_MAGIC	0xD			  	/* 13 kbit/s RPE-LTP */

#define	GSM_PATCHLEVEL	7
#define	GSM_MINOR	0
#define	GSM_MAJOR	1

#define	GSM_OPT_VERBOSE	1
#define	GSM_OPT_FAST	2
#define	GSM_OPT_LTP_CUT	3

extern gsm  gsm_create 	(void);
extern void gsm_destroy (gsm);	
extern int  gsm_print   (FILE *, gsm, gsm_byte  *);
extern int  gsm_option  (gsm, int, int *);
extern void gsm_encode  (gsm, gsm_signal *, gsm_byte  *);
extern int  gsm_decode  (gsm, gsm_byte   *, gsm_signal *);
extern int  gsm_explode (gsm, gsm_byte   *, gsm_signal *);
extern void gsm_implode (gsm, gsm_signal *, gsm_byte   *);

#endif	/* _GSM_P_H_ */
