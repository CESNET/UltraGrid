/*
//      DVS Setup Header File, Copyright (c) 1998-2008 DVS Digital Video Systems AG
*/

#ifndef _DVS_SETUP_H_
#define _DVS_SETUP_H_

#ifdef _DEBUG
# ifndef DEBUG
#  define DEBUG
# endif
#endif
#ifdef DEBUG
# ifndef _DEBUG
#  define _DEBUG
# endif
#endif


/*
//	WIN16	16 bit Windows user program
//	WIN32	32 bit Windows user program
//      WIN64   64 bit Windows user program
//	VXD	32 bit Virtual device driver
//	WIN	Defined for Windows & Windows NT
//
//	LINUX	Running under linux
//	SGI	Running on SGI machine
//	SOLARIS	Running on Sun machine
//	UNIX	Defined for any unix machine
//
//	POINTER64BIT	Pointer size is 64 bit
//
*/

#if defined(_MSC_VER)
#if (_MSC_VER < 900)	/* 16 bit Microsoft Visual C */
#define WIN16	1
#define WIN	1
#endif

#if (_MSC_VER >= 900)	/* 32 bit Microsoft Visual C */
#ifndef WIN32
#define WIN32	1
#endif
#define WIN	1
#endif
#endif

#if defined(_M_AMD64)
# if !defined(WIN64)
#  define WIN64
# endif
#endif

#if defined(_M_IA64)
# if !defined(WIN64)
#  define WIN64
# endif
#endif


#if defined(linux) 
# define LINUX
# define UNIX
#endif

#if defined(sgi)
# define SGI
# define UNIX
#endif

#if defined(__sun)
# define SOLARIS
# define UNIX
#endif

#if defined(WIN32) && !defined(__CYGWIN__)
/*
//	Disable some warnings so that we can use the Warning level 4 of the MSVC
//	compiler. 
*/
#if !defined(RC_INVOKED)
#pragma warning (disable : 4001)	/* Single line comment 		*/
#pragma warning (disable : 4010)	/* Single line comment 2	*/
#pragma warning (disable : 4115)	/* Named type definition	*/
#pragma warning (disable : 4201)	/* Extension: nameless struct	*/
#pragma warning (disable : 4209)	/* Extension: benign typedef	*/
#pragma warning (disable : 4214)	/* Extension: bit field != int	*/
#pragma warning (disable : 4244)	/* Assignment truncation	*/
#pragma warning (disable : 4704)	/* Inline assembler		*/
#pragma warning (disable : 4705)	/* Statement has no effect	*/
#pragma warning (disable : 4706)	/* Assignment within if(..)	*/
#pragma warning (disable : 4100)	/* Unused parameter		*/
#pragma warning (disable : 4514)	/* Unused parameter		*/

#if (_MSC_VER >= 1400)	/* Visual Studio 8.0 */
#pragma warning (disable : 4100) // Unreferenced formal parameter
#pragma warning (disable : 4996) // Function was declared deprecated
#endif

#ifdef _DEBUG
#pragma warning (disable : 4127)
#pragma warning (disable : 4245)
#endif



//#define NOGDICAPMASKS     // CC_*, LC_*, PC_*, CP_*, TC_*, RC_
#define NOVIRTUALKEYCODES // VK_*
//#define NOWINMESSAGES     // WM_*, EM_*, LB_*, CB_*
//#define NOWINSTYLES       // WS_*, CS_*, ES_*, LBS_*, SBS_*, CBS_*
#define NOSYSMETRICS      // SM_*
//#define NOMENUS           // MF_*
//#define NOICONS           // IDI_*
#define NOKEYSTATES       // MK_*
//#define NOSYSCOMMANDS     // SC_*
#define NORASTEROPS       // Binary an Tertiardy raster ops
//#define NOSHOWWINDOW      // SW_*
#define OEMRESOURCE       // OEM Resource values
#define NOATOM            // Atom Manager routines
//#define NOCLIPBOARD       // Clipboard routines
//#define NOCOLOR           // Screen colors
//#define NOCTLMGR          // Control and Dialog routines
//#define NODRAWTEXT        // DrawText() and DT_*
//#define NOGDI             // All GDI defines and routines
//#define NOKERNEL          // All KERNEL defines and routines
//#define NOUSER            // All USER defines and routines
//#define NONLS             //- All NLS defines and routines
//#define NOMB              // MB_* and MessageBox()
//#define NOMEMMGR          // GMEM_*, LMEM_*, GHND, LHND, associated routines
#define NOMETAFILE        //- typedef METAFILEPICT
//#define NOMINMAX          //- Macros min(a,b) and max(a,b)
//#define NOMSG             //- typedef MSG and associated routines
#define NOOPENFILE        //- OpenFile(), OemToAnsi, AnsiToOem, and OF_*
//#define NOSCROLL          //- SB_* and scrolling routines
#define NOSERVICE         //- All Service Controller routines, SERVICE_ equates, etc.
#define NOSOUND           //- Sound driver routines
#define NOTEXTMETRIC      //- typedef TEXTMETRIC and associated routines
#define NOWH              //- SetWindowsHook and WH_*
//#define NOWINOFFSETS      //- GWL_*, GCL_*, associated routines
#define NOCOMM            //- COMM driver routines
#define NOKANJI           //- Kanji support stuff.
#define NOHELP            //- Help engine interface.
#define NOPROFILER        //- Profiler interface.
#define NODEFERWINDOWPOS  // DeferWindowPos routines
#define NOMCX		// Modem Configuration Extensions

#endif

/*
//	Enable strict typechecking for windows functions
*/
#ifndef STRICT
#define STRICT
#endif

#endif



#if defined(LINUX)

#define hal_DebugPrint	printk

#endif /* #if defined(linux) */


/*
//	Typedefs to be sure what length a certain type has
*/
#if !defined(WIN32) && !defined(sgi)
typedef   signed char		int8;
typedef unsigned char		uint8;
typedef   signed short		int16;	
typedef unsigned short		uint16;
typedef   signed int		int32;
typedef unsigned int		uint32;
typedef   signed long long	int64;
typedef unsigned long long	uint64;


#if defined(__alpha__)
typedef unsigned long long 	uintptr;
# define POINTER64BIT
#elif defined(SOLARIS)
# pragma align 8 (uint64, int64)
# ifdef COMPILE32BIT
typedef unsigned int		uintptr;
# else
typedef unsigned long long	uintptr;
#  pragma align 8 (uintptr)
#  define POINTER64BIT
# endif
#elif defined(__ia64__)
typedef unsigned long long 	uintptr;
# define POINTER64BIT
#elif defined(__x86_64__)
typedef unsigned long long 	uintptr;
# define POINTER64BIT
#else
typedef unsigned int		uintptr;
#endif

# ifdef linux
typedef unsigned long long	uintphysaddr;
# else
typedef uintptr			uintphysaddr;
# endif
#endif

#if defined (sgi)
typedef   signed char		int8;
typedef unsigned char		uint8;
typedef   signed short		int16;	
typedef unsigned short		uint16;
typedef   signed int		int32;
typedef unsigned int		uint32;
typedef   signed long long	int64;
typedef unsigned long long	uint64;

#if defined(DRIVER)
typedef unsigned  int	  	uint;
#endif
#if defined(COMPILE32BIT)
typedef unsigned  long int	uintptr;
#else
typedef unsigned long long	uintptr;
#define POINTER64BIT
#endif
typedef uintptr			uintphysaddr;
#endif

#if defined(WIN32) && !defined(__CYGWIN__)
typedef   signed char		int8;
typedef unsigned char	        uint8;
typedef   signed short		int16;
typedef unsigned short		uint16;
typedef   signed int		int32;
typedef unsigned int		uint32;
typedef   signed __int64	int64;
typedef unsigned __int64	uint64;

typedef unsigned int		uint;



#if defined(WIN64) || defined(_WIN64)
typedef unsigned __int64  uintptr;
#define POINTER64BIT
#else
#if (_MSC_VER >= 1400)
typedef __w64 unsigned int	uintptr;
#else
typedef unsigned int		uintptr;
#endif
#endif

typedef unsigned __int64        uintphysaddr;
#endif

#if defined(SOLARIS) 
#define memset(a,b,c) bzero(a,c) 
#define memcpy(a,b,c) bcopy(b,a,c)
#endif

#if defined(sgi) && defined(_KERNEL)
#define memset(a,b,c) bzero(a,c) 
#define memcpy(a,b,c) bcopy(b,a,c)

#ifndef __STRINGS_H__
#ifndef __SYS_SYSTM_H__
extern void     bcopy(const void *, void *, int);
extern void     bzero(void *, int);
#endif
#pragma intrinsic (bcopy)
#pragma intrinsic (bzero)
#endif

#ifndef __STRING_H__
extern char *strcpy(char *, const char *);
#pragma intrinsic (strcpy)

extern char *strcat(char *, const char *);
#endif
#endif

#ifndef donttag
# define donttag
#endif

#if !defined(offsetof) 
#define offsetof(s,m) (int)((uintptr)(&(((s *)0)->m)))
#endif

#ifndef arraysize
#define arraysize(x)  (sizeof(x)/sizeof((x)[0]))
#endif
#ifndef WIN32 
# define ARRAYSIZE(a)	(sizeof(a)/sizeof(a[0]))
#endif

#ifndef WIN32 
# ifndef min
#  define min(a,b)	(((a)<(b))?(a):(b))
# endif
#endif

#if (_MSC_VER >= 1400)	/* Visual C 8.0 */
# define dvsstrcpy(dest, destsize, source) strcpy_s(dest, destsize, source)
# define dvsstrcat(dest, destsize, source) strcat_s(dest, destsize, source)
# define dvsstrlen(string, stringsize) strlen_s(string,stringsize)
#else
# define dvsstrcpy(dest, destsize, source) strcpy(dest, source)
# define dvsstrcat(dest, destsize, source) strcat(dest, source)
# define dvsstrlen(string, stringsize) (int)strlen(string)
#endif

#endif /* #ifndef _DVS_SETUP_H */
