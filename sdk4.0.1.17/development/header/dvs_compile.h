/**DVS*HEADER*
//
//      $Header$
//
*/

#ifndef _DVS_COMPILE_H_
#define _DVS_COMPILE_H_

/*
// Define if a card-type should be handled by separate drivers.
*/
#define COMPILE_CARDSPLIT

/*
// A specialized driver should define one or more of COMPILE_* before
// including this file.
*/
#ifdef COMPILE_CARDSPLIT
# ifdef COMPILE_SDIO
#  define COMPILECARD
# endif
# ifdef COMPILE_IRIS
#  define COMPILECARD
# endif
# ifdef COMPILE_HUGO
#  define COMPILECARD
# endif
# ifdef COMPILE_JPEG
#  define COMPILECARD
# endif
# ifdef COMPILE_HYDRAX
#  define COMPILECARD
# endif
#endif

/*
// Define which cards should be handled by the base driver.
*/
#ifdef COMPILE_CARDSPLIT
# ifndef COMPILECARD
#  define COMPILE_HUGO
#  define COMPILE_JPEG
#  define COMPILE_HYDRAX
# endif
#endif

/*
// Compile one driver for all cards.
*/
#ifndef COMPILE_CARDSPLIT
# undef  COMPILE_SDIO
# define COMPILE_SDIO
# undef  COMPILE_IRIS
# define COMPILE_IRIS
# undef  COMPILE_HUGO
# define COMPILE_HUGO
# undef  COMPILE_JPEG
# define COMPILE_JPEG
# undef  COMPILE_HYDRAX
# define COMPILE_HYDRAX
#endif

#undef COMPILECARD

#endif /* _DVS_COMPILE_H_ */
