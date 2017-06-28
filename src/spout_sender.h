#ifndef SPOUT_SENDER_H_
#define SPOUT_SENDER_H_

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

#if defined _MSC_VER || defined __MINGW32__
#ifdef EXPORT_DLL_SYMBOLS
#define SPOUT_SENDER_DLL_API __declspec(dllexport)
#else
#define SPOUT_SENDER_DLL_API __declspec(dllimport)
#endif
#else // other platforms
#define SPOUT_SENDER_DLL_API
#endif

SPOUT_SENDER_DLL_API void *spout_sender_register(const char *name, int width, int height);
SPOUT_SENDER_DLL_API void spout_sender_sendframe(void *s, int width, int height, unsigned int id);
SPOUT_SENDER_DLL_API void spout_sender_unregister(void *s);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // SPOUT_SENDER_H_

