#ifndef SPOUT_RECEIVER_H_
#define SPOUT_RECEIVER_H_

#include <GL/gl.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

#if defined _MSC_VER || defined __MINGW32__
#ifdef EXPORT_DLL_SYMBOLS
#define SPOUT_RECEIVER_DLL_API __declspec(dllexport)
#else
#define SPOUT_RECEIVER_DLL_API __declspec(dllimport)
#endif
#else // other platforms
#define SPOUT_RECEIVER_DLL_API
#endif

SPOUT_RECEIVER_DLL_API void *spout_create_receiver(char *name, unsigned int *width, unsigned int *height);
SPOUT_RECEIVER_DLL_API bool spout_receiver_recvframe(void *s, char *sender_name, unsigned int width, unsigned int height, char *data, GLenum glFormat);
SPOUT_RECEIVER_DLL_API void spout_receiver_delete(void *s);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // SPOUT_RECEIVER_H_

