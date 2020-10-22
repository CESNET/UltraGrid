#ifndef UG_LIB_87DD87FE_8111_4485_B05F_805029047F81
#define UG_LIB_87DD87FE_8111_4485_B05F_805029047F81

#if ! defined __cplusplus
#include <stddef.h>
#else
#include <cstddef>
extern "C" {
#endif

#if (defined(_MSC_VER) || defined(__MINGW32__))
#ifdef LIBUG_EXPORT
#define LIBUG_DLL __declspec(dllexport)
#else
#define LIBUG_DLL __declspec(dllimport)
#endif
#else
#define LIBUG_DLL
#endif

typedef enum {
        UG_VIDEO_CODEC_NONE = 0, ///< dummy color spec
        UG_RGBA,     ///< RGBA 8-bit
} ug_codec_t;

struct ug_sender;

LIBUG_DLL struct ug_sender *ug_sender_init(const char *receiver, int mtu);
LIBUG_DLL void ug_send_frame(struct ug_sender *state, const char *data, size_t len, ug_codec_t codec, int width, int height);
LIBUG_DLL void ug_sender_done(struct ug_sender *state);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // defined UG_LIB_87DD87FE_8111_4485_B05F_805029047F81
