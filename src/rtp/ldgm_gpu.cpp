#include "config_msvc.h"
#include "config_unix.h"
#include "config_win32.h"

#include "ldgm/src/ldgm-session-gpu.h"
#include "ldgm.h"
#include "lib_common.h"

static LDGM_session_gpu *new_ldgm_session_gpu() {
        return new LDGM_session_gpu();
}

REGISTER_MODULE(ldgm_gpu, reinterpret_cast<const void *>(new_ldgm_session_gpu), LIBRARY_CLASS_UNDEFINED, LDGM_GPU_API_VERSION);

