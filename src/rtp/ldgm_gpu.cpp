#include "ldgm/src/ldgm-session-gpu.h"
#include "ldgm.h"
#include "lib_common.h"

static LDGM_session_gpu *new_ldgm_session_gpu() {
        return new LDGM_session_gpu();
}

static void init(void)  __attribute__((constructor));
static void init(void)
{
        register_library("ldgm_gpu", reinterpret_cast<void *>(new_ldgm_session_gpu), LIBRARY_CLASS_UNDEFINED,
                        LDGM_GPU_API_VERSION);
}

