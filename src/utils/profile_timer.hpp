#ifndef PROFILE_TIMER_HPP
#define PROFILE_TIMER_HPP

#ifdef BUILD_PROFILED

#ifdef __cplusplus

#include "tracy/Tracy.hpp"

#define PROFILE_FUNC ZoneScoped
#define PROFILE_NAMED_SCOPE(name) ZoneScopedN(name)
#define PROFILE_DETAIL(name) TracyMessage(name, (sizeof(name) - 1))

#else //__cplusplus

#include "tracy/TracyC.h"

typedef struct TracyCCtxWrapper{
        TracyCZoneCtx *ctx;
} TracyCCtxWrapper;

__attribute__((always_inline)) static inline void tracy_c_ctx_cleanup(TracyCCtxWrapper *wrap){
        TracyCZoneEnd(*(wrap->ctx));
}

#define C_PROFILER_PUSH(name) TracyCZoneN(tracyCCtx, name, true)
#define C_PROFILER_POP TracyCZoneEnd(tracyCCtx)

#define PROFILE_FUNC_C_NAME(name) TracyCZoneN(tracyCCtx, name, true); __attribute__((cleanup(tracy_c_ctx_cleanup))) TracyCCtxWrapper tracy_wrapper = {&tracyCCtx}

#define PROFILE_FUNC_C PROFILE_FUNC_C_NAME(__func__)

#endif

#else //BUILD_PROFILED

#define PROFILE_FUNC
#define PROFILE_DETAIL(name)
#define PROFILE_NAMED_SCOPE(name)

#define C_PROFILER_PUSH(name)
#define C_PROFILER_POP
#define PROFILE_FUNC_C_NAME(name)
#define PROFILE_FUNC_C

#endif //BUILD_PROFILED

#endif
