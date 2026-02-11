#include "platform_sched.h"
#include "debug.h"

#define MOD_NAME "[sched] "

#ifdef __unix__

#include <sched.h>

bool set_realtime_sched_this_thread(){
        sched_param p = {};

        int policy = SCHED_RR;
        p.sched_priority = sched_get_priority_max(policy);
        int ret = sched_setscheduler(0, policy | SCHED_RESET_ON_FORK, &p);
        if(ret >= 0){
                log_msg(LOG_LEVEL_NOTICE, MOD_NAME "Set realtime scheduling with priority %d\n", p.sched_priority);
        } else {
                log_msg(LOG_LEVEL_WARNING, MOD_NAME "Failed to set realtime scheduling\n");
        }

        return ret >= 0;
}

#elif defined(_WIN32)

#include <windows.h>

bool set_realtime_sched_this_thread(){
        if(!SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL)){
                log_msg(LOG_LEVEL_WARNING, MOD_NAME "SetThreadPriority failed\n");
                return false;
        }

        log_msg(LOG_LEVEL_NOTICE, MOD_NAME "Set thread priority to TIME_CRITICAL\n");
        return true;
}
#else

bool set_realtime_sched_this_thread(){
        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unimplemented\n");
        return false;
}

#endif
