#include "platform_sched.h"
#include <sched.h>
#include "debug.h"

#define MOD_NAME "[sched] "

bool set_realtime_sched_this_thread(){
        struct sched_param p;
        memset(&p, 0, sizeof(p));

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
