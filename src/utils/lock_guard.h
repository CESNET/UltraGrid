#ifndef LOCK_GUARD_H_
#define LOCK_GUARD_H_

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif // HAVE_CONFIG_H

class lock_guard {
        public:
                lock_guard(pthread_mutex_t &lock) :
                        m_lock(lock)
                {
                        pthread_mutex_lock(&m_lock);
                }

                ~lock_guard()
                {
                        pthread_mutex_unlock(&m_lock);
                }
        private:
                pthread_mutex_t &m_lock;
};

#endif // LOCK_GUARD_H_

