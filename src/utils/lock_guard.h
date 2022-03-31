/**
 * @deprecated
 * Do not use these functions in new code, use rather std::lock_guard and similar
 */
#ifndef LOCK_GUARD_H_
#define LOCK_GUARD_H_

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif // HAVE_CONFIG_H

#include <pthread.h>

namespace ultragrid {

struct lock_guard_retain_ownership_t {
};

template <typename T, int (*lock_func)(T *), int (*unlock_func)(T *)>
class generic_lock_guard {
        public:
                generic_lock_guard(T &lock) :
                        m_lock(lock)
                {
                        lock_func(&m_lock);
                }

                generic_lock_guard(T &lock, lock_guard_retain_ownership_t) :
                        m_lock(lock)
                {
                }

                ~generic_lock_guard()
                {
                        unlock_func(&m_lock);
                }
        private:
                T &m_lock;
};

typedef class generic_lock_guard<pthread_mutex_t, pthread_mutex_lock, pthread_mutex_unlock>
        pthread_mutex_guard;
typedef class generic_lock_guard<pthread_rwlock_t, pthread_rwlock_wrlock, pthread_rwlock_unlock>
        pthread_rwlock_guard_write;
typedef class generic_lock_guard<pthread_rwlock_t, pthread_rwlock_rdlock, pthread_rwlock_unlock>
        pthread_rwlock_guard_read;

} // end of namespace ultragrid

#endif // LOCK_GUARD_H_

