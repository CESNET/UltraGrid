#ifndef PROFILE_TIMER_HPP
#define PROFILE_TIMER_HPP

#ifdef BUILD_PROFILED

#define C_PROFILER_PUSH(name) \
        push_prof_timer((name))

#define C_PROFILER_POP \
        pop_prof_timer()

#ifdef __cplusplus

#include <chrono>
#include <vector>
#include <mutex>
#include <thread>
#include <fstream>
#include <functional>

#define THREAD_EVENT_BUF_SIZE 256

#define PROFILE_FUNC \
        Profile_timer PROFILER_PROFILE_TIMER_FUNC(Profiler_thread_inst::get_instance(), __PRETTY_FUNCTION__); \
        Profile_timer PROFILER_PROFILE_TIMER_DETAIL(Profiler_thread_inst::get_instance(), "");

#define PROFILE_DETAIL(name) \
        PROFILER_PROFILE_TIMER_DETAIL = Profile_timer(Profiler_thread_inst::get_instance(), (name));


struct Profile_event {
        Profile_event(std::string name,
                        std::thread::id thread_id) :
                name(name),
                start(),
                end(),
                thread_id(thread_id) {  }

        Profile_event() = default;

        std::string name;
        std::chrono::steady_clock::time_point::rep start;
        std::chrono::steady_clock::time_point::rep end;

        std::thread::id thread_id;
};

class Profiler_thread_inst;

class Profiler {
        friend Profiler_thread_inst;
        public:
        static Profiler& get_instance();

        Profiler(const Profiler&) = delete;
        Profiler(Profiler&&) = delete;
        Profiler& operator=(Profiler&&) = delete;
        Profiler& operator=(const Profiler&) = delete;

        void write_events(const std::vector<Profile_event>& events);

        static std::chrono::steady_clock::time_point get_start_point(){
                return start_point;
        }

        private:
        Profiler();

        std::mutex io_mut;
        std::ofstream out_file;
        std::thread::id pid;

        bool active = false;

        static std::chrono::steady_clock::time_point start_point;
};

class Profile_timer;

class Profiler_thread_inst {
        friend Profile_timer;
        public:

        Profiler_thread_inst(Profiler& profiler);
        ~Profiler_thread_inst();

        Profiler_thread_inst(const Profiler_thread_inst&) = delete;
        Profiler_thread_inst(Profiler_thread_inst&&) = delete;
        Profiler_thread_inst& operator=(Profiler_thread_inst&&) = delete;
        Profiler_thread_inst& operator=(const Profiler_thread_inst&) = delete;

        static Profiler_thread_inst& get_instance();

        void add_event(Profile_event&& event);

        void push_timer(const char *name);
        void pop_timer();

        private:
        Profiler& profiler;
        std::vector<Profile_event> thread_events;
        std::chrono::steady_clock::time_point::rep last_ts;
        std::vector<Profile_timer> c_timers;
};

class Profile_timer {
        friend Profiler;
        public:
        Profile_timer(Profiler_thread_inst& profiler, std::string name);
        Profile_timer(Profile_timer&&);
        Profile_timer(const Profile_timer&) = delete;
        ~Profile_timer();

        Profile_timer& operator=(Profile_timer&& rhs);
        Profile_timer& operator=(const Profile_timer&) = delete;

        private:
        void commit();

        std::reference_wrapper<Profiler_thread_inst> profiler;
        Profile_event event;
};

#endif //__cplusplus

#ifdef __cplusplus
extern "C" {
#endif //__cplusplus

        void push_prof_timer(const char *name);
        void pop_prof_timer();

#ifdef __cplusplus
}
#endif //__cplusplus

#else //BUILD_PROFILED

#define PROFILE_FUNC
#define PROFILE_DETAIL(name)
#define C_PROFILER_PUSH(name)
#define C_PROFILER_POP

#endif //BUILD_PROFILED

#endif
