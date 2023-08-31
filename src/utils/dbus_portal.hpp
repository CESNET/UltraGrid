#ifndef DBUS_PORTAL_HPP_511c58c91818
#define DBUS_PORTAL_HPP_511c58c91818

#include <memory>
#include <thread>
#include <stdint.h>

class ScreenCastPortal_impl;

struct ScreenCastPortalResult{
        int pipewire_fd;
        uint32_t pipewire_node;
};

class ScreenCastPortal{
public:
        ScreenCastPortal();
        ~ScreenCastPortal();

        ScreenCastPortalResult run(std::string restore_file, bool show_cursor);
private:
        std::unique_ptr<ScreenCastPortal_impl> impl;
        std::thread thread;
};

#endif
