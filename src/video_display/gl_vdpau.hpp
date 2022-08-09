#ifndef GL_VDPAU_HPP_667244de5757
#define GL_VDPAU_HPP_667244de5757

#ifdef HWACC_VDPAU

#ifdef HAVE_MACOSX
#include <OpenGL/gl.h>
#include <OpenGL/OpenGL.h> // CGL
#include <OpenGL/glext.h>
#else
#include <GL/glew.h>
#endif /* HAVE_MACOSX */

#include "hwaccel_vdpau.h"
typedef GLintptr vdpauSurfaceNV;
#define NV_CAST(x) ((void *)(uintptr_t)(x))

struct state_vdpau {
        bool initialized = false;
        GLuint textures[4] = {0};
        hw_vdpau_frame lastFrame;

        bool interopInitialized = false;
        bool mixerInitialized = false;
        VdpDevice device = VDP_INVALID_HANDLE;
        VdpGetProcAddress *get_proc_address = nullptr;
        vdpauSurfaceNV vdpgl_surf = 0;

        VdpOutputSurface out_surf = VDP_INVALID_HANDLE;
        VdpVideoMixer mixer = VDP_INVALID_HANDLE;

        uint32_t surf_width = 0;
        uint32_t surf_height = 0;
        VdpChromaType surf_ct = 0;

        bool init();
        void checkInterop(VdpDevice device, VdpGetProcAddress *get_proc_address);
        void initInterop(VdpDevice device, VdpGetProcAddress *get_proc_address);
        void uninitInterop();

        void loadFrame(hw_vdpau_frame *frame);

        void initMixer(uint32_t w, uint32_t h, VdpChromaType ct);
        void mixerRender(VdpVideoSurface f);
        void uninitMixer();

        void uninit();

        vdp_funcs funcs;

};



#endif //HWACC_VDPAU
#endif //GL_VDPAU_HPP_667244de5757
