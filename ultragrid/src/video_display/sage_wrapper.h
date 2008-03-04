#ifndef _SAGE_WRAPPER
#define _SAGE_WRAPPER

//switching between dircet sage YUV support and yuv2rgba conversion in ug 
#define SAGE_GLSL_YUV 1

// SAGE headers
#ifdef __cplusplus
extern "C" void initSage(int appID, int nodeID);
extern "C" void sage_shutdown();
extern "C" void sage_swapBuffer();
extern "C" GLubyte * sage_getBuffer();
#else
void initSage(int appID, int nodeID);
void sage_swapBuffer();
GLubyte * sage_getBuffer();
void sage_shutdown();
#endif

#endif
