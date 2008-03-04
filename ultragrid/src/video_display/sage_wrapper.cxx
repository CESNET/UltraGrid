#include <GL/glut.h>
#include "sage_wrapper.h"
#include <sail.h>
#include <misc.h>

sail sageInf; // sage sail object

void initSage(int appID, int nodeID)
{
    int width = 1920;
    int height = 1080;

    sailConfig sailCfg;
    sailCfg.init("ultragrid.conf");
    sailCfg.setAppName("ultragrid");
    sailCfg.rank = nodeID;
    sailCfg.resX = width;
    sailCfg.resY = height;

    sageRect renderImageMap;
    renderImageMap.left = 0.0;
    renderImageMap.right = 1.0;
    renderImageMap.bottom = 0.0;
    renderImageMap.top = 1.0;

    sailCfg.imageMap = renderImageMap;
#ifdef SAGE_GLSL_YUV
    sailCfg.pixFmt = PIXFMT_YUV;
#else
    sailCfg.pixFmt = PIXFMT_8888_INV;
#endif
    //sailCfg.rowOrd = BOTTOM_TO_TOP;
    sailCfg.rowOrd = TOP_TO_BOTTOM;
    sailCfg.master = true;

    sageInf.init(sailCfg);
}

void sage_shutdown()
{
    sageInf.shutdown();
}

void sage_swapBuffer()
{
    sageInf.swapBuffer(SAGE_NON_BLOCKING);
}

GLubyte * sage_getBuffer()
{
    return (GLubyte *)sageInf.getBuffer();
}
