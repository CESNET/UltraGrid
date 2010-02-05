
#define GLSL_YCOCG 1

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>

#if defined(WIN32)
#include <io.h>
#else
#include <unistd.h>
#define GLEW_STATIC 1
#endif

#if defined(GLSL_YCOCG)

#include <GL/glew.h>


extern GLhandleARB PHandle;

int GLSLreadShader(char *fileName, int shaderType, char *shaderText, int size);
int GLSLreadShaderSource(char *fileName, GLchar **vertexShader, GLchar **fragmentShader);
GLuint GLSLinstallShaders(const GLchar *Vertex, const GLchar *Fragment);

#endif

