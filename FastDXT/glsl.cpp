
#include "glsl.h"

#if defined(GLSL_YCOCG)

GLhandleARB FSHandle,PHandle;

/*---------------------------------------------------------------------------*/


#define GLSLVertexShader   1
#define GLSLFragmentShader 2

int GLSLprintlError(char *file, int line)
{
    //
    // Returns 1 if an OpenGL error occurred, 0 otherwise.
    //
    GLenum glErr;
    int    retCode = 0;

    glErr = glGetError();
    while (glErr != GL_NO_ERROR)
    {
        fprintf(stderr, "GLSL> glError in file %s @ line %d: %s\n", file, line, gluErrorString(glErr));
        retCode = 1;
        glErr = glGetError();
    }
    return retCode;
}

//
// Print out the information log for a shader object
//
static
void GLSLprintShaderInfoLog(GLuint shader)
{
    GLint infologLength = 0;
    GLint charsWritten  = 0;
    GLchar *infoLog;

    GLSLprintlError(__FILE__, __LINE__);  // Check for OpenGL errors

    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infologLength);

    GLSLprintlError(__FILE__, __LINE__);  // Check for OpenGL errors

    if (infologLength > 0)
    {
        infoLog = (GLchar *)malloc(infologLength);
        if (infoLog == NULL)
        {
            fprintf(stderr, "GLSL> ERROR: Could not allocate InfoLog buffer\n");
            exit(1);
        }
        glGetShaderInfoLog(shader, infologLength, &charsWritten, infoLog);
        fprintf(stderr, "GLSL> Shader InfoLog:%s\n", infoLog);
        free(infoLog);
    }
    GLSLprintlError(__FILE__, __LINE__);  // Check for OpenGL errors
}

//
// Print out the information log for a program object
//
static
void GLSLprintProgramInfoLog(GLuint program)
{
    GLint infologLength = 0;
    GLint charsWritten  = 0;
    GLchar *infoLog;

    GLSLprintlError(__FILE__, __LINE__);  // Check for OpenGL errors

    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &infologLength);

    GLSLprintlError(__FILE__, __LINE__);  // Check for OpenGL errors

    if (infologLength > 0)
    {
        infoLog = (GLchar *)malloc(infologLength);
        if (infoLog == NULL)
        {
            fprintf(stderr, "GLSL> ERROR: Could not allocate InfoLog buffer\n");
            exit(1);
        }
        glGetProgramInfoLog(program, infologLength, &charsWritten, infoLog);
        fprintf(stderr, "GLSL> Program InfoLog:%s\n", infoLog);
        free(infoLog);
    }
    GLSLprintlError(__FILE__, __LINE__);  // Check for OpenGL errors
}

static
int GLSLShaderSize(char *fileName, int shaderType)
{
    //
    // Returns the size in bytes of the shader fileName.
    // If an error occurred, it returns -1.
    //
    // File name convention:
    //
    // <fileName>.vert
    // <fileName>.frag
    //
    int fd;
    char name[256];
    int count = -1;

        memset(name, 0, 256);
    strcpy(name, fileName);

    switch (shaderType)
    {
        case GLSLVertexShader:
            strcat(name, ".vert");
            break;
        case GLSLFragmentShader:
            strcat(name, ".frag");
            break;
        default:
            fprintf(stderr, "GLSL> ERROR: unknown shader file type\n");
            exit(1);
            break;
    }

    //
    // Open the file, seek to the end to find its length
    //
#ifdef WIN32
    fd = _open(name, _O_RDONLY);
    if (fd != -1)
    {
        count = _lseek(fd, 0, SEEK_END) + 1;
        _close(fd);
    }
#else
    fd = open(name, O_RDONLY);
    if (fd != -1)
    {
        count = lseek(fd, 0, SEEK_END) + 1;
        close(fd);
    }
#endif
    return count;
}


int GLSLreadShader(char *fileName, int shaderType, char *shaderText, int size)
{
    //
    // Reads a shader from the supplied file and returns the shader in the
    // arrays passed in. Returns 1 if successful, 0 if an error occurred.
    // The parameter size is an upper limit of the amount of bytes to read.
    // It is ok for it to be too big.
    //
    FILE *fh;
    char name[100];
    int count;

    strcpy(name, fileName);

    switch (shaderType)
    {
        case GLSLVertexShader:
            strcat(name, ".vert");
            break;
        case GLSLFragmentShader:
            strcat(name, ".frag");
            break;
        default:
            fprintf(stderr, "GLSL> ERROR: unknown shader file type\n");
            exit(1);
            break;
    }

    //
    // Open the file
    //
    fh = fopen(name, "r");
    if (!fh)
        return -1;


    //
    // Get the shader from a file.
    //
    fseek(fh, 0, SEEK_SET);
    count = (int) fread(shaderText, 1, size, fh);
    shaderText[count] = '\0';

    if (ferror(fh))
        count = 0;

    fclose(fh);
    return count;
}

int GLSLreadShaderSource(char *fileName, GLchar **vertexShader, GLchar **fragmentShader)
{
    int vSize, fSize;

    //
    // Allocate memory to hold the source of our shaders.
    //

    fprintf(stderr, "GLSL> load shader %s\n", fileName);
    if (vertexShader) {
        vSize = GLSLShaderSize(fileName, GLSLVertexShader);

        if (vSize == -1) {
            fprintf(stderr, "GLSL> Cannot determine size of the vertex shader %s\n", fileName);
            return 0;
        }

        *vertexShader = (GLchar *) malloc(vSize);

        //
        // Read the source code
        //
        if (!GLSLreadShader(fileName, GLSLVertexShader, *vertexShader, vSize)) {
            fprintf(stderr, "GLSL> Cannot read the file %s.vert\n", fileName);
            return 0;
        }
    }

    if (fragmentShader) {
        fSize = GLSLShaderSize(fileName, GLSLFragmentShader);

        if (fSize == -1) {
            fprintf(stderr, "GLSL> Cannot determine size of the fragment shader %s\n", fileName);
            return 0;
        }

        *fragmentShader = (GLchar *) malloc(fSize);

        if (!GLSLreadShader(fileName, GLSLFragmentShader, *fragmentShader, fSize)) {
            fprintf(stderr, "GLSL> Cannot read the file %s.frag\n", fileName);
            return 0;
        }
    }

    return 1;
}


GLuint GLSLinstallShaders(const GLchar *Vertex, const GLchar *Fragment)
{
        GLuint VS, FS, Prog;   // handles to objects
        GLint  vertCompiled, fragCompiled;    // status values
        GLint  linked;

        // Create a program object
        Prog = glCreateProgram();

        // Create a vertex shader object and a fragment shader object
        if (Vertex) {
                VS = glCreateShader(GL_VERTEX_SHADER);

                // Load source code strings into shaders
                glShaderSource(VS, 1, &Vertex, NULL);

                // Compile the vertex shader, and print out
                // the compiler log file.

                glCompileShader(VS);
                GLSLprintlError(__FILE__, __LINE__);  // Check for OpenGL errors
                glGetShaderiv(VS, GL_COMPILE_STATUS, &vertCompiled);
                GLSLprintShaderInfoLog(VS);

                if (!vertCompiled)
                  return 0;

                glAttachShader(Prog, VS);
        }


        if (Fragment) {
                FS = glCreateShader(GL_FRAGMENT_SHADER);

                glShaderSource(FS, 1, &Fragment, NULL);

                glCompileShader(FS);
                GLSLprintlError(__FILE__, __LINE__);  // Check for OpenGL errors
                glGetShaderiv(FS, GL_COMPILE_STATUS, &fragCompiled);
                GLSLprintShaderInfoLog(FS);

                if (!fragCompiled)
                  return 0;

                glAttachShader(Prog, FS);
        }

        // Link the program object and print out the info log
        glLinkProgram(Prog);
        GLSLprintlError(__FILE__, __LINE__);  // Check for OpenGL errors
        glGetProgramiv(Prog, GL_LINK_STATUS, &linked);
        GLSLprintProgramInfoLog(Prog);

        if (!linked)
          return 0;

        return Prog;
}

#endif

