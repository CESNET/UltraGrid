/**
 * Copyright (c) 2011, Martin Srom
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "dxt_common.h" 
#include "dxt_util.h"

/** Documented at declaration */
GLuint
dxt_shader_create_from_source(const char* source, GLenum type)
{
    // Create shader
    GLuint shader = glCreateShader(type);
    
    // Compile shader source
    glShaderSource(shader, 1, (const GLchar**)&source, NULL);
    glCompileShader(shader);    
    
    // Check result
    GLint status;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    if ( status == GL_FALSE ) {
        char log[32768];
        glGetShaderInfoLog(shader, 32768, NULL, (GLchar*)log);
        printf("Shader Compilation Failed : %s\n", log);
        return 0;
    }
    
    return shader;
}

/** Documented at declaration */
GLhandleARB
dxt_shader_create_from_file(const char* filename, GLenum type)
{    
    // Load program from file
    FILE* file;
	file = fopen(filename, "r");
	if ( !file ) {
		fprintf(stderr, "Failed open %s for reading!\n", filename);
		return 0;
	}
    fseek(file, 0, SEEK_END);
    int data_size = ftell(file);
    rewind(file);
    char* program = (char*)malloc((data_size  + 1) * sizeof(char));
    if ( (size_t) data_size != fread(program, sizeof(char), data_size, file) ) {
        fprintf(stderr, "Failed to load program [%d bytes] from file %s!\n", data_size, filename);
        return 0;
    }
    fclose(file);
    program[data_size] = '\0';
 
    printf("Compiling program [%s]...\n", filename);
    GLhandleARB shader =  dxt_shader_create_from_source(program, type);
    if ( shader == 0 )
        return 0;
    
    free(program);
    
    return shader;
}
