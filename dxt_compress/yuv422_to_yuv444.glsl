#version 130

#extension GL_EXT_gpu_shader4 : enable

#define lerp mix

in vec4 TEX0;
uniform sampler2D image;
uniform vec2 imageSize;
out vec4 color;

void main()
{
        color.rgba  = texture(image, TEX0.xy).grba; // store Y0UVY1 ro rgba
        if(TEX0.x * imageSize.x / 2 - floor(TEX0.x * imageSize.x / 2) > 0.25)
                color.r = color.a; // use Y1 instead of Y0
}
