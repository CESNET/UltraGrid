#version 130

#extension GL_ARB_texture_non_power_of_two : enable

#define lerp mix

in vec4 TEX0;
uniform sampler2D image;
uniform float imageWidth; // is original image width, it means twice as wide as ours

void main()
{
        vec4 rgba1, rgba2;
        vec4 yuv1, yuv2;
        vec2 coor1, coor2;
        float U, V;

        coor1 = gl_TexCoord[0].xy - vec2(1.0 / (imageWidth * 2), 0.0);
        coor2 = gl_TexCoord[0].xy + vec2(1.0 / (imageWidth * 2), 0.0);

        rgba1  = texture(image, coor1);
        rgba2  = texture(image, coor2);
        
        yuv1.x = rgba1.r * 0.299 + rgba1.g * 0.587 + rgba1.b * 0.114; // Y
        yuv1.y = 0.5-rgba1.r * 0.168736 - rgba1.g * 0.331264 + rgba1.b * 0.5;
        yuv1.z = 0.5+rgba1.r * 0.5 - rgba1.g * 0.418688 - rgba1.b * 0.081312;
        
        yuv2.x = rgba2.r * 0.299 + rgba2.g * 0.587 + rgba2.b * 0.114; // Y
        yuv2.y = 0.5-rgba2.r * 0.168736 - rgba2.g * 0.331264 + rgba2.b * 0.5;
        yuv2.z = 0.5+rgba2.r * 0.5 - rgba2.g * 0.418688 - rgba2.b * 0.081312;
        
        U = mix(yuv1.y, yuv2.y, 0.5);
        V = mix(yuv1.z, yuv2.z, 0.5);
        
        gl_FragColor = vec4(U,yuv1.x, V, yuv2.x);
}
