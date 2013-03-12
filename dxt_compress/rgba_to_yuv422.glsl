#if legacy
#define TEXCOORD gl_TexCoord[0]
#else
#define TEXCOORD TEX0
#define texture2D texture
#endif

#if legacy
#define colorOut gl_FragColor
#else
out vec4 colorOut;
#endif

#if ! legacy
in vec4 TEX0;
#endif

uniform sampler2D image;
uniform float imageWidth; // is original image width, it means twice as wide as ours

void main()
{
        vec4 rgba1, rgba2;
        vec4 yuv1, yuv2;
        vec2 coor1, coor2;
        float U, V;

        coor1 = TEXCOORD.xy - vec2(1.0 / (imageWidth * 2.0), 0.0);
        coor2 = TEXCOORD.xy + vec2(1.0 / (imageWidth * 2.0), 0.0);

        rgba1  = texture2D(image, coor1);
        rgba2  = texture2D(image, coor2);
        
        yuv1.x = 1.0/16.0 + (rgba1.r * 0.2126 + rgba1.g * 0.7152 + rgba1.b * 0.0722) * 0.8588; // Y
        yuv1.y = 0.5 + (-rgba1.r * 0.1145 - rgba1.g * 0.3854 + rgba1.b * 0.5) * 0.8784;
        yuv1.z = 0.5 + (rgba1.r * 0.5 - rgba1.g * 0.4541 - rgba1.b * 0.0458) * 0.8784;
        
        yuv2.x = 1.0/16.0 + (rgba2.r * 0.2126 + rgba2.g * 0.7152 + rgba2.b * 0.0722) * 0.8588; // Y
        yuv2.y = 0.5 + (-rgba2.r * 0.1145 - rgba2.g * 0.3854 + rgba2.b * 0.5) * 0.8784;
        yuv2.z = 0.5 + (rgba2.r * 0.5 - rgba2.g * 0.4541 - rgba2.b * 0.0458) * 0.8784;
        
        U = mix(yuv1.y, yuv2.y, 0.5);
        V = mix(yuv1.z, yuv2.z, 0.5);
        
        colorOut = vec4(U,yuv1.x, V, yuv2.x);
}
