uniform sampler2D image;
uniform float imageWidth;

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


void main()
{
	vec4 color;
        color.rgba  = texture2D(image, TEXCOORD.xy).grba; // store Y0UVY1 ro rgba
        if(TEXCOORD.x * imageWidth / 2.0 - floor(TEXCOORD.x * imageWidth / 2.0) > 0.5) // 'odd' pixel
                color.r = color.a; // use Y1 instead of Y0
	colorOut = color;
}

