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

uniform sampler2D yuvtex;

void main(void) {
    vec4 col = texture2D(yuvtex, TEXCOORD.st);

    float Y = 1.1643 * (col[0] - 0.0625);
    float U = 1.1384 * (col[1] - 0.5);
    float V = 1.1384 * (col[2] - 0.5);

    float G = Y-0.39173*U-0.81290*V;
    float B = Y+2.017*U;
    float R = Y+1.5958*V;

    colorOut=vec4(R,G,B,1.0);
}
