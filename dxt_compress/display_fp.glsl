#if GL_legacy
#define TEXCOORD gl_TexCoord[0]
#else
#define TEXCOORD TEX0
#define texture2D texture
#endif

#if GL_legacy
#define colorOut gl_FragColor
#else
out vec4 colorOut;
#endif

#if ! GL_legacy
in vec4 TEX0;
#endif

uniform sampler2D _image;
void main()
{
    vec4 _rgba;
    _rgba = texture2D(_image, TEXCOORD.xy);
    colorOut = _rgba;
    return;
} // main end
