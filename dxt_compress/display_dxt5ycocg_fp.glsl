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

uniform sampler2D _image;
void main()
{
    vec4 _rgba;
    float _scale;
    float _Co;
    float _Cg;
    float _R;
    float _G;
    float _B;
    _rgba = texture2D(_image, TEXCOORD.xy);
    _scale = 1.00000000E+00/(3.18750000E+01*_rgba.z + 1.00000000E+00);
    _Co = (_rgba.x - 5.01960814E-01)*_scale;
    _Cg = (_rgba.y - 5.01960814E-01)*_scale;
    _R = (_rgba.w + _Co) - _Cg;
    _G = _rgba.w + _Cg;
    _B = (_rgba.w - _Co) - _Cg;
    _rgba = vec4(_R, _G, _B, 1.00000000E+00);
    colorOut = _rgba;
    return;
} // main end
