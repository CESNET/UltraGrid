#extension GL_EXT_gpu_shader4 : enable
uniform sampler2D _image;
void main()
{
    vec4 _rgba;
    _rgba = texture2D(_image, gl_TexCoord[0].xy);
    gl_FragColor = _rgba;
    return;
} // main end
