#if GL_compatibility_profile
    varying out vec4 TEX0;
#else
    out vec4 TEX0;
#endif

void main() {
    gl_Position = gl_Vertex;
    gl_TexCoord[0] = gl_MultiTexCoord0;
    TEX0 = gl_TexCoord[0];
}
