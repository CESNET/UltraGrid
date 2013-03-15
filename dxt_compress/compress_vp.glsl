#if ! legacy
out vec4 TEX0;
in vec4 position;
#endif

void main() {
#if legacy
    gl_Position = gl_Vertex;
    gl_TexCoord[0] = gl_MultiTexCoord0;
#else
    gl_Position = position;
    TEX0 = position * vec4(0.5) + vec4(0.5);
#endif
}
