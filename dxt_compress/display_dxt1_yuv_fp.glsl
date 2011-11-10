uniform sampler2D yuvtex;

void main(void) {
vec4 col = texture2D(yuvtex, gl_TexCoord[0].st);

float Y = col[0];
float U = col[1]-0.5;
float V = col[2]-0.5;
Y=1.1643*(Y-0.0625);

float G = Y-0.39173*U-0.81290*V;
float B = Y+2.017*U;
float R = Y+1.5958*V;

gl_FragColor=vec4(R,G,B,1.0);
}
