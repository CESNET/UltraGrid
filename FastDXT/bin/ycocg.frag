uniform sampler2D yuvtex;

void main(void)
{
  vec4 col = texture2D(yuvtex, gl_TexCoord[0].st);

  float Co = col[0] - 0.5;
  float Cg = col[1] - 0.375;
  float  Y = col[3];

  float s = Y - (Cg / 2.0);
  float G = Cg + s;
  float B = s - (Co / 2.0);
  float R = B + Co;
  
  gl_FragColor=vec4(R,G,B,1.0);
}

