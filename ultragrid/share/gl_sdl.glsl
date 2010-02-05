uniform sampler2D Ytex;
uniform sampler2D Utex,Vtex;
void main(void) {
    float nx,ny,r,g,b,y,u,v;
    vec4 txl,ux,vx;
    nx=gl_TexCoord[0].x;
    ny=gl_TexCoord[0].y;
    y=texture2D(Ytex,vec2(nx,ny)).r;
    u=texture2D(Utex,vec2(nx,ny)).r;
    v=texture2D(Vtex,vec2(nx,ny)).r;

    y=1.1643*(y-0.0625);
    u=u-0.5;
    v=v-0.5;

    r=y+1.5958*v;
    g=y-0.39173*u-0.81290*v;
    b=y+2.017*u;

    gl_FragColor=vec4(r,g,b,1.0);
}
