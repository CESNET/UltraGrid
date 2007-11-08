uniform sampler2D Ytex, Utex, Vtex;
void main()
{ 
	float y,u,v; 
	y=texture2D(Ytex,vec2(gl_TexCoord[0])).x;
	u=texture2D(Utex,vec2(gl_TexCoord[1])).x;
	v=texture2D(Vtex,vec2(gl_TexCoord[2])).x;
	y=1.1643*(y-0.0625);
	u=u-0.5;
	v=v-0.5;
	
	gl_FragColor=vec4(y+1.5958*v,y-0.39173*u-0.81290*v,y+2.017*u,1);
}
