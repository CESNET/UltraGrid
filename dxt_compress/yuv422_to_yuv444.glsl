uniform sampler2D image;
uniform float imageWidth;
varying out vec4 color;

void main()
{
        color.rgba  = texture2D(image, gl_TexCoord[0].xy).grba; // store Y0UVY1 ro rgba
        if(gl_TexCoord[0].x * imageWidth / 2.0 - floor(gl_TexCoord[0].x * imageWidth / 2.0) > 0.5) // 'odd' pixel
                color.r = color.a; // use Y1 instead of Y0
}

