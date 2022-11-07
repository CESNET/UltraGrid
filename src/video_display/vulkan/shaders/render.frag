#version 450

layout(push_constant) uniform constants
{
	uint x;
	uint y;
	uint width;
	uint height;
} render_area;

layout(binding = 0) uniform sampler2D texSampler;

layout(location = 0) out vec4 outColor;

void main() {
	float x = (gl_FragCoord.x - render_area.x) / render_area.width;
	float y = (gl_FragCoord.y - render_area.y) / render_area.height;
    outColor = texture(texSampler, vec2(x, y));
}
