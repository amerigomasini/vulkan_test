#version 450

layout (binding = 0) uniform lightUniformObject
{
	mat4 mvp;
} ubo;

layout (location = 0) in vec3 lightPos;

void main()
{
	gl_Position = ubo.mvp * vec4(lightPos, 1.0);
}