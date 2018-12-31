#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 1) uniform sampler2D texSampler;

layout(constant_id = 0) const bool USE_TEXTURES = true; 

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;

void main()
{
	if (USE_TEXTURES) {
		outColor = texture(texSampler, fragTexCoord);
	}
	else {
		outColor = vec4(1.0f, 0.0f, 0.0f, 1.0f);
	}

}