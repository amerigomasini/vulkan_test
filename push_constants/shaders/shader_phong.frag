#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(constant_id = 0) const bool USE_TEXTURES = false;

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragTexCoord;
layout(location = 2) in vec3 fragViewVec;
layout(location = 3) in vec3 fragLightVec;
layout(location = 4) in vec3 fragNormal;

layout(location = 0) out vec4 outColor;

void main()
{
	if (USE_TEXTURES) {
		outColor = vec4(1, 0, 0, 1);
	}
	else {
		vec3 ambient = fragColor * vec3(0.3f);
		vec3 N = normalize(fragNormal);
		vec3 L = normalize(fragLightVec);
		vec3 V = normalize(fragViewVec);
		vec3 R = reflect(-L, N);

		vec3 diffuse = max(dot(N, L), 0.0f) * fragColor;
		vec3 specular = pow(max(dot(R, V), 0.0f), 32.0f) * vec3(0.35f);

		outColor = vec4(ambient + diffuse + specular, 1.0f);
	}

}