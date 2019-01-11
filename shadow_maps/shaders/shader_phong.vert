#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0) uniform UniformBufferObject {
	mat4 model;
	mat4 view;
	mat4 proj;
	mat4 lightMVP;
	vec4 lightPos;
} ubo;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec2 inTexCoord;
layout(location = 3) in vec3 inNormal;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragtexCoord;
layout(location = 2) out vec3 fragViewVec;
layout(location = 3) out vec3 fragLightVec;
layout(location = 4) out vec3 fragNormal;
layout(location = 5) out vec4 outShadowCoord;

const mat4 biasMat = mat4( 
	0.5, 0.0, 0.0, 0.0,
	0.0, 0.5, 0.0, 0.0,
	0.0, 0.0, 1.0, 0.0,
	0.5, 0.5, 0.0, 1.0 );


void main() {
	gl_Position = ubo.proj * ubo.view * ubo.model * vec4(inPosition, 1.0);

	fragColor = inColor;
	fragtexCoord = inTexCoord;

	vec4 pos = ubo.view * ubo.model * vec4(inPosition, 1.0f);
	fragNormal = mat3(ubo.view * ubo.model) * inNormal;
	vec3 lPos = mat3(ubo.view * ubo.model) * ubo.lightPos.xyz;
	fragLightVec = lPos - pos.xyz;
	fragViewVec = -pos.xyz;

	outShadowCoord = ( biasMat * ubo.lightMVP * ubo.model ) * vec4(inPosition, 1.0);	
}