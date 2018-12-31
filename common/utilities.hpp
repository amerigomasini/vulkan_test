#pragma once
#include <vector>
#include <vector>
#include "vulkan/vulkan_core.h"
#include "glm/glm.hpp"

struct Vertex
{
	glm::vec3 pos;
	glm::vec3 color;
	glm::vec2 texCoord;
	glm::vec3 normal;

	static VkVertexInputBindingDescription getBindingDescription()
	{
		VkVertexInputBindingDescription bindingInputDescription = {};
		bindingInputDescription.binding = 0;									//index of this binding in array of bindings
		bindingInputDescription.stride = sizeof(Vertex);						//how many bytes to move between entries
		bindingInputDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;		//these are per-vertex attributes
		return bindingInputDescription;
	}

	//We have two attributes in Vertex, so we need two attribute descriptions
	static std::array<VkVertexInputAttributeDescription, 4> getAttributeDescriptions()
	{
		/*
		common formats:
		float:	VK_FORMAT_R32_SFLOAT
		double:	VK_FORMAT_R64_SFLOAT
		vec2:	VK_FORMAT_R32G32_SFLOAT
		ivec2:	VK_FORMAT_R32G32_SINT
		vec3:	VK_FORMAT_R32G32B32_SFLOAT
		vec4:	VK_FORMAT_R32G32B32A32_SFLOAT
		uvec4:	VK_FORMAT_R32G32B32A32_SUINT
		*/

		std::array<VkVertexInputAttributeDescription, 4> attributeDescriptions = {};

		attributeDescriptions[0].binding = 0;							//array of referenced binding in binding array
		attributeDescriptions[0].location = 0;							//location of attribute within selected binding
		attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;	//vec3 of floats
		attributeDescriptions[0].offset = offsetof(Vertex, pos);		//offset of pos into the Vertex structure

		attributeDescriptions[1].binding = 0;
		attributeDescriptions[1].location = 1;
		attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[1].offset = offsetof(Vertex, color);

		attributeDescriptions[2].binding = 0;
		attributeDescriptions[2].location = 2;
		attributeDescriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
		attributeDescriptions[2].offset = offsetof(Vertex, texCoord);

		attributeDescriptions[3].binding = 0;
		attributeDescriptions[3].location = 3;
		attributeDescriptions[3].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[3].offset = offsetof(Vertex, normal);

		return attributeDescriptions;
	}

	bool operator==(const Vertex& other) const {
		return pos == other.pos && color == other.color && texCoord == other.texCoord;
	}
};

namespace std
{
	template<> struct hash<Vertex>
	{
		size_t operator()(Vertex const& vertex) const
		{
			return ((hash<glm::vec3>()(vertex.pos) ^
				(hash<glm::vec3>()(vertex.color) << 1)) >> 1) ^
				(hash<glm::vec2>()(vertex.texCoord) << 1);
		}
	};
}

class Utilities
{
public:
	static std::vector<uint32_t> getCubeIndices()
	{
		std::vector<uint32_t> indices = {
			3, 1, 0, 3, 2, 1,
			7, 2, 3, 6, 2, 7,
			2, 6, 5, 5, 1, 2,
			5, 6, 7, 4, 5, 7,
			1, 5, 4, 1, 4, 0,
			4, 7, 3, 4, 3, 0,
		};

		return indices;
	}

	static std::vector<Vertex> getCubeVertices()
	{
		std::vector<Vertex> vertices = {
			//position             color               uv (not used)  normal
			{{-0.5f, -0.5f, 0.5f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}, {-0.5773f, -0.5773f, 0.5773f}},
			{{-0.5f, 0.5f, 0.5f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}, {-0.5773f, 0.5773f, 0.5773f}},
			{{0.5f, 0.5f, 0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}, {0.5773f, 0.5773f, 0.5773f}},
			{{0.5f, -0.5f, 0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}, {0.5773f, -0.5773f, 0.5773f}},
			{{-0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f}, {-0.5773f, -0.5773f, -0.5773f}},
			{{-0.5f, 0.5f, -0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f}, {-0.5773f, 0.5773f, -0.5773f}},
			{{0.5f, 0.5f, -0.5f}, {1.0f, 1.0f, 1.0f}, {0.0f, 0.0f}, {0.5773f, 0.5773f, -0.5773f}},
			{{0.5f, -0.5f, -0.5f}, {1.0f, 1.0f, 1.0f}, {0.0f, 0.0f}, {0.5773f, -0.5773f, -0.5773f}},
		};
		return vertices;
	}

	static std::vector<uint32_t> getSquareIndices()
	{
		std::vector<uint32_t> indices = {
			0, 1, 2, 2, 3, 0
		};

		return indices;
	}

	static std::vector<Vertex> getSquareVertices()
	{
		std::vector<Vertex> vertices = {
			//position				color				uv				normal
			{{-0.5f, -0.5f, 0.0f},	{1.0f, 0.0f, 0.0f}, {1.0f, 0.0f},	{0.0f, 0.0f, 1.0f}},
			{{0.5f, -0.5f, 0.0f},	{0.0f, 1.0f, 0.0f}, {0.0f, 0.0f},	{0.0f, 0.0f, 1.0f}},
			{{0.5f, 0.5f, 0.0f},	{0.0f, 0.0f, 1.0f}, {0.0f, 1.0f},	{0.0f, 0.0f, 1.0f}},
			{{-0.5f, 0.5f, 0.0f},	{1.0f, 1.0f, 1.0f}, {1.0f, 1.0f},	{0.0f, 0.0f, 1.0f}}
		};

		return vertices;
	}
};