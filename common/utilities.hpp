#pragma once
#include <vector>
#include <vector>
#include "vulkan/vulkan_core.h"
#include "glm/glm.hpp"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#include "assimp/Importer.hpp"
#include "assimp/scene.h"
#include "assimp/postprocess.h"
#include "assimp/cimport.h"

struct Vertex
{
	glm::vec3 pos;
	glm::vec3 color;
	glm::vec2 texCoord;
	glm::vec3 normal;

	enum class VertexComponent
	{
		Position,
		Color,
		TextureUV,
		Normal,
	};

	static VkVertexInputBindingDescription getBindingDescription()
	{
		VkVertexInputBindingDescription bindingInputDescription = {};
		bindingInputDescription.binding = 0;									//index of this binding in array of bindings
		bindingInputDescription.stride = sizeof(Vertex);						//how many bytes to move between entries
		bindingInputDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;		//these are per-vertex attributes
		return bindingInputDescription;
	}

	//We have two attributes in Vertex, so we need two attribute descriptions
	static std::vector<VkVertexInputAttributeDescription> getAttributeDescriptions(std::vector<VertexComponent> const & requiredComponents)
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

		std::vector<VkVertexInputAttributeDescription> attributeDescriptions;
		attributeDescriptions.resize(requiredComponents.size());
		for (size_t i = 0; i < requiredComponents.size(); ++i)
		{
			attributeDescriptions[i].binding = 0;
			attributeDescriptions[i].location = static_cast<uint32_t>(i);
			
			switch (requiredComponents[i])
			{
			case VertexComponent::Position:
			{
				attributeDescriptions[i].format = VK_FORMAT_R32G32B32_SFLOAT;
				attributeDescriptions[i].offset = offsetof(Vertex, pos);
			} break;
			case VertexComponent::Color:
			{
				attributeDescriptions[i].format = VK_FORMAT_R32G32B32_SFLOAT;
				attributeDescriptions[i].offset = offsetof(Vertex, color);
			} break;
			case VertexComponent::TextureUV:
			{
				attributeDescriptions[i].format = VK_FORMAT_R32G32_SFLOAT;
				attributeDescriptions[i].offset = offsetof(Vertex, texCoord);
			} break;
			case VertexComponent::Normal:
			{
				attributeDescriptions[i].format = VK_FORMAT_R32G32B32_SFLOAT;
				attributeDescriptions[i].offset = offsetof(Vertex, normal);
			} break;
			}
		}

		return attributeDescriptions;
	}

	bool operator==(const Vertex& other) const {
		return pos == other.pos && color == other.color && texCoord == other.texCoord;
	}
};

struct Part
{
	size_t vertexBase;	//start of vertices in the vertex uber-buffer
	size_t indexBase;	//start of indices in the indices uber-buffer
	size_t vertexCount;	//number of vertices which make up this part
	size_t indexCount;	//number of indices which make up this part
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

	//Model Loading (supported: OBJ)
	static void loadOBJModel(std::string const & modelPath, std::vector<Vertex> & outVertices, std::vector<uint32_t> & outIndices)
	{
		tinyobj::attrib_t attrib;
		std::vector<tinyobj::shape_t> shapes;
		std::vector<tinyobj::material_t> materials;
		std::string warn, err;

		if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, modelPath.c_str()))
			throw std::runtime_error("failed to load model");

		std::unordered_map<Vertex, uint32_t> uniqueVertices = {};

		for (auto const & shape : shapes)
		{
			for (auto const & index : shape.mesh.indices)
			{
				Vertex vertex = {};

				vertex.pos =
				{
					attrib.vertices[3 * index.vertex_index + 0],
					attrib.vertices[3 * index.vertex_index + 1],
					attrib.vertices[3 * index.vertex_index + 2]
				};

				if (index.texcoord_index < 0)
				{
					vertex.texCoord = { 0, 0 };
				}
				else
				{
					vertex.texCoord =
					{
						attrib.texcoords[2 * index.texcoord_index + 0],
						1.0f - attrib.texcoords[2 * index.texcoord_index + 1]
					};
				}

				vertex.color = { 0.44f, 0.44f, 0.44f };

				if (uniqueVertices.count(vertex) == 0)
				{
					uniqueVertices[vertex] = static_cast<uint32_t>(outVertices.size());
					outVertices.push_back(vertex);
				}

				outIndices.push_back(uniqueVertices[vertex]);
			}
		}
	}

	static void loadModel(std::string const & modelPath, std::vector<Vertex> & outVertices, std::vector<uint32_t> & outIndices, std::vector<Part> & outParts)
	{
		Assimp::Importer importer;
		const aiScene * scene;

		static const int defaultFlags = aiProcess_FlipWindingOrder | aiProcess_Triangulate | aiProcess_PreTransformVertices | aiProcess_CalcTangentSpace | aiProcess_GenSmoothNormals;

		scene = importer.ReadFile(modelPath.c_str(), defaultFlags);
		if (scene == nullptr)
			throw std::runtime_error("failed to load model with assimp");

		size_t vertexCount = 0;
		size_t indexCount = 0;

		outParts.resize(scene->mNumMeshes);

		// Load meshes
		for (unsigned int i = 0; i < scene->mNumMeshes; i++)
		{
			const aiMesh* paiMesh = scene->mMeshes[i];

			outParts[i] = {};
			outParts[i].vertexBase = vertexCount;
			outParts[i].indexBase = indexCount;

			vertexCount += scene->mMeshes[i]->mNumVertices;

			aiColor3D pColor(0.f, 0.f, 0.f);
			scene->mMaterials[paiMesh->mMaterialIndex]->Get(AI_MATKEY_COLOR_DIFFUSE, pColor);

			const aiVector3D Zero3D(0.0f, 0.0f, 0.0f);

			for (unsigned int j = 0; j < paiMesh->mNumVertices; j++)
			{
				const aiVector3D* pPos = &(paiMesh->mVertices[j]);
				const aiVector3D* pNormal = &(paiMesh->mNormals[j]);
				const aiVector3D* pTexCoord = (paiMesh->HasTextureCoords(0)) ? &(paiMesh->mTextureCoords[0][j]) : &Zero3D;

// 				const aiVector3D* pTangent = (paiMesh->HasTangentsAndBitangents()) ? &(paiMesh->mTangents[j]) : &Zero3D;
// 				const aiVector3D* pBiTangent = (paiMesh->HasTangentsAndBitangents()) ? &(paiMesh->mBitangents[j]) : &Zero3D;

				Vertex oneVertex;
				oneVertex.pos = glm::vec3(pPos->x, -pPos->y, pPos->z);
				oneVertex.normal = glm::vec3(pNormal->x, -pNormal->y, pNormal->z);
				oneVertex.texCoord = glm::vec2(pTexCoord->x, pTexCoord->y);
				oneVertex.color = glm::vec3(pColor.r, pColor.g, pColor.b);
				outVertices.push_back(oneVertex);
			}

			outParts[i].vertexCount = paiMesh->mNumVertices;

			uint32_t indexBase = static_cast<uint32_t>(outIndices.size());
			for (unsigned int j = 0; j < paiMesh->mNumFaces; j++)
			{
				const aiFace& Face = paiMesh->mFaces[j];
				if (Face.mNumIndices != 3)
					continue;
				outIndices.push_back(indexBase + Face.mIndices[0]);
				outIndices.push_back(indexBase + Face.mIndices[1]);
				outIndices.push_back(indexBase + Face.mIndices[2]);
				outParts[i].indexCount += 3;
				indexCount += 3;
			}
		}
	}
};