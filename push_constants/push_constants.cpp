#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#include <iostream>
#include <stdexcept>
#include <functional>
#include <cstdlib>
#include <optional>
#include <fstream>
#include <array>
#include <chrono>
#include <unordered_map>

#include <algorithm>
#include <sstream>
#include <set>

#include "utilities.hpp"

const int WIDTH = 800;
const int HEIGHT = 600;

const std::string MODEL_PATH = "";
const std::string TEXTURE_PATH = "";

const int MAX_FRAMES_IN_FLIGHT = 2;

const std::vector<const char *> validationLayers = {
	"VK_LAYER_LUNARG_standard_validation"
};

const std::vector<const char *> deviceExtensions = {
	VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

#ifdef NDEBUG
const bool enableValidationLayers = true;
#else
const bool enableValidationLayers = true;
#endif


struct UniformBufferObject
{
	glm::mat4 view;
	glm::mat4 proj;
	glm::vec4 lighPos = glm::vec4(0.0f, 2.0f, 1.0f, 0.0f);
};

struct PushContantsObject
{
	glm::mat4 model;
};

std::vector<Vertex> vertices = Utilities::getCubeVertices();;
std::vector<uint32_t> indices = Utilities::getCubeIndices();

//Used to receive messages from validation layers about errors and warnings
//To see how to configure validation layers, read <vulkan_dir>/Config/vk_layer_settings.txt
//That file can be edited and placed next to the executable for custom behavior
static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
	VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
	VkDebugUtilsMessageTypeFlagsEXT messageType,
	const VkDebugUtilsMessengerCallbackDataEXT * pCallbackData,
	void * pUserData)
{
	std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;
	return VK_FALSE;
}

//Functions that are part of Extensions (end in EXT) are not loaded automatically, and 
//need to be loaded manually instead
VkResult CreateDebugUtilsMessengerEXT(
	VkInstance instance,
	VkDebugUtilsMessengerCreateInfoEXT const * pCreateInfo,
	VkAllocationCallbacks const * pAllocator,
	VkDebugUtilsMessengerEXT * pCallback)
{
	auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
	if (func != nullptr)
		return func(instance, pCreateInfo, pAllocator, pCallback);
	else
		return VK_ERROR_EXTENSION_NOT_PRESENT;
}

//Should be either in a static class or outside of a class
void DestroyDebugUtilsMessengerEXT(
	VkInstance instance,
	VkDebugUtilsMessengerEXT callback,
	VkAllocationCallbacks const * pAllocator)
{
	auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
	if (func != nullptr)
		func(instance, callback, pAllocator);
}

static std::vector<char> readFile(std::string const & filename)
{
	std::ifstream file(filename, std::ios::ate | std::ios::binary);

	if (!file.is_open())
		throw std::runtime_error("failed to open file");

	size_t fileSize = (size_t)file.tellg();
	std::vector<char> buffer(fileSize);

	file.seekg(0);
	file.read(buffer.data(), fileSize);

	file.close();

	return buffer;
}

class HelloTriangleApplication
{
public:
	void run()
	{
		initWindow();
		initVulkan();
		mainLoop();
		cleanup();
	}

private:
	GLFWwindow * window;
	VkInstance instance;
	VkDebugUtilsMessengerEXT callback;
	VkSurfaceKHR surface;
	VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;		//implicitly destroyed with the instance
	VkDevice device;
	VkQueue graphicsQueue;									//implicitly destroyed with logical device
	VkQueue presentQueue;									//implicitly destroyed with logical device
	VkSwapchainKHR swapChain;
	std::vector<VkImage> swapChainImages;					//implicitly destroyed with the swap chain
	VkFormat swapChainImageFormat;
	VkExtent2D swapChainExtent;
	std::vector<VkImageView> swapChainImageViews;
	VkRenderPass renderPass;
	VkDescriptorSetLayout descriptorSetLayout;
	VkPipelineLayout pipelineLayout;
	VkPipeline graphicsPipeline;
	std::vector<VkFramebuffer> swapChainFramebuffers;
	VkCommandPool commandPool;
	VkCommandPool transientCommandPool;
	std::vector<VkCommandBuffer> commandBuffers;			//implicitly destroyed with the command pool

	std::vector<VkSemaphore> imageAvailableSemaphores;
	std::vector<VkSemaphore> renderFinishedSemaphores;
	std::vector<VkFence> inFlightFences;
	size_t currentFrame = 0;

	VkBuffer vertexBuffer;
	VkDeviceMemory vertexBufferMemory;
	VkBuffer indexBuffer;
	VkDeviceMemory indexBufferMemory;

	std::vector<VkBuffer> uniformBuffers;
	std::vector<VkDeviceMemory> uniformBuffersMemory;
	VkDescriptorPool descriptorPool;
	std::vector<VkDescriptorSet> descriptorSets;			//implicitly destroyed with the descriptor set

	VkImage textureImage;
	VkDeviceMemory textureImageMemory;
	VkImageView textureImageView;
	VkSampler textureSampler;

	VkImage depthImage;
	VkDeviceMemory depthImageMemory;
	VkImageView depthImageView;

	std::vector<VkShaderModule> shaderModules;

	//std::vector<Vertex> vertices;
	//std::vector<uint32_t> indices;

	bool framebufferResized = false;

	struct QueueFamilyIndices
	{
		//The queues might be the same
		std::optional<uint32_t> graphicsFamily;		//Queue capable of executing graphics command
		std::optional<uint32_t> presentFamily;		//Queue capable of presenting to our window surface

		bool isComplete()
		{
			return graphicsFamily.has_value() && presentFamily.has_value();
		}
	};

	struct SwapChainSupportDetails
	{
		VkSurfaceCapabilitiesKHR capabilities;
		std::vector<VkSurfaceFormatKHR> formats;
		std::vector<VkPresentModeKHR> presentModes;
	};

	//Instance
	void createInstance()
	{
		VkApplicationInfo appInfo = {};
		appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
		appInfo.pNext = nullptr;
		appInfo.pApplicationName = "Hello Triangle";
		appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
		appInfo.pEngineName = nullptr;
		appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
		appInfo.apiVersion = VK_API_VERSION_1_1;

		VkInstanceCreateInfo createInfo = {};
		createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
		createInfo.pApplicationInfo = &appInfo;
		createInfo.enabledLayerCount = 0;

		if (enableValidationLayers && !checkValidationLayerSupport())
			throw std::runtime_error("validation layers requested, but not available");
		if (enableValidationLayers)
		{
			createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
			createInfo.ppEnabledLayerNames = validationLayers.data();
		}

		std::vector<const char *>extensions;
		if (!checkInstanceExtensionSupport(extensions))
			throw std::runtime_error("required extension is missing");

		createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
		createInfo.ppEnabledExtensionNames = extensions.data();

		if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS)
			throw std::runtime_error("failed to create instance");
	}

	bool checkValidationLayerSupport()
	{
		uint32_t layerCount;
		vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

		std::vector<VkLayerProperties> availableLayers(layerCount);
		vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

		for (auto const & layerName : validationLayers)
		{
			bool layerFound = false;
			for (auto const & layerProperty : availableLayers)
			{
				if (strcmp(layerName, layerProperty.layerName) == 0)
				{
					layerFound = true;
					break;
				}
			}

			if (!layerFound)
				return false;
		}

		return true;
	}

	bool checkInstanceExtensionSupport(std::vector<const char *> & requiredExtensions)
	{
		//Check which extensions are available
		uint32_t extensionCount = 0;
		vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);

		std::vector<VkExtensionProperties> extensions(extensionCount);
		vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions.data());
		std::cout << "available extensions:" << std::endl;
		for (auto const & one_extension : extensions)
			std::cout << "\t" << one_extension.extensionName << std::endl;

		uint32_t glfwExtensionCount = 0;
		const char ** glfwExtensions;
		glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

		requiredExtensions.insert(requiredExtensions.end(), glfwExtensions, glfwExtensions + glfwExtensionCount);
		if (enableValidationLayers)
			requiredExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

		//Check that all necessary extensions are supported
		bool necessaryExtensionsSupported = true;
		for (uint32_t i = 0; i < glfwExtensionCount; ++i)
		{
			auto it = std::find_if(extensions.begin(), extensions.end(), [&](VkExtensionProperties & extension)
			{ return strcmp(extension.extensionName, glfwExtensions[i]); });

			if (it == extensions.end())
			{
				necessaryExtensionsSupported = false;
				break;
			}
		}

		if (necessaryExtensionsSupported)
			return true;

		return false;
	}


	//Debug Messenger
	void setupDebugCallback()
	{
		if (!enableValidationLayers)
			return;

		//NOTE: messages might be printed out multiple times, because multiple layers might check the same things
		VkDebugUtilsMessengerCreateInfoEXT createInfo = {};
		createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
		createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
			VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
		createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
			VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
		createInfo.pfnUserCallback = debugCallback;
		createInfo.pUserData = nullptr;

		if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &callback) != VK_SUCCESS)
			throw std::runtime_error("failed to setup debug callback");
	}


	//Window Surface
	void createSurface()
	{
		if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS)
			throw std::runtime_error("failed to create window surface");
	}


	//Physical Device
	QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device)
	{
		QueueFamilyIndices indices;

		uint32_t queueFamilyCount = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

		std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());


		//TODO: Try to find a queue that supports all the operations we need (graphics and presenting)
		//for better performance
		int i = 0;
		for (auto const & queueFamily : queueFamilies)
		{
			if (queueFamily.queueCount > 0 && queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT)
				indices.graphicsFamily = i;

			VkBool32 presentSupport = false;
			vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);
			if (queueFamily.queueCount > 0 && presentSupport)
				indices.presentFamily = i;

			if (indices.isComplete())
				break;

			i++;
		}

		return indices;
	}

	bool checkDeviceExtensionSupport(VkPhysicalDevice device)
	{
		uint32_t extensionCount = 0;
		vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

		std::vector<VkExtensionProperties> availableExtensions(extensionCount);
		vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

		bool necessaryExtensionsSupported = true;
		for (size_t i = 0; i < deviceExtensions.size(); ++i)
		{
			auto it = std::find_if(availableExtensions.begin(), availableExtensions.end(), [&](VkExtensionProperties & extension)
			{ return strcmp(extension.extensionName, deviceExtensions[i]); });

			if (it == availableExtensions.end())
			{
				necessaryExtensionsSupported = false;
				break;
			}
		}

		if (necessaryExtensionsSupported)
			return true;

		return false;
	}

	bool isDeviceSuitable(VkPhysicalDevice device)
	{
		//Device is deemed suitable if it contains
		//1. A queue capable of presenting to our window surface
		//2. A queue capable of graphics operations
		//3. An extension capable of creating a Swap Chain which supports at least one format and one presentation mode for our surface
		//4. Supports Anisotropic filtering

		QueueFamilyIndices indices = findQueueFamilies(device);

		bool extenionsSupported = checkDeviceExtensionSupport(device);

		bool swapChainAdequate = false;
		if (extenionsSupported)
		{
			SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
			swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
		}

		VkPhysicalDeviceFeatures supportedFeatures;
		vkGetPhysicalDeviceFeatures(device, &supportedFeatures);

		return indices.isComplete() && extenionsSupported && swapChainAdequate && supportedFeatures.samplerAnisotropy;

		//Example: How to get physical properties and features
		/*
		VkPhysicalDeviceProperties deviceProperties;
		vkGetPhysicalDeviceProperties(device, &deviceProperties);

		VkPhysicalDeviceFeatures deviceFeatures;
		vkGetPhysicalDeviceFeatures(device, &deviceFeatures);
		*/

		//Example: Only consider discrete GPUs which support geometry shaders as suitable
		//Each device could be given a score based on which features it supports, and then we can
		//choose the device with the highest score.
		/*
		return deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU &&
			deviceFeatures.geometryShader;
			*/
	}

	void pickPhysicalDevice()
	{
		uint32_t deviceCount = 0;
		vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

		if (deviceCount == 0)
			throw std::runtime_error("failed to find GPUs that support Vulkan");

		std::vector<VkPhysicalDevice> devices(deviceCount);
		vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

		for (auto const & device : devices)
		{
			if (isDeviceSuitable(device))
			{
				physicalDevice = device;
				break;
			}
		}

		if (physicalDevice == VK_NULL_HANDLE)
			throw std::runtime_error("failed to find a suitable GPU");
	}


	//Logical Device
	void createLogicalDevice()
	{
		QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

		std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
		std::set<uint32_t> uniqueQueueFamilies = { indices.graphicsFamily.value(), indices.presentFamily.value() };

		float queuePriority = 1.0f;
		for (uint32_t uniqueFamily : uniqueQueueFamilies)
		{
			VkDeviceQueueCreateInfo queueCreateInfo = {};
			queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
			queueCreateInfo.queueFamilyIndex = indices.graphicsFamily.value();
			queueCreateInfo.queueCount = 1;
			queueCreateInfo.pQueuePriorities = &queuePriority;
			queueCreateInfos.push_back(queueCreateInfo);
		}

		//Features which we will need. They have to be supported by the physical device
		VkPhysicalDeviceFeatures deviceFeatures = {};
		deviceFeatures.samplerAnisotropy = VK_TRUE;

		VkDeviceCreateInfo createInfo = {};
		createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
		createInfo.pQueueCreateInfos = queueCreateInfos.data();
		createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
		createInfo.pEnabledFeatures = &deviceFeatures;

		createInfo.enabledLayerCount = 0;
		if (enableValidationLayers)
		{
			createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
			createInfo.ppEnabledLayerNames = validationLayers.data();
		}

		createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
		createInfo.ppEnabledExtensionNames = deviceExtensions.data();

		if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS)
			throw std::runtime_error("failed to create logical device");

		vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
		vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);
	}


	//Swap Chain
	SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device)
	{
		SwapChainSupportDetails details;

		vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

		uint32_t formatCount;
		vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);
		if (formatCount > 0)
		{
			details.formats.resize(formatCount);
			vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
		}

		uint32_t presentModeCount;
		vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);
		if (presentModeCount > 0)
		{
			details.presentModes.resize(presentModeCount);
			vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
		}

		return details;
	}

	VkSurfaceFormatKHR chooseSwapSurfaceFormat(std::vector<VkSurfaceFormatKHR> const & availableFormats)
	{
		//if the surface has no preferred format, we choose the one we like the most
		if (availableFormats.size() == 1 && availableFormats[0].format == VK_FORMAT_UNDEFINED)
			return { VK_FORMAT_B8G8R8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR };

		//otherwise see if our preferred format is available
		for (auto const & availableFormat : availableFormats)
		{
			if (availableFormat.format == VK_FORMAT_B8G8R8_UNORM && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
				return availableFormat;
		}

		//otherwise just pick one
		return availableFormats[0];
	}

	VkPresentModeKHR chooseSwapPresentMode(std::vector<VkPresentModeKHR> const & availablePresentModes)
	{
		VkPresentModeKHR bestMode = VK_PRESENT_MODE_FIFO_KHR;

		//check if triple-buffering is available, and if so pick it
		for (auto const & availablePresentMode : availablePresentModes)
		{
			if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR)
				return availablePresentMode;
			else if (availablePresentMode == VK_PRESENT_MODE_IMMEDIATE_KHR)
				bestMode = availablePresentMode;
		}

		return bestMode;
	}

	VkExtent2D chooseSwapExtent(VkSurfaceCapabilitiesKHR const & capabilities)
	{
		//the resolution of the swap chain images

		//usually we just match the window size
		if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max())
			return capabilities.currentExtent;
		else
		{
			int width, height;
			glfwGetFramebufferSize(window, &width, &height);
			//if the value is set to std::numeric_limits<uint32_t>::max() it means that our resolution can be anywhere between 
			//minImageExtent and maxImageExtent, as long as it is within the window limits
			VkExtent2D actualExtent = { static_cast<uint32_t>(width), static_cast<uint32_t>(height) };

			return actualExtent;
		}
	}

	void createSwapChain()
	{
		SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);

		VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
		VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
		swapChainExtent = chooseSwapExtent(swapChainSupport.capabilities);

		swapChainImageFormat = surfaceFormat.format;

		//choose how many images to store in the swap chain. we add one to use triple buffering
		uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
		if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount)
			imageCount = swapChainSupport.capabilities.maxImageCount;

		VkSwapchainCreateInfoKHR createInfo = {};
		createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
		createInfo.surface = surface;
		createInfo.minImageCount = imageCount;
		createInfo.imageFormat = surfaceFormat.format;
		createInfo.imageColorSpace = surfaceFormat.colorSpace;
		createInfo.imageExtent = swapChainExtent;
		createInfo.presentMode = presentMode;
		createInfo.clipped = VK_TRUE;									//do not care about pixelx obscured by other windows
		createInfo.imageArrayLayers = 1;								//always 1 unless developing a stereo 3D app
		//render directly to images. Change if we need to render images to a separate image first to perform post-processing
		createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
		createInfo.oldSwapchain = VK_NULL_HANDLE;

		QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
		uint32_t queueFamilyIndices[] = { indices.graphicsFamily.value(), indices.presentFamily.value() };
		if (indices.graphicsFamily != indices.presentFamily)
		{
			createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
			createInfo.queueFamilyIndexCount = 2;
			createInfo.pQueueFamilyIndices = queueFamilyIndices;
		}
		else
		{
			createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;		//best performance
			createInfo.queueFamilyIndexCount = 1;
			createInfo.pQueueFamilyIndices = nullptr;
		}

		//change to an enum value to pre-transforms images in the swap chain (ex. 90 degree rotation)
		createInfo.preTransform = swapChainSupport.capabilities.currentTransform;

		//setting for blending with other windows in the window system. should almost always be ignored.
		createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;

		if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) != VK_SUCCESS)
			throw std::runtime_error("failed to create swap chain");

		vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
		swapChainImages.resize(imageCount);
		vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());
	}

	void cleanupSwapChain()
	{
		vkDestroyImageView(device, depthImageView, nullptr);
		vkDestroyImage(device, depthImage, nullptr);
		vkFreeMemory(device, depthImageMemory, nullptr);

		for (auto framebuffer : swapChainFramebuffers)
			vkDestroyFramebuffer(device, framebuffer, nullptr);

		vkFreeCommandBuffers(device, commandPool, static_cast<uint32_t>(commandBuffers.size()), &commandBuffers[0]);

		vkDestroyPipeline(device, graphicsPipeline, nullptr);
		vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
		vkDestroyRenderPass(device, renderPass, nullptr);

		for (auto imageView : swapChainImageViews)
			vkDestroyImageView(device, imageView, nullptr);

		vkDestroySwapchainKHR(device, swapChain, nullptr);
	}

	void recreateSwapChain()
	{
		int width = 0, height = 0;
		while (width == 0 || height == 0)
		{
			glfwGetFramebufferSize(window, &width, &height);
			glfwWaitEvents();
		}

		//we are about to modify resources, make sure they are not in use
		vkDeviceWaitIdle(device);

		cleanupSwapChain();

		createSwapChain();
		createImageViews();
		createRenderPass();
		createGraphicsPipeline();
		createDepthResources();
		createFramebuffers();
		createCommandBuffers();
	}


	//Image Views
	VkImageView createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags)
	{
		VkImageViewCreateInfo viewInfo = {};
		viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInfo.image = image;
		viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		viewInfo.format = format;
		viewInfo.subresourceRange.aspectMask = aspectFlags;
		viewInfo.subresourceRange.baseMipLevel = 0;
		viewInfo.subresourceRange.levelCount = 1;
		viewInfo.subresourceRange.baseArrayLayer = 0;
		viewInfo.subresourceRange.layerCount = 1;

		VkImageView imageView;
		if (vkCreateImageView(device, &viewInfo, nullptr, &imageView) != VK_SUCCESS) {
			throw std::runtime_error("failed to create texture image view!");
		}

		return imageView;
	}

	void createImageViews()
	{
		swapChainImageViews.resize(swapChainImages.size());

		for (size_t i = 0; i < swapChainImageViews.size(); ++i)
		{
			swapChainImageViews[i] = createImageView(swapChainImages[i], swapChainImageFormat, VK_IMAGE_ASPECT_COLOR_BIT);
		}
	}


	//Render Passes
	void createRenderPass()
	{
		VkAttachmentDescription colorAttachment = {};
		colorAttachment.format = swapChainImageFormat;			//should match swap chain settings
		colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;		//should match multi-sampling settings
		colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;	//what to do with color and depth data in the buffer before rendering
		colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;	//what to do with color and depth data in the buffer after rendering
		colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;	//stencil buffer will be undefined. OK because we are not using it
		colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;

		//images need to be transitioned to specific layouts, based on what we plan to do with them
		colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;		//do not care about layout before render begins
		colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;	//after render is done the layout should be one that allows the image to be presented

		VkAttachmentDescription depthAttachment = {};
		depthAttachment.format = findDepthFormat();
		depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
		depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;		//Do not need depth info after the drawing has finished
		depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;



		//each render pass is made up of sub-passes
		//for example, an initial render can be followed up by post-processing
		//each subpass references an attachment description
		VkAttachmentReference colorAttachmentRef = {};
		//index in the array of color attachments to reference
		colorAttachmentRef.attachment = 0;
		//layout to use for this subpass
		colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;


		VkAttachmentReference depthAttachmentRef = {};
		depthAttachmentRef.attachment = 1;
		depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;


		VkSubpassDescription subpass = {};
		subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;		//compute sub-passes might be supported in the future
		subpass.colorAttachmentCount = 1;
		//The index of attachments in this array is referenced in the fragment shader
		//layout(location = 0) out vec4 outColor
		subpass.pColorAttachments = &colorAttachmentRef;
		subpass.pDepthStencilAttachment = &depthAttachmentRef;

		std::array<VkAttachmentDescription, 2> attachments = { colorAttachment, depthAttachment };
		VkRenderPassCreateInfo renderPassInfo = {};
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		renderPassInfo.subpassCount = 1;
		renderPassInfo.pSubpasses = &subpass;
		renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
		renderPassInfo.pAttachments = attachments.data();

		//this dependencies assures that we do not try to access an image before it is available
		//identifies a dependency between the source and the destination subpass
		VkSubpassDependency dependency = {};

		//first synchronization scope includes commands executed before the current render pass
		dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
		//the synchronization scope is limited to this stage of the pipeline (after blending, before outputting to image)
		dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependency.srcAccessMask = 0;

		//second synchronization scope includes 0th subpass of the render pass
		dependency.dstSubpass = 0;
		//the synchronization scope is limited to this stage of the pipeline (after blending, before outputting to image)
		dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

		renderPassInfo.dependencyCount = 1;
		renderPassInfo.pDependencies = &dependency;

		if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS)
			throw std::runtime_error("failed to create render pass");

	}


	//Uniforms Descriptors
	void createDescriptorSetLayout()
	{
		VkDescriptorSetLayoutBinding uboLayoutBinding = {};
		uboLayoutBinding.binding = 0;											//binding used in the shader
		uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;	//type of descriptor
		uboLayoutBinding.descriptorCount = 1;
		uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;				//which shader stage will use it. different flags can be ORd together
		uboLayoutBinding.pImmutableSamplers = nullptr;							//only relevant for image sampling


		std::array<VkDescriptorSetLayoutBinding, 1> bindings = { uboLayoutBinding };
		VkDescriptorSetLayoutCreateInfo layoutInfo = {};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
		layoutInfo.pBindings = bindings.data();

		if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS)
			throw std::runtime_error("failed to create descriptor set layout");

	}

	VkPipelineShaderStageCreateInfo loadShader(std::string shader_file, VkShaderStageFlagBits stage)
	{
		VkPipelineShaderStageCreateInfo shaderStage = {};
		shaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		shaderStage.stage = stage;
		auto code = readFile(shader_file.c_str());
		shaderStage.module = createShaderModule(code);
		shaderStage.pName = "main"; // todo : make param
		assert(shaderStage.module != VK_NULL_HANDLE);
		shaderModules.push_back(shaderStage.module);
		return shaderStage;
	}

	//Graphics Pipeline
	void createGraphicsPipeline()
	{
		std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages;
		shaderStages[0] = loadShader("shaders/phong_vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
		shaderStages[1] = loadShader("shaders/phong_frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);



		//Vertex Input
		//Describes format of vertex data passed to Vertex Shader
		auto bindingDescription = Vertex::getBindingDescription();
		auto attributeDescription = Vertex::getAttributeDescriptions();
		VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
		vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		vertexInputInfo.vertexBindingDescriptionCount = 1;
		vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
		vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescription.size());	//Descriptors of data passed to Vertex Shader
		vertexInputInfo.pVertexAttributeDescriptions = attributeDescription.data();




		//Input Assembly
		//Describes what kind of geometry will be drawn, and if primitive restart is enabled
		//possible geometry values: point list, line list, line strip, triangle list, triangle strip
		//primitive restart is used to break up lines and triangles in the strip topologies
		VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
		inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		inputAssembly.primitiveRestartEnable = VK_FALSE;




		//Viewport & Scissors
		//region of the framebuffer that will be rendered to
		//scissors can be used to filter out fragments at the rasterizer stage based on a rectangle
		VkViewport viewport = {};
		viewport.x = 0.0f;
		viewport.y = 0.0f;
		viewport.width = (float)swapChainExtent.width;		//size of swapchain might be different than the size of the window
		viewport.height = (float)swapChainExtent.height;
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;

		VkRect2D scissor = {};
		scissor.offset = { 0, 0 };
		scissor.extent = swapChainExtent;

		VkPipelineViewportStateCreateInfo viewportState = {};
		viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportState.pViewports = &viewport;
		viewportState.viewportCount = 1;
		viewportState.pScissors = &scissor;
		viewportState.scissorCount = 1;




		//Rasterizer
		//rasterizes the output of the vertex shader, performs depth and culling checks
		//also applies scissors and can be configured for wireframe rendering.
		VkPipelineRasterizationStateCreateInfo rasterizer = {};
		rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterizer.depthClampEnable = VK_FALSE;			//if true, clamps fragments that are beyond the near/far plane instead of discarding them
		rasterizer.rasterizerDiscardEnable = VK_FALSE;	//if true, disables the all output to the framebuffer
		rasterizer.polygonMode = VK_POLYGON_MODE_FILL;	//other modes: line = polygon edges drawn as lines. point: polygon vertices drawn as points
		rasterizer.lineWidth = 1.0f;					//line width in fragments. Thicker lines require to enable wideLines GPU feature.
		rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;	//back face culling
		rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;	//face winding
		rasterizer.depthBiasEnable = VK_FALSE;			//depth biasing. similar to face displacement
		rasterizer.depthBiasConstantFactor = 0.0f;
		rasterizer.depthBiasClamp = 0.0f;
		rasterizer.depthBiasSlopeFactor = 0.0f;




		//Multisampling AA
		//less expensive than superscaling AA because it is only run if multiple polygons map to the same pixel
		//requires enabling a GPU feature to use
		VkPipelineMultisampleStateCreateInfo multisampling = {};
		multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		multisampling.sampleShadingEnable = VK_FALSE;
		multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
		multisampling.minSampleShading = 1.0;
		multisampling.pSampleMask = nullptr;
		multisampling.alphaToCoverageEnable = VK_FALSE;
		multisampling.alphaToOneEnable = VK_FALSE;




		//Depth / Stencil Buffer
		VkPipelineDepthStencilStateCreateInfo depthStencil = {};
		depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depthStencil.depthTestEnable = VK_TRUE;
		depthStencil.depthWriteEnable = VK_TRUE;	//make this false for transparent objects
		depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;

		depthStencil.depthBoundsTestEnable = VK_FALSE;	//optional depth test. only keeps fragment that fall in a specified range
		depthStencil.minDepthBounds = 0.0f;
		depthStencil.maxDepthBounds = 1.0f;

		depthStencil.stencilTestEnable = VK_FALSE;		//use stenciling ?
		depthStencil.front = {};						//stencil option
		depthStencil.back = {};							//stencil option


		//Color Blending
		//blends color of current pixel with another pixel which is already in the framebuffer at the same location
		//can be done by mixing the two colors or by combining them in a bitwise operation
		//VkPipelineColorBlendAttachmentState is a per-framebuffer configuration of blend states
		VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
		colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
		colorBlendAttachment.blendEnable = VK_FALSE;
		colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
		colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
		colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
		colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
		colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
		colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;

		/*
			if (blendEnable) {
				finalColor.rgb = (srcColorBlendFactor * newColor.rgb) <colorBlendOp> (dstColorBlendFactor * oldColor.rgb);
				finalColor.a = (srcAlphaBlendFactor * newColor.a) <alphaBlendOp> (dstAlphaBlendFactor * oldColor.a);
			} else {
				finalColor = newColor;
			}

			finalColor = finalColor & colorWriteMask;
		*/

		//VkPipelineColorBlendStateCreateInfo is for global color blending settings
		VkPipelineColorBlendStateCreateInfo colorBlending = {};
		colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		colorBlending.logicOpEnable = VK_FALSE;					//Set to true to use bit-wise operations for blending, rather than mixing
		colorBlending.logicOp = VK_LOGIC_OP_COPY;
		colorBlending.attachmentCount = 1;
		colorBlending.pAttachments = &colorBlendAttachment;
		colorBlending.blendConstants[0] = 0.0f;
		colorBlending.blendConstants[1] = 0.0f;
		colorBlending.blendConstants[2] = 0.0f;
		colorBlending.blendConstants[3] = 0.0f;




		//Pipeline Layout
		//Used to define shader uniforms
		//the space reserved for push constants is at least 128 bytes, anything more needs to be confirmed to fit
		//in the space returned by maxPushConstantSize.
		//in our case we are storing the model matrix: 4 floats: 4 * 4 = 16 bytes
		VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
		pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipelineLayoutInfo.setLayoutCount = 1;
		pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
		pipelineLayoutInfo.pushConstantRangeCount = 1;
		VkPushConstantRange pushConstantRange = {};
		pushConstantRange.size = sizeof(PushContantsObject);
		pushConstantRange.offset = 0;
		pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
		pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
		if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS)
			throw std::runtime_error("failed to create pipeline layout");


		//Dynamic states
		//these states can be changed through command buffers
		std::vector<VkDynamicState> dynamicStateEnables = {};

		VkPipelineDynamicStateCreateInfo dynamicStateInfo = {};
		dynamicStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
		dynamicStateInfo.dynamicStateCount = static_cast<uint32_t>(dynamicStateEnables.size());
		dynamicStateInfo.pDynamicStates = dynamicStateEnables.data();

		VkGraphicsPipelineCreateInfo pipelineInfo = {};
		pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipelineInfo.stageCount = 2;
		pipelineInfo.pStages = shaderStages.data();						//the shaders to be used
		pipelineInfo.pVertexInputState = &vertexInputInfo;
		pipelineInfo.pInputAssemblyState = &inputAssembly;
		pipelineInfo.pViewportState = &viewportState;
		pipelineInfo.pRasterizationState = &rasterizer;
		pipelineInfo.pMultisampleState = &multisampling;
		pipelineInfo.pDepthStencilState = &depthStencil;
		pipelineInfo.pColorBlendState = &colorBlending;
		pipelineInfo.pDynamicState = &dynamicStateInfo;
		pipelineInfo.layout = pipelineLayout;
		pipelineInfo.renderPass = renderPass;
		pipelineInfo.subpass = 0;
		pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;			//in case this is a derived pipeline
		pipelineInfo.basePipelineIndex = -1;

		if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS)
			throw std::runtime_error("failed to create pipeline");
	}

	VkShaderModule createShaderModule(std::vector<char> const & code)
	{
		VkShaderModuleCreateInfo createInfo = {};
		createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
		createInfo.pCode = reinterpret_cast<const uint32_t *>(code.data());
		createInfo.codeSize = code.size();

		VkShaderModule shaderModule;
		if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS)
			throw std::runtime_error("failed to create shader module");

		return shaderModule;
	}


	//Framebuffers
	void createFramebuffers()
	{
		swapChainFramebuffers.resize(swapChainImageViews.size());

		for (size_t i = 0; i < swapChainFramebuffers.size(); ++i)
		{
			std::array<VkImageView, 2> attachments =
			{
				swapChainImageViews[i],
				depthImageView
			};

			VkFramebufferCreateInfo framebufferInfo = {};
			framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
			framebufferInfo.height = swapChainExtent.height;
			framebufferInfo.width = swapChainExtent.width;
			framebufferInfo.renderPass = renderPass;
			framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
			framebufferInfo.pAttachments = attachments.data();
			framebufferInfo.layers = 1;					//number of layers in image arrays (how many images in swap chain images)
			if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS)
				throw std::runtime_error("failed to create frame buffer");
		}
	}

	//Model Loading (supported: OBJ)
	void loadModel()
	{
		tinyobj::attrib_t attrib;
		std::vector<tinyobj::shape_t> shapes;
		std::vector<tinyobj::material_t> materials;
		std::string warn, err;

		if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, MODEL_PATH.c_str()))
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
					uniqueVertices[vertex] = static_cast<uint32_t>(vertices.size());
					vertices.push_back(vertex);
				}

				indices.push_back(uniqueVertices[vertex]);
			}
		}
	}

	//Buffers
	uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties)
	{
		VkPhysicalDeviceMemoryProperties memProperties;
		vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

		//TODO: this should probably be revisited and done with one of the memory managers discussed at GDC
		//Specifically this code does not take into account which type of heap the memory comes from, but it should for better performance

		//Find a memory type which is suitable for the buffer and which supports specific properties, like being writable from the CPU
		//and being host-coherent (which I think means has a 1:1 relationship with CPU memory)
		for (uint32_t i = 0; i < memProperties.memoryTypeCount; ++i)
		{
			if ((typeFilter & (1 << i)) &&
				(memProperties.memoryTypes[i].propertyFlags & properties) == properties)
				return i;
		}

		throw std::runtime_error("failed to find suitable memory type");
	}

	void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer & buffer, VkDeviceMemory & bufferMemory)
	{
		//TODO
		//We should not allocate memory for each buffer individually, instead we should aggregate different buffers and use offsets
		//so that we do not flood the GPU with requests for allocation. The number of simultaneous memory allocations is limited by 
		//device, and even on brand new cards it might be as low as 4k (this can be checked from the maxMemoryAllocationCount physical device
		//flag).
		//We should look at the VulkanMemoryAllocator by GPUOpen library as a way to do this.

		//1. create the buffer
		VkBufferCreateInfo bufferInfo = {};
		bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		bufferInfo.size = size;										//size in bytes of this buffer
		bufferInfo.usage = usage;									//how will we use this generic buffer? multiple flags can be ORd together
		bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;			//is this buffer going to be owned by a single queue or will it be shared ?

		if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS)
			throw std::runtime_error("failed to create vertex buffer");

		//2. query its memory requirements
		VkMemoryRequirements memRequirements;
		vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

		//3. allocate memory
		VkMemoryAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize = memRequirements.size;
		allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);
		if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS)
			throw std::runtime_error("failed to allocate vertex buffer memory");

		//4. bind the memory to the buffer
		vkBindBufferMemory(device, buffer, bufferMemory, 0);
	}

	void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size)
	{
		VkCommandBuffer commandBuffer = beginSingleTimeCommands();

		VkBufferCopy copyRegion = {};
		copyRegion.srcOffset = 0;
		copyRegion.srcOffset = 0;
		copyRegion.size = size;
		vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

		endSingleTimeCommands(commandBuffer);
	}

	void createVertexBuffers()
	{
		VkDeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();

		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;
		createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

		// fill the buffer with data
		void * data;
		vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
		memcpy(data, vertices.data(), bufferSize);
		//If we were unable to secure host coherent memory, we need to call vkFlushMappedMemoryRanges to make sure the data is written
		//In that case we will also need vkInvalidateMappedMemoryRanges before reading from mapped memory
		vkUnmapMemory(device, stagingBufferMemory);


		//cannot map memory which is local to the device, instead we can copy buffers, if they were created with transfer flags
		createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertexBuffer, vertexBufferMemory);
		copyBuffer(stagingBuffer, vertexBuffer, bufferSize);

		vkDestroyBuffer(device, stagingBuffer, nullptr);
		vkFreeMemory(device, stagingBufferMemory, nullptr);
	}

	void createIndexBuffers()
	{
		//TODO 
		//it is recommended that the index buffer and the vertex buffer should be part of the SAME buffer
		//we would then use offsets to tell the driver where one ends and the other one starts
		//see last paragraph in https://vulkan-tutorial.com/Vertex_buffers/Index_buffer


		VkDeviceSize bufferSize = sizeof(indices[0]) * indices.size();

		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;
		createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

		void * data;
		vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
		memcpy(data, indices.data(), (size_t)bufferSize);
		vkUnmapMemory(device, stagingBufferMemory);

		createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indexBuffer, indexBufferMemory);
		copyBuffer(stagingBuffer, indexBuffer, bufferSize);

		vkDestroyBuffer(device, stagingBuffer, nullptr);
		vkFreeMemory(device, stagingBufferMemory, nullptr);
	}

	void createUniformBuffers()
	{
		VkDeviceSize bufferSize = sizeof(UniformBufferObject);

		uniformBuffers.resize(swapChainImages.size());
		uniformBuffersMemory.resize(swapChainImages.size());

		for (size_t i = 0; i < uniformBuffers.size(); ++i)
		{
			createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uniformBuffers[i], uniformBuffersMemory[i]);
		}
	}


	//Images and Textures
	void createImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling,
		VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage & image, VkDeviceMemory & imageMemory)
	{
		VkImageCreateInfo imageInfo = {};
		imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageInfo.imageType = VK_IMAGE_TYPE_2D;
		imageInfo.extent.width = static_cast<uint32_t>(width);
		imageInfo.extent.height = static_cast<uint32_t>(height);
		imageInfo.extent.depth = 1;
		imageInfo.mipLevels = 1;
		imageInfo.arrayLayers = 1;
		imageInfo.format = format;	//should be the same format as the one used for pixels in the buffer
		imageInfo.tiling = tiling;
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		imageInfo.usage = usage;
		imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;		//only relevant to images used as attachments
		imageInfo.flags = 0;

		if (vkCreateImage(device, &imageInfo, nullptr, &image) != VK_SUCCESS)
			throw std::runtime_error("failed to create texture image");

		VkMemoryRequirements memRequirements;
		vkGetImageMemoryRequirements(device, image, &memRequirements);

		VkMemoryAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize = memRequirements.size;
		allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

		if (vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS)
			throw std::runtime_error("failed to allocate memory for texture image");

		vkBindImageMemory(device, image, imageMemory, 0);
	}

	void createTextureImage()
	{
		//TODO
		//this function uses several command buffers to change images layouts and copy buffers to images.
		//all these command buffers should be condensed into a single one
		//if we could execute them asynchronously it would be even better

		int texWidth, texHeight, texChannels;
		stbi_uc * pixels = stbi_load(TEXTURE_PATH.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
		VkDeviceSize imageSize = texWidth * texHeight * 4;

		if (!pixels)
			throw std::runtime_error("failed to read texture image");

		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;
		createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

		void * data;
		vkMapMemory(device, stagingBufferMemory, 0, imageSize, 0, &data);
		memcpy(data, pixels, static_cast<size_t>(imageSize));
		vkUnmapMemory(device, stagingBufferMemory);
		stbi_image_free(pixels);

		createImage(texWidth, texHeight, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_TILING_OPTIMAL,
			VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			textureImage, textureImageMemory);

		transitionImageLayout(textureImage, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
		copyBufferToImage(stagingBuffer, textureImage, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));
		transitionImageLayout(textureImage, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

		vkDestroyBuffer(device, stagingBuffer, nullptr);
		vkFreeMemory(device, stagingBufferMemory, nullptr);
	}

	void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height)
	{
		VkCommandBuffer commandBuffer = beginSingleTimeCommands();

		VkBufferImageCopy region = {};
		region.bufferOffset = 0;
		region.bufferRowLength = 0;				//means pixels are tightly packed
		region.bufferImageHeight = 0;			//means pixels are tightly packed

		region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		region.imageSubresource.mipLevel = 0;
		region.imageSubresource.baseArrayLayer = 0;
		region.imageSubresource.layerCount = 1;

		region.imageOffset = { 0, 0, 0 };
		region.imageExtent = { width, height, 1 };

		vkCmdCopyBufferToImage(commandBuffer, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

		endSingleTimeCommands(commandBuffer);
	}

	void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout)
	{
		VkCommandBuffer commandBuffer = beginSingleTimeCommands();

		VkImageMemoryBarrier barrier = {};
		barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		barrier.oldLayout = oldLayout;
		barrier.newLayout = newLayout;
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;	//not using a barrier to transfer queue ownership, so these should be ignored
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.image = image;
		barrier.subresourceRange.baseMipLevel = 0;
		barrier.subresourceRange.levelCount = 1;
		barrier.subresourceRange.baseArrayLayer = 0;
		barrier.subresourceRange.layerCount = 1;

		VkPipelineStageFlags sourceStage;
		VkPipelineStageFlags destinationStage;

		if (newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
		{
			barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;

			if (hasStencilComponent(format))
				barrier.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
		}
		else
			barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;


		if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
		{
			barrier.srcAccessMask = 0;
			barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;	//block before we start writing to the image

			sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
			destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
		}
		else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
		{
			barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;	//block until we are done writing to the image
			barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;		//block before the shader starts reading the image

			sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
			destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
		}
		else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
		{
			barrier.srcAccessMask = 0;
			barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

			sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
			destinationStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
		}
		else
			throw std::invalid_argument("unsupported layout transition");

		vkCmdPipelineBarrier(commandBuffer,
			sourceStage, 		//at which point in the pipeline the operations that should happen before the sync take place
			destinationStage,	//at which point in the pipeline the operations that should happen after the sync take place
			0,
			0, nullptr,			//memory barriers
			0, nullptr,			//buffer memory barriers
			1, &barrier);		//image barriers

		endSingleTimeCommands(commandBuffer);
	}

	void createTextureImageView()
	{
		textureImageView = createImageView(textureImage, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT);
	}

	void createTextureSampler()
	{
		VkSamplerCreateInfo samplerInfo = {};
		samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
		samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;	//only relevant if address mode is border clamp

		samplerInfo.magFilter = VK_FILTER_LINEAR;			//how to interpolate texels that are magnified (oversampling -> too many pixels per texel)
		samplerInfo.minFilter = VK_FILTER_LINEAR;			//how to interpolate texels that are minified (undersampling -> too many texels per pixel)
		samplerInfo.anisotropyEnable = VK_TRUE;
		samplerInfo.maxAnisotropy = 16;						//beyond 16 the difference is negligible

		samplerInfo.unnormalizedCoordinates = VK_FALSE;		//normalize UVs to the 0-1 range

		samplerInfo.compareEnable = VK_FALSE;				//used to compare texels with a specific value
		samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;

		samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;	//mipmapping stuff
		samplerInfo.mipLodBias = 0.0f;
		samplerInfo.minLod = 0.0f;
		samplerInfo.maxLod = 0.0f;

		if (vkCreateSampler(device, &samplerInfo, nullptr, &textureSampler) != VK_SUCCESS)
			throw std::runtime_error("failed to create sampler");
	}

	//Command Pool
	void createCommandPool()
	{
		QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

		//Command buffers are executed by submitting them to queues
		//Each command pool can only target a single type of queue
		VkCommandPoolCreateInfo poolInfo = {};
		poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();

		if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS)
			throw std::runtime_error("failed to create command pool");

		//create command pool for short lived commands, like transfers
		poolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
		if (vkCreateCommandPool(device, &poolInfo, nullptr, &transientCommandPool) != VK_SUCCESS)
			throw std::runtime_error("failed to create transient command pool");

	}

	void createCommandBuffers()
	{
		commandBuffers.resize(swapChainFramebuffers.size());

		VkCommandBufferAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.commandPool = commandPool;
		allocInfo.commandBufferCount = static_cast<uint32_t>(commandBuffers.size());
		//primary: can be submitted to a queue for execution, cannot be called from other command buffers
		//secondary: cannot be submitted to a queue, can be called from a primary command buffer
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;

		if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS)
			throw std::runtime_error("failed to allocate command buffers");


		for (size_t i = 0; i < commandBuffers.size(); ++i)
		{
			VkCommandBufferBeginInfo beginInfo = {};
			beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
			beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;		//we will potentially use the command buffer while it is still executing previous commands
			beginInfo.pInheritanceInfo = nullptr;								//only relevant for secondary command buffers

			if (vkBeginCommandBuffer(commandBuffers[i], &beginInfo) != VK_SUCCESS)
				throw std::runtime_error("failed to begin recording command buffer");

			VkRenderPassBeginInfo renderPassInfo = {};
			renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
			renderPassInfo.renderPass = renderPass;
			renderPassInfo.framebuffer = swapChainFramebuffers[i];
			renderPassInfo.renderArea.offset = { 0, 0 };
			renderPassInfo.renderArea.extent = swapChainExtent;

			std::array<VkClearValue, 2> clearValues = {};
			clearValues[0].color = { 0.0f, 0.0f, 0.0f, 1.0f };
			clearValues[1].depthStencil = { 1.0f, 0 };			//clear the depth to 1 (the far plane)
			renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
			renderPassInfo.pClearValues = clearValues.data();

			//start a render pass
			vkCmdBeginRenderPass(commandBuffers[i], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

			//bind to the graphics pipeline
			vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

			//bind vertex buffers to the vertex pipeline
			VkBuffer vertexBuffers[] = { vertexBuffer };
			VkDeviceSize offsets[] = { 0 };
			vkCmdBindVertexBuffers(commandBuffers[i], 0, 1, vertexBuffers, offsets);

			vkCmdBindIndexBuffer(commandBuffers[i], indexBuffer, 0, VK_INDEX_TYPE_UINT32);

			vkCmdBindDescriptorSets(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSets[i], 0, nullptr);

			static auto startTime = std::chrono::high_resolution_clock::now();

			auto currentTime = std::chrono::high_resolution_clock::now();
			float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

			PushContantsObject pushConstant;
			pushConstant.model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
			vkCmdPushConstants(commandBuffers[i], pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(PushContantsObject), &pushConstant);

			//draw
			vkCmdDrawIndexed(commandBuffers[i], static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);

			//end render pass
			vkCmdEndRenderPass(commandBuffers[i]);

			//end the command
			if (vkEndCommandBuffer(commandBuffers[i]) != VK_SUCCESS)
				throw std::runtime_error("failed to record command buffer");

		}
	}

	VkCommandBuffer beginSingleTimeCommands()
	{
		VkCommandBufferAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.commandBufferCount = 1;
		allocInfo.commandPool = transientCommandPool;
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;

		VkCommandBuffer commandBuffer;
		if (vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer) != VK_SUCCESS)
			throw std::runtime_error("failed to create single use command buffer");

		VkCommandBufferBeginInfo beginInfo = {};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

		vkBeginCommandBuffer(commandBuffer, &beginInfo);
		return commandBuffer;
	}

	void endSingleTimeCommands(VkCommandBuffer commandBuffer)
	{
		vkEndCommandBuffer(commandBuffer);

		VkSubmitInfo submitInfo = {};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffer;

		vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
		vkQueueWaitIdle(graphicsQueue);

		vkFreeCommandBuffers(device, transientCommandPool, 1, &commandBuffer);
	}


	//Depth Resources
	VkFormat findSupportedFormat(std::vector<VkFormat> const & candidates, VkImageTiling tiling, VkFormatFeatureFlags features)
	{
		for (VkFormat format : candidates)
		{
			VkFormatProperties props;
			vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &props);
			if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features)
				return format;
			else if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features)
				return format;
		}

		throw std::runtime_error("failed to find supported format");
	}

	VkFormat findDepthFormat()
	{
		return findSupportedFormat(
			{ VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT },
			VK_IMAGE_TILING_OPTIMAL,
			VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
	}

	bool hasStencilComponent(VkFormat format)
	{
		return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
	}

	void createDepthResources()
	{
		VkFormat depthFormat = findDepthFormat();
		createImage(swapChainExtent.width, swapChainExtent.height, depthFormat, VK_IMAGE_TILING_OPTIMAL,
			VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, depthImage, depthImageMemory);
		depthImageView = createImageView(depthImage, depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT);

		transitionImageLayout(depthImage, depthFormat, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
	}


	//Synchronization
	void createSynchObjects()
	{
		VkSemaphoreCreateInfo semaphoreInfo = {};
		semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

		VkFenceCreateInfo fenceInfo = {};
		fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

		imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
		renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
		inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
		{
			if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS ||
				vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS ||
				vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS)
				throw std::runtime_error("failed to create semaphores or fences");
		}


	}


	//Descriptors
	void createDescriptorPool()
	{
		std::array<VkDescriptorPoolSize, 1> poolSizes = {};
		poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		poolSizes[0].descriptorCount = static_cast<uint32_t>(swapChainImages.size());

		VkDescriptorPoolCreateInfo poolInfo = {};
		poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
		poolInfo.pPoolSizes = poolSizes.data();
		poolInfo.maxSets = static_cast<uint32_t>(swapChainImages.size());

		if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS)
			throw std::runtime_error("failed to create descriptor pool");
	}

	void createDescriptorSets()
	{
		std::vector<VkDescriptorSetLayout> layouts(swapChainImages.size(), descriptorSetLayout);

		VkDescriptorSetAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		allocInfo.descriptorPool = descriptorPool;
		allocInfo.descriptorSetCount = static_cast<uint32_t>(swapChainImages.size());
		allocInfo.pSetLayouts = layouts.data();

		descriptorSets.resize(swapChainImages.size());
		if (vkAllocateDescriptorSets(device, &allocInfo, descriptorSets.data()) != VK_SUCCESS)
			throw std::runtime_error("failed to create descriptor sets");

		for (size_t i = 0; i < descriptorSets.size(); ++i)
		{
			VkDescriptorBufferInfo bufferInfo = {};
			bufferInfo.buffer = uniformBuffers[i];
			bufferInfo.offset = 0;
			bufferInfo.range = sizeof(UniformBufferObject);

			std::array<VkWriteDescriptorSet, 1> descriptorWrites = {};

			descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[0].descriptorCount = 1;
			descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
			descriptorWrites[0].dstBinding = 0;
			descriptorWrites[0].dstSet = descriptorSets[i];
			descriptorWrites[0].dstArrayElement = 0;
			descriptorWrites[0].pBufferInfo = &bufferInfo;	//only relevant for descriptors that refer to buffers
			descriptorWrites[0].pImageInfo = nullptr;		//only relevant for descriptors that refer to images
			descriptorWrites[0].pTexelBufferView = nullptr;	//only relevant for descriptors that refer to buffer views

			vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
		}
	}

	static void framebufferResizeCallback(GLFWwindow * window, int width, int height)
	{
		auto app = reinterpret_cast<HelloTriangleApplication *>(glfwGetWindowUserPointer(window));
		app->framebufferResized = true;
	}

	void initWindow()
	{
		glfwInit();

		//GLFW was made for OpenGL.
		//specify NOT to create an OGL context
		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

		window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
		glfwSetWindowUserPointer(window, this);
		glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
	}

	void initVulkan()
	{
		//1. create vulkan instance
		createInstance();

		//2. enable debug layers
		setupDebugCallback();

		//3. create a window surface
		createSurface();

		//4. choose which GPU to use
		pickPhysicalDevice();

		//5. create a logical device
		createLogicalDevice();

		//6. create a swap chain
		createSwapChain();

		//7. create image views for each image in the swap chain
		createImageViews();

		//8. create render passes
		createRenderPass();

		//9. create descriptors for uniforms
		createDescriptorSetLayout();

		//10. create the pipeline
		createGraphicsPipeline();

		//12.create commands pool
		createCommandPool();

		createDepthResources();

		createFramebuffers();

		//loadModel();

		//13.create vertex buffers
		createVertexBuffers();

		//14.create index buffers
		createIndexBuffers();

		//15.create uniform buffers
		createUniformBuffers();

		//16.create descriptor pools
		createDescriptorPool();

		//17.create descriptor set
		createDescriptorSets();

		//18.create command buffers
		createCommandBuffers();

		//19.create semaphores
		createSynchObjects();
	}


	void updateUniformBufferAndPushConstants(uint32_t currentImage)
	{
		static auto startTime = std::chrono::high_resolution_clock::now();

		auto currentTime = std::chrono::high_resolution_clock::now();
		float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

		UniformBufferObject ubo = {};
		ubo.view = glm::lookAt(glm::vec3(2, 2, 2), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
		ubo.proj = glm::perspective(glm::radians(45.0f), swapChainExtent.width / (float)swapChainExtent.height, 0.1f, 10.0f);

		//need to flip the Y axis, since the Y axis in Vulkan is inverted with respect to OpenGL, for which glm was designed
		ubo.proj[1][1] *= -1;

		void * data;
		vkMapMemory(device, uniformBuffersMemory[currentImage], 0, sizeof(ubo), 0, &data);
		memcpy(data, &ubo, sizeof(ubo));
		vkUnmapMemory(device, uniformBuffersMemory[currentImage]);
	}

	void drawFrame()
	{
		//1. get image from swap chain
		//2. execute command buffer for that image
		//3. return image to the swap chain, ready to be presented

		//operations take place asynchronously

		//CPU-GPU synchronization: the GPU signals the CPU when commands have finished executing for a particular frame
		vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, std::numeric_limits<uint64_t>::max());

		uint32_t imageIndex;
		VkResult result = vkAcquireNextImageKHR(device, swapChain, std::numeric_limits<uint64_t>::max(), imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);

		//cant check framebufferResized here, because I need the semaphore in AcquireNextImage to resolve before trying to continue
		if (result == VK_ERROR_OUT_OF_DATE_KHR)
		{
			recreateSwapChain();
			return;
		}
		else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
			throw std::runtime_error("failed to acquire swap chain image");

		createCommandBuffers();
		updateUniformBufferAndPushConstants(imageIndex);

		VkSubmitInfo submitInfo = {};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

		//wait for the image to become available before the the pipeline reaches a point where it needs to output to it
		//these are parallel arrays
		VkSemaphore waitSemaphore[] = { imageAvailableSemaphores[currentFrame] };
		VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };	//which pipeline stage should wait
		submitInfo.waitSemaphoreCount = 1;
		submitInfo.pWaitSemaphores = waitSemaphore;
		submitInfo.pWaitDstStageMask = waitStages;

		//the command buffers to execute with this submission
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffers[imageIndex];

		//which semaphores to signal once the command buffers have finished executing
		VkSemaphore signalSemaphore[] = { renderFinishedSemaphores[currentFrame] };
		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pSignalSemaphores = signalSemaphore;

		//Reset fences, they are now unlocked
		vkResetFences(device, 1, &inFlightFences[currentFrame]);

		if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS)
			throw std::runtime_error("failed to submit draw commands to queue");

		//wait for the queue command to complete before presenting it to the screen
		VkPresentInfoKHR presentInfo = {};
		presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
		presentInfo.waitSemaphoreCount = 1;
		presentInfo.pWaitSemaphores = signalSemaphore;

		VkSwapchainKHR swapChains[] = { swapChain };
		presentInfo.swapchainCount = 1;
		presentInfo.pSwapchains = swapChains;
		presentInfo.pImageIndices = &imageIndex;
		presentInfo.pResults = nullptr;		//only useful when using multiple swap chains

		result = vkQueuePresentKHR(presentQueue, &presentInfo);

		if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized)
		{
			framebufferResized = false;
			recreateSwapChain();
		}
		else if (result != VK_SUCCESS)
			throw std::runtime_error("failed to present swap chain image");

		currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
	}

	void mainLoop()
	{
		while (!glfwWindowShouldClose(window))
		{
			glfwPollEvents();
			drawFrame();
		}

		//wait for commands buffer to finish executing
		vkDeviceWaitIdle(device);
	}

	void cleanup()
	{
		cleanupSwapChain();

		vkDestroySampler(device, textureSampler, nullptr);
		vkDestroyImageView(device, textureImageView, nullptr);
		vkDestroyImage(device, textureImage, nullptr);
		vkFreeMemory(device, textureImageMemory, nullptr);

		vkDestroyDescriptorPool(device, descriptorPool, nullptr);
		vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);

		for (size_t i = 0; i < uniformBuffers.size(); ++i)
		{
			vkDestroyBuffer(device, uniformBuffers[i], nullptr);
			vkFreeMemory(device, uniformBuffersMemory[i], nullptr);
		}

		vkDestroyBuffer(device, vertexBuffer, nullptr);
		vkFreeMemory(device, vertexBufferMemory, nullptr);
		vkDestroyBuffer(device, indexBuffer, nullptr);
		vkFreeMemory(device, indexBufferMemory, nullptr);

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
		{
			vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
			vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
			vkDestroyFence(device, inFlightFences[i], nullptr);
		}

		for (auto shaderModule : shaderModules)
			vkDestroyShaderModule(device, shaderModule, nullptr);

		vkDestroyCommandPool(device, transientCommandPool, nullptr);
		vkDestroyCommandPool(device, commandPool, nullptr);

		vkDestroyDevice(device, nullptr);

		if (enableValidationLayers)
			DestroyDebugUtilsMessengerEXT(instance, callback, nullptr);

		vkDestroySurfaceKHR(instance, surface, nullptr);
		vkDestroyInstance(instance, nullptr);

		glfwDestroyWindow(window);
		glfwTerminate();
	}
};

int main()
{
	HelloTriangleApplication app;

	try
	{
		app.run();
	}
	catch (std::exception const & e)
	{
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}