# vulkan_test
Simple Vulkan implementation &amp; tests

*<b>Building the projects:</b>*

Only the x64-bit projects have been tested.
You will need the following environment variables defined for the Visual Studio projects to build:
1. <b>VULKAN_SDK_DIR</b>, pointing to the directory where you installed Vulkan. For example, $(VULKAN_SDK_DIR)/include should be a valid path
2. <b>GLM_DIR</b>, pointing to the directory containing GLM. For example, $(GLM_DIR)/glm/glm.hpp should be a valid path.
3. <b>GLFW_DIR</b>, pointing to the directory containing GLFW. For example, $(GLFW_DIR)/include should be a valid path.

*<b>Downloading dependencies:</b>*
1. Vulkan SDK: https://www.lunarg.com/vulkan-sdk/
2. GLM: http://glm.g-truc.net/0.9.8/
3. GLFW: https://www.glfw.org/download.html
