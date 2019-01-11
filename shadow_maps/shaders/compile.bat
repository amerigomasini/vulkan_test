%VULKAN_SDK_DIR%\Bin\glslangValidator.exe -V -o phong_vert.spv shader_phong.vert
%VULKAN_SDK_DIR%\Bin\glslangValidator.exe -V -o phong_frag.spv shader_phong.frag
%VULKAN_SDK_DIR%\Bin\glslangValidator.exe -V -o shadow_map_vert.spv shadow_map.vert
pause