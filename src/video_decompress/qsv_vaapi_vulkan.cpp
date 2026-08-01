#include "video_decompress/qsv_vaapi_vulkan.h"

#include <unistd.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include <va/va.h>
#include <va/va_drmcommon.h>
#include <vulkan/vulkan.h>

#include "qsv_vaapi_spv.h"

namespace {

bool
vk_ok(VkResult result)
{
        return result == VK_SUCCESS;
}

struct converter {
        int width{};
        int height{};
        size_t output_size{};
        VADisplay va_display{};
        std::string error;

        VkInstance instance{};
        VkPhysicalDevice physical{};
        VkDevice device{};
        uint32_t queue_family{};
        VkQueue queue{};
        VkCommandPool command_pool{};
        VkCommandBuffer command{};

        VkBuffer output{};
        VkDeviceMemory output_memory{};
        void *output_mapping{};
        bool output_coherent{};

        VkDescriptorSetLayout descriptor_layout{};
        VkDescriptorPool descriptor_pool{};
        VkDescriptorSet descriptor_set{};
        VkPipelineLayout pipeline_layout{};
        VkPipeline pipeline{};

        VkImage input{};
        VkDeviceMemory input_memory{};
        VkImageView input_view{};

        bool fail(const char *message)
        {
                error = message;
                return false;
        }

        uint32_t memory_type(uint32_t bits, VkMemoryPropertyFlags required,
                             VkMemoryPropertyFlags preferred = 0)
        {
                VkPhysicalDeviceMemoryProperties props{};
                vkGetPhysicalDeviceMemoryProperties(physical, &props);
                for (unsigned pass = 0; pass < 2; ++pass) {
                        VkMemoryPropertyFlags flags =
                            pass == 0 ? required | preferred : required;
                        for (uint32_t i = 0; i < props.memoryTypeCount; ++i)
                                if ((bits & (1U << i)) != 0U &&
                                    (props.memoryTypes[i].propertyFlags & flags) ==
                                        flags)
                                        return i;
                }
                return UINT32_MAX;
        }

        void destroy_input()
        {
                if (input_view)
                        vkDestroyImageView(device, input_view, nullptr);
                if (input)
                        vkDestroyImage(device, input, nullptr);
                if (input_memory)
                        vkFreeMemory(device, input_memory, nullptr);
                input_view = {};
                input = {};
                input_memory = {};
        }

        ~converter()
        {
                if (device)
                        vkDeviceWaitIdle(device);
                destroy_input();
                if (pipeline)
                        vkDestroyPipeline(device, pipeline, nullptr);
                if (pipeline_layout)
                        vkDestroyPipelineLayout(device, pipeline_layout,
                                                nullptr);
                if (descriptor_pool)
                        vkDestroyDescriptorPool(device, descriptor_pool,
                                                nullptr);
                if (descriptor_layout)
                        vkDestroyDescriptorSetLayout(device, descriptor_layout,
                                                     nullptr);
                if (output_mapping)
                        vkUnmapMemory(device, output_memory);
                if (output)
                        vkDestroyBuffer(device, output, nullptr);
                if (output_memory)
                        vkFreeMemory(device, output_memory, nullptr);
                if (command_pool)
                        vkDestroyCommandPool(device, command_pool, nullptr);
                if (device)
                        vkDestroyDevice(device, nullptr);
                if (instance)
                        vkDestroyInstance(instance, nullptr);
        }

        bool initialize()
        {
                VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
                app.pApplicationName = "UltraGrid QSV R10k";
                app.apiVersion = VK_API_VERSION_1_2;
                VkInstanceCreateInfo instance_info{
                    VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
                instance_info.pApplicationInfo = &app;
                if (!vk_ok(vkCreateInstance(&instance_info, nullptr,
                                             &instance)))
                        return fail("Cannot create Vulkan instance");

                uint32_t count = 0;
                vkEnumeratePhysicalDevices(instance, &count, nullptr);
                std::vector<VkPhysicalDevice> devices(count);
                vkEnumeratePhysicalDevices(instance, &count, devices.data());
                for (auto candidate : devices) {
                        VkPhysicalDeviceProperties props{};
                        vkGetPhysicalDeviceProperties(candidate, &props);
                        if (props.vendorID == 0x8086) {
                                physical = candidate;
                                break;
                        }
                }
                if (!physical)
                        return fail("Intel Vulkan device not found");

                uint32_t family_count = 0;
                vkGetPhysicalDeviceQueueFamilyProperties(
                    physical, &family_count, nullptr);
                std::vector<VkQueueFamilyProperties> families(family_count);
                vkGetPhysicalDeviceQueueFamilyProperties(
                    physical, &family_count, families.data());
                bool found_family = false;
                for (uint32_t i = 0; i < family_count; ++i) {
                        if ((families[i].queueFlags &
                             VK_QUEUE_COMPUTE_BIT) != 0U) {
                                queue_family = i;
                                found_family = true;
                                break;
                        }
                }
                if (!found_family)
                        return fail("No Vulkan compute queue");

                float priority = 1.0F;
                VkDeviceQueueCreateInfo queue_info{
                    VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
                queue_info.queueFamilyIndex = queue_family;
                queue_info.queueCount = 1;
                queue_info.pQueuePriorities = &priority;
                const char *extensions[] = {
                    VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
                    VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME,
                    VK_EXT_IMAGE_DRM_FORMAT_MODIFIER_EXTENSION_NAME,
                    VK_EXT_QUEUE_FAMILY_FOREIGN_EXTENSION_NAME,
                };
                VkDeviceCreateInfo device_info{
                    VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
                device_info.queueCreateInfoCount = 1;
                device_info.pQueueCreateInfos = &queue_info;
                device_info.enabledExtensionCount =
                    sizeof extensions / sizeof extensions[0];
                device_info.ppEnabledExtensionNames = extensions;
                if (!vk_ok(vkCreateDevice(physical, &device_info, nullptr,
                                          &device)))
                        return fail("Cannot create Vulkan device");
                vkGetDeviceQueue(device, queue_family, 0, &queue);

                VkBufferCreateInfo buffer_info{
                    VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
                buffer_info.size = output_size;
                buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
                buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
                if (!vk_ok(vkCreateBuffer(device, &buffer_info, nullptr,
                                           &output)))
                        return fail("Cannot create R10k output buffer");
                VkMemoryRequirements requirements{};
                vkGetBufferMemoryRequirements(device, output, &requirements);
                uint32_t type = memory_type(
                    requirements.memoryTypeBits,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                        VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
                if (type == UINT32_MAX)
                        return fail("No host-visible Vulkan memory");
                VkPhysicalDeviceMemoryProperties memory_props{};
                vkGetPhysicalDeviceMemoryProperties(physical, &memory_props);
                output_coherent =
                    (memory_props.memoryTypes[type].propertyFlags &
                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0U;
                VkMemoryAllocateInfo allocation{
                    VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
                allocation.allocationSize = requirements.size;
                allocation.memoryTypeIndex = type;
                if (!vk_ok(vkAllocateMemory(device, &allocation, nullptr,
                                             &output_memory)) ||
                    !vk_ok(vkBindBufferMemory(device, output, output_memory,
                                              0)) ||
                    !vk_ok(vkMapMemory(device, output_memory, 0, output_size,
                                      0, &output_mapping)))
                        return fail("Cannot allocate mapped R10k output");

                VkDescriptorSetLayoutBinding bindings[2]{};
                bindings[0] = {0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1,
                               VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
                bindings[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                               VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
                VkDescriptorSetLayoutCreateInfo layout_info{
                    VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
                layout_info.bindingCount = 2;
                layout_info.pBindings = bindings;
                if (!vk_ok(vkCreateDescriptorSetLayout(
                        device, &layout_info, nullptr, &descriptor_layout)))
                        return fail("Cannot create Vulkan descriptor layout");
                VkDescriptorPoolSize pool_sizes[] = {
                    {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
                    {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1},
                };
                VkDescriptorPoolCreateInfo pool_info{
                    VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
                pool_info.maxSets = 1;
                pool_info.poolSizeCount = 2;
                pool_info.pPoolSizes = pool_sizes;
                if (!vk_ok(vkCreateDescriptorPool(device, &pool_info, nullptr,
                                                   &descriptor_pool)))
                        return fail("Cannot create Vulkan descriptor pool");
                VkDescriptorSetAllocateInfo set_info{
                    VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
                set_info.descriptorPool = descriptor_pool;
                set_info.descriptorSetCount = 1;
                set_info.pSetLayouts = &descriptor_layout;
                if (!vk_ok(vkAllocateDescriptorSets(device, &set_info,
                                                     &descriptor_set)))
                        return fail("Cannot allocate Vulkan descriptor set");

                VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                         sizeof(uint32_t)};
                VkPipelineLayoutCreateInfo pipeline_layout_info{
                    VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
                pipeline_layout_info.setLayoutCount = 1;
                pipeline_layout_info.pSetLayouts = &descriptor_layout;
                pipeline_layout_info.pushConstantRangeCount = 1;
                pipeline_layout_info.pPushConstantRanges = &push;
                if (!vk_ok(vkCreatePipelineLayout(
                        device, &pipeline_layout_info, nullptr,
                        &pipeline_layout)))
                        return fail("Cannot create Vulkan pipeline layout");
                VkShaderModuleCreateInfo shader_info{
                    VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
                shader_info.codeSize = qsv_vaapi_spv_len;
                shader_info.pCode =
                    reinterpret_cast<const uint32_t *>(qsv_vaapi_spv);
                VkShaderModule shader{};
                if (!vk_ok(vkCreateShaderModule(device, &shader_info, nullptr,
                                                 &shader)))
                        return fail("Cannot create Vulkan shader");
                VkComputePipelineCreateInfo pipeline_info{
                    VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
                pipeline_info.stage = {
                    VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
                pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
                pipeline_info.stage.module = shader;
                pipeline_info.stage.pName = "main";
                pipeline_info.layout = pipeline_layout;
                VkResult pipeline_result = vkCreateComputePipelines(
                    device, {}, 1, &pipeline_info, nullptr, &pipeline);
                vkDestroyShaderModule(device, shader, nullptr);
                if (!vk_ok(pipeline_result))
                        return fail("Cannot create Vulkan compute pipeline");

                VkCommandPoolCreateInfo pool{
                    VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
                pool.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
                pool.queueFamilyIndex = queue_family;
                if (!vk_ok(vkCreateCommandPool(device, &pool, nullptr,
                                                &command_pool)))
                        return fail("Cannot create Vulkan command pool");
                VkCommandBufferAllocateInfo command_info{
                    VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
                command_info.commandPool = command_pool;
                command_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
                command_info.commandBufferCount = 1;
                if (!vk_ok(vkAllocateCommandBuffers(device, &command_info,
                                                     &command)))
                        return fail("Cannot allocate Vulkan command buffer");
                return true;
        }

        bool import_surface(VASurfaceID surface)
        {
                destroy_input();
                if (vaSyncSurface(va_display, surface) != VA_STATUS_SUCCESS)
                        return fail("vaSyncSurface failed");
                VADRMPRIMESurfaceDescriptor desc{};
                VAStatus va_status = vaExportSurfaceHandle(
                    va_display, surface,
                    VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2,
                    VA_EXPORT_SURFACE_READ_ONLY |
                        VA_EXPORT_SURFACE_COMPOSED_LAYERS,
                    &desc);
                if (va_status != VA_STATUS_SUCCESS)
                        return fail("VAAPI cannot export decoded Y410 surface");
                auto close_fds = [&] {
                        for (uint32_t i = 0; i < desc.num_objects; ++i)
                                if (desc.objects[i].fd >= 0)
                                        close(desc.objects[i].fd);
                };
                if (desc.fourcc != VA_FOURCC_Y410 || desc.num_objects != 1 ||
                    desc.num_layers != 1 ||
                    desc.layers[0].num_planes != 1) {
                        close_fds();
                        return fail("Unsupported exported Y410 layout");
                }
                VkSubresourceLayout plane{};
                plane.offset = desc.layers[0].offset[0];
                plane.rowPitch = desc.layers[0].pitch[0];
                plane.size = desc.objects[0].size;
                VkImageDrmFormatModifierExplicitCreateInfoEXT modifier{
                    VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_EXPLICIT_CREATE_INFO_EXT};
                modifier.drmFormatModifier =
                    desc.objects[0].drm_format_modifier;
                modifier.drmFormatModifierPlaneCount = 1;
                modifier.pPlaneLayouts = &plane;
                VkExternalMemoryImageCreateInfo external{
                    VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO};
                external.pNext = &modifier;
                external.handleTypes =
                    VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
                VkImageCreateInfo image_info{
                    VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
                image_info.pNext = &external;
                image_info.imageType = VK_IMAGE_TYPE_2D;
                image_info.format = VK_FORMAT_A2B10G10R10_UNORM_PACK32;
                image_info.extent = {desc.width, desc.height, 1};
                image_info.mipLevels = 1;
                image_info.arrayLayers = 1;
                image_info.samples = VK_SAMPLE_COUNT_1_BIT;
                image_info.tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT;
                image_info.usage = VK_IMAGE_USAGE_STORAGE_BIT;
                image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
                image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                if (!vk_ok(vkCreateImage(device, &image_info, nullptr,
                                         &input))) {
                        close_fds();
                        return fail("Vulkan cannot create imported Y410 image");
                }
                VkMemoryRequirements requirements{};
                vkGetImageMemoryRequirements(device, input, &requirements);
                uint32_t type = memory_type(
                    requirements.memoryTypeBits,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
                if (type == UINT32_MAX) {
                        close_fds();
                        return fail("No Vulkan memory type for Y410 DMA-BUF");
                }
                VkMemoryDedicatedAllocateInfo dedicated{
                    VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
                dedicated.image = input;
                VkImportMemoryFdInfoKHR import{
                    VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR};
                import.pNext = &dedicated;
                import.handleType =
                    VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
                import.fd = desc.objects[0].fd;
                desc.objects[0].fd = -1;
                VkMemoryAllocateInfo allocation{
                    VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
                allocation.pNext = &import;
                allocation.allocationSize = requirements.size;
                allocation.memoryTypeIndex = type;
                if (!vk_ok(vkAllocateMemory(device, &allocation, nullptr,
                                             &input_memory)) ||
                    !vk_ok(vkBindImageMemory(device, input, input_memory, 0))) {
                        close_fds();
                        return fail("Vulkan cannot import Y410 DMA-BUF");
                }
                close_fds();
                VkImageViewCreateInfo view{
                    VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
                view.image = input;
                view.viewType = VK_IMAGE_VIEW_TYPE_2D;
                view.format = image_info.format;
                view.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                view.subresourceRange.levelCount = 1;
                view.subresourceRange.layerCount = 1;
                if (!vk_ok(vkCreateImageView(device, &view, nullptr,
                                              &input_view)))
                        return fail("Cannot create imported Y410 image view");
                VkDescriptorImageInfo image_descriptor{};
                image_descriptor.imageView = input_view;
                image_descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                VkDescriptorBufferInfo buffer_descriptor{
                    output, 0, output_size};
                VkWriteDescriptorSet writes[2]{};
                writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[0].dstSet = descriptor_set;
                writes[0].dstBinding = 0;
                writes[0].descriptorCount = 1;
                writes[0].descriptorType =
                    VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                writes[0].pImageInfo = &image_descriptor;
                writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[1].dstSet = descriptor_set;
                writes[1].dstBinding = 1;
                writes[1].descriptorCount = 1;
                writes[1].descriptorType =
                    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                writes[1].pBufferInfo = &buffer_descriptor;
                vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);
                return true;
        }

        bool convert(VASurfaceID surface, unsigned char *destination,
                     size_t destination_stride)
        {
                if (!import_surface(surface))
                        return false;
                vkResetCommandBuffer(command, 0);
                VkCommandBufferBeginInfo begin{
                    VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
                begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
                if (!vk_ok(vkBeginCommandBuffer(command, &begin)))
                        return fail("Cannot begin Vulkan conversion");
                VkImageMemoryBarrier acquire{
                    VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
                acquire.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
                acquire.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                acquire.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
                acquire.newLayout = VK_IMAGE_LAYOUT_GENERAL;
                acquire.srcQueueFamilyIndex = VK_QUEUE_FAMILY_FOREIGN_EXT;
                acquire.dstQueueFamilyIndex = queue_family;
                acquire.image = input;
                acquire.subresourceRange.aspectMask =
                    VK_IMAGE_ASPECT_COLOR_BIT;
                acquire.subresourceRange.levelCount = 1;
                acquire.subresourceRange.layerCount = 1;
                vkCmdPipelineBarrier(command,
                                     VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                                     0, nullptr, 0, nullptr, 1, &acquire);
                vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                                  pipeline);
                vkCmdBindDescriptorSets(
                    command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout,
                    0, 1, &descriptor_set, 0, nullptr);
                uint32_t parameter = static_cast<uint32_t>(width);
                vkCmdPushConstants(command, pipeline_layout,
                                   VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                   sizeof parameter, &parameter);
                vkCmdDispatch(command, (width + 15U) / 16U,
                              (height + 15U) / 16U, 1);
                VkMemoryBarrier output_barrier{
                    VK_STRUCTURE_TYPE_MEMORY_BARRIER};
                output_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                output_barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
                vkCmdPipelineBarrier(command,
                                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                     VK_PIPELINE_STAGE_HOST_BIT, 0, 1,
                                     &output_barrier, 0, nullptr, 0, nullptr);
                VkImageMemoryBarrier release = acquire;
                release.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
                release.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
                release.srcQueueFamilyIndex = queue_family;
                release.dstQueueFamilyIndex = VK_QUEUE_FAMILY_FOREIGN_EXT;
                vkCmdPipelineBarrier(command,
                                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                     VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0,
                                     nullptr, 0, nullptr, 1, &release);
                if (!vk_ok(vkEndCommandBuffer(command)))
                        return fail("Cannot end Vulkan conversion");
                VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
                submit.commandBufferCount = 1;
                submit.pCommandBuffers = &command;
                if (!vk_ok(vkQueueSubmit(queue, 1, &submit, {})) ||
                    !vk_ok(vkQueueWaitIdle(queue)))
                        return fail("Vulkan conversion submission failed");
                if (!output_coherent) {
                        VkMappedMemoryRange range{
                            VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
                        range.memory = output_memory;
                        range.size = VK_WHOLE_SIZE;
                        if (!vk_ok(vkInvalidateMappedMemoryRanges(
                                device, 1, &range)))
                                return fail(
                                    "Cannot invalidate Vulkan output memory");
                }
                const size_t row = static_cast<size_t>(width) * 4U;
                if (destination_stride == row) {
                        std::memcpy(destination, output_mapping, output_size);
                } else {
                        for (int y = 0; y < height; ++y)
                                std::memcpy(
                                    destination + y * destination_stride,
                                    static_cast<unsigned char *>(
                                        output_mapping) +
                                        y * row,
                                    row);
                }
                return true;
        }
};

} // namespace

extern "C" void *
qsv_vaapi_vulkan_create(void *va_display, int width, int height)
{
        auto *state = new converter;
        state->va_display = static_cast<VADisplay>(va_display);
        state->width = width;
        state->height = height;
        state->output_size = static_cast<size_t>(width) * 4U * height;
        if (!state->initialize()) {
                delete state;
                return nullptr;
        }
        return state;
}

extern "C" void
qsv_vaapi_vulkan_destroy(void *state)
{
        delete static_cast<converter *>(state);
}

extern "C" bool
qsv_vaapi_vulkan_convert(void *state, unsigned int va_surface,
                         unsigned char *destination,
                         size_t destination_stride)
{
        return static_cast<converter *>(state)->convert(
            static_cast<VASurfaceID>(va_surface), destination,
            destination_stride);
}

extern "C" const char *
qsv_vaapi_vulkan_error(void *state)
{
        return static_cast<converter *>(state)->error.c_str();
}
