#include "r12l_vaapi_vulkan.hpp"

#include <unistd.h>

#include <cstring>
#include <vector>

#include <va/va.h>
#include <va/va_drmcommon.h>
#include <vulkan/vulkan.h>

#include "r12l_vaapi_spv.h"

namespace {

template<typename T>
bool vk_ok(T status)
{
        return status == VK_SUCCESS;
}

}

struct r12l_vaapi_vulkan::impl {
        int width{};
        int height{};
        std::size_t source_size{};
        VADisplay va_display{};
        VASurfaceID surface{VA_INVALID_SURFACE};
        std::string error;

        VkInstance instance{};
        VkPhysicalDevice physical{};
        VkDevice device{};
        uint32_t queue_family{};
        VkQueue queue{};
        VkCommandPool command_pool{};
        VkCommandBuffer command{};

        VkBuffer input{};
        VkDeviceMemory input_memory{};
        void *input_mapping{};
        bool input_coherent{};

        VkImage output{};
        VkDeviceMemory output_memory{};
        VkImageView output_view{};
        bool output_initialized{};

        VkDescriptorSetLayout descriptor_layout{};
        VkDescriptorPool descriptor_pool{};
        VkDescriptorSet descriptor_set{};
        VkPipelineLayout pipeline_layout{};
        VkPipeline pipeline{};

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
                for (uint32_t pass = 0; pass < 2; ++pass) {
                        VkMemoryPropertyFlags flags =
                            pass == 0 ? required | preferred : required;
                        for (uint32_t i = 0; i < props.memoryTypeCount; ++i)
                                if ((bits & (1U << i)) &&
                                    (props.memoryTypes[i].propertyFlags & flags) ==
                                        flags)
                                        return i;
                }
                return UINT32_MAX;
        }

        void destroy_output()
        {
                if (device) vkDeviceWaitIdle(device);
                if (output_view) vkDestroyImageView(device, output_view, nullptr);
                if (output) vkDestroyImage(device, output, nullptr);
                if (output_memory) vkFreeMemory(device, output_memory, nullptr);
                output_view = {};
                output = {};
                output_memory = {};
                output_initialized = false;
                surface = VA_INVALID_SURFACE;
        }

        ~impl()
        {
                destroy_output();
                if (pipeline) vkDestroyPipeline(device, pipeline, nullptr);
                if (pipeline_layout)
                        vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
                if (descriptor_pool)
                        vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
                if (descriptor_layout)
                        vkDestroyDescriptorSetLayout(device, descriptor_layout,
                                                     nullptr);
                if (input_mapping) vkUnmapMemory(device, input_memory);
                if (input) vkDestroyBuffer(device, input, nullptr);
                if (input_memory) vkFreeMemory(device, input_memory, nullptr);
                if (command_pool)
                        vkDestroyCommandPool(device, command_pool, nullptr);
                if (device) vkDestroyDevice(device, nullptr);
                if (instance) vkDestroyInstance(instance, nullptr);
        }

        bool import_surface(VASurfaceID new_surface)
        {
                destroy_output();
                if (vaSyncSurface(va_display, new_surface) != VA_STATUS_SUCCESS)
                        return fail("vaSyncSurface failed before DMABUF export");

                VADRMPRIMESurfaceDescriptor desc{};
                VAStatus va_status = vaExportSurfaceHandle(
                    va_display, new_surface,
                    VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2,
                    VA_EXPORT_SURFACE_READ_WRITE |
                        VA_EXPORT_SURFACE_COMPOSED_LAYERS,
                    &desc);
                if (va_status != VA_STATUS_SUCCESS)
                        return fail("VAAPI cannot export the encoder surface");
                auto close_fds = [&] {
                        for (uint32_t i = 0; i < desc.num_objects; ++i)
                                if (desc.objects[i].fd >= 0)
                                        close(desc.objects[i].fd);
                };
                if (desc.fourcc != VA_FOURCC_Y410 || desc.num_objects != 1 ||
                    desc.num_layers != 1 ||
                    desc.layers[0].num_planes != 1) {
                        close_fds();
                        return fail("VAAPI exported an unsupported Y410 layout");
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
                if (!vk_ok(vkCreateImage(device, &image_info, nullptr, &output))) {
                        close_fds();
                        return fail("Vulkan cannot create the imported Y410 image");
                }

                VkMemoryRequirements requirements{};
                vkGetImageMemoryRequirements(device, output, &requirements);
                uint32_t type =
                    memory_type(requirements.memoryTypeBits,
                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
                if (type == UINT32_MAX) {
                        close_fds();
                        return fail("No Vulkan memory type for the Y410 DMABUF");
                }
                VkMemoryDedicatedAllocateInfo dedicated{
                    VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
                dedicated.image = output;
                VkImportMemoryFdInfoKHR import{
                    VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR};
                import.pNext = &dedicated;
                import.handleType =
                    VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
                import.fd = desc.objects[0].fd;
                desc.objects[0].fd = -1; // ownership passes to Vulkan
                VkMemoryAllocateInfo allocation{
                    VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
                allocation.pNext = &import;
                allocation.allocationSize = requirements.size;
                allocation.memoryTypeIndex = type;
                if (!vk_ok(vkAllocateMemory(device, &allocation, nullptr,
                                             &output_memory)) ||
                    !vk_ok(vkBindImageMemory(device, output, output_memory, 0))) {
                        close_fds();
                        return fail("Vulkan cannot import the Y410 DMABUF");
                }
                close_fds();

                VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
                view.image = output;
                view.viewType = VK_IMAGE_VIEW_TYPE_2D;
                view.format = image_info.format;
                view.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                view.subresourceRange.levelCount = 1;
                view.subresourceRange.layerCount = 1;
                if (!vk_ok(vkCreateImageView(device, &view, nullptr,
                                              &output_view)))
                        return fail("Vulkan cannot create the Y410 image view");

                VkDescriptorBufferInfo buffer_info{input, 0, source_size};
                VkDescriptorImageInfo image_descriptor{};
                image_descriptor.imageView = output_view;
                image_descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                VkWriteDescriptorSet writes[2]{};
                writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[0].dstSet = descriptor_set;
                writes[0].dstBinding = 0;
                writes[0].descriptorCount = 1;
                writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                writes[0].pBufferInfo = &buffer_info;
                writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[1].dstSet = descriptor_set;
                writes[1].dstBinding = 1;
                writes[1].descriptorCount = 1;
                writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                writes[1].pImageInfo = &image_descriptor;
                vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);
                surface = new_surface;
                return true;
        }
};

r12l_vaapi_vulkan::r12l_vaapi_vulkan() : m(std::make_unique<impl>()) {}
r12l_vaapi_vulkan::~r12l_vaapi_vulkan() = default;

bool
r12l_vaapi_vulkan::init(int width, int height, void *va_display)
{
        m->width = width;
        m->height = height;
        m->source_size =
            static_cast<std::size_t>(width) * 9U / 2U * height;
        m->va_display = static_cast<VADisplay>(va_display);

        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.pApplicationName = "UltraGrid R12L";
        app.apiVersion = VK_API_VERSION_1_2;
        VkInstanceCreateInfo instance_info{
            VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        instance_info.pApplicationInfo = &app;
        if (!vk_ok(vkCreateInstance(&instance_info, nullptr, &m->instance)))
                return m->fail("Cannot create Vulkan instance");

        uint32_t count = 0;
        vkEnumeratePhysicalDevices(m->instance, &count, nullptr);
        std::vector<VkPhysicalDevice> devices(count);
        vkEnumeratePhysicalDevices(m->instance, &count, devices.data());
        for (auto device : devices) {
                VkPhysicalDeviceProperties props{};
                vkGetPhysicalDeviceProperties(device, &props);
                if (props.vendorID == 0x8086) {
                        m->physical = device;
                        break;
                }
        }
        if (!m->physical) return m->fail("Intel Vulkan device not found");

        uint32_t family_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(m->physical, &family_count,
                                                  nullptr);
        std::vector<VkQueueFamilyProperties> families(family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(m->physical, &family_count,
                                                  families.data());
        for (uint32_t i = 0; i < family_count; ++i)
                if (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                        m->queue_family = i;
                        break;
                }
        float priority = 1.0F;
        VkDeviceQueueCreateInfo queue_info{
            VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        queue_info.queueFamilyIndex = m->queue_family;
        queue_info.queueCount = 1;
        queue_info.pQueuePriorities = &priority;
        const char *extensions[] = {
            VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
            VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME,
            VK_EXT_IMAGE_DRM_FORMAT_MODIFIER_EXTENSION_NAME,
        };
        VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        device_info.queueCreateInfoCount = 1;
        device_info.pQueueCreateInfos = &queue_info;
        device_info.enabledExtensionCount = 3;
        device_info.ppEnabledExtensionNames = extensions;
        if (!vk_ok(vkCreateDevice(m->physical, &device_info, nullptr,
                                   &m->device)))
                return m->fail("Cannot create Vulkan device");
        vkGetDeviceQueue(m->device, m->queue_family, 0, &m->queue);

        VkBufferCreateInfo buffer_info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        buffer_info.size = m->source_size;
        buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (!vk_ok(vkCreateBuffer(m->device, &buffer_info, nullptr, &m->input)))
                return m->fail("Cannot create Vulkan input buffer");
        VkMemoryRequirements buffer_requirements{};
        vkGetBufferMemoryRequirements(m->device, m->input,
                                      &buffer_requirements);
        uint32_t input_type =
            m->memory_type(buffer_requirements.memoryTypeBits,
                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (input_type == UINT32_MAX)
                return m->fail("No host-visible Vulkan input memory");
        VkPhysicalDeviceMemoryProperties memory_props{};
        vkGetPhysicalDeviceMemoryProperties(m->physical, &memory_props);
        m->input_coherent =
            memory_props.memoryTypes[input_type].propertyFlags &
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        VkMemoryAllocateInfo buffer_allocation{
            VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        buffer_allocation.allocationSize = buffer_requirements.size;
        buffer_allocation.memoryTypeIndex = input_type;
        if (!vk_ok(vkAllocateMemory(m->device, &buffer_allocation, nullptr,
                                     &m->input_memory)) ||
            !vk_ok(vkBindBufferMemory(m->device, m->input, m->input_memory, 0)) ||
            !vk_ok(vkMapMemory(m->device, m->input_memory, 0, m->source_size, 0,
                               &m->input_mapping)))
                return m->fail("Cannot allocate Vulkan input memory");

        VkDescriptorSetLayoutBinding bindings[2]{};
        bindings[0] = {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                       VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        bindings[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1,
                       VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo layout_info{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        layout_info.bindingCount = 2;
        layout_info.pBindings = bindings;
        if (!vk_ok(vkCreateDescriptorSetLayout(
                m->device, &layout_info, nullptr, &m->descriptor_layout)))
                return m->fail("Cannot create Vulkan descriptor layout");
        VkDescriptorPoolSize pool_sizes[2] = {
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1},
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
        };
        VkDescriptorPoolCreateInfo pool_info{
            VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pool_info.maxSets = 1;
        pool_info.poolSizeCount = 2;
        pool_info.pPoolSizes = pool_sizes;
        if (!vk_ok(vkCreateDescriptorPool(m->device, &pool_info, nullptr,
                                           &m->descriptor_pool)))
                return m->fail("Cannot create Vulkan descriptor pool");
        VkDescriptorSetAllocateInfo set_info{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        set_info.descriptorPool = m->descriptor_pool;
        set_info.descriptorSetCount = 1;
        set_info.pSetLayouts = &m->descriptor_layout;
        if (!vk_ok(vkAllocateDescriptorSets(m->device, &set_info,
                                             &m->descriptor_set)))
                return m->fail("Cannot allocate Vulkan descriptor set");

        VkPushConstantRange push_range{VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                       sizeof(uint32_t) * 2};
        VkPipelineLayoutCreateInfo pipeline_layout_info{
            VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pipeline_layout_info.setLayoutCount = 1;
        pipeline_layout_info.pSetLayouts = &m->descriptor_layout;
        pipeline_layout_info.pushConstantRangeCount = 1;
        pipeline_layout_info.pPushConstantRanges = &push_range;
        if (!vk_ok(vkCreatePipelineLayout(m->device, &pipeline_layout_info,
                                           nullptr, &m->pipeline_layout)))
                return m->fail("Cannot create Vulkan pipeline layout");
        VkShaderModuleCreateInfo shader_info{
            VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        shader_info.codeSize = r12l_vaapi_spv_len;
        shader_info.pCode =
            reinterpret_cast<const uint32_t *>(r12l_vaapi_spv);
        VkShaderModule shader{};
        if (!vk_ok(vkCreateShaderModule(m->device, &shader_info, nullptr,
                                         &shader)))
                return m->fail("Cannot create Vulkan compute shader");
        VkComputePipelineCreateInfo pipeline_info{
            VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pipeline_info.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipeline_info.stage.module = shader;
        pipeline_info.stage.pName = "main";
        pipeline_info.layout = m->pipeline_layout;
        VkResult pipeline_status = vkCreateComputePipelines(
            m->device, {}, 1, &pipeline_info, nullptr, &m->pipeline);
        vkDestroyShaderModule(m->device, shader, nullptr);
        if (!vk_ok(pipeline_status))
                return m->fail("Cannot create Vulkan compute pipeline");

        VkCommandPoolCreateInfo command_pool_info{
            VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        command_pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        command_pool_info.queueFamilyIndex = m->queue_family;
        if (!vk_ok(vkCreateCommandPool(m->device, &command_pool_info, nullptr,
                                        &m->command_pool)))
                return m->fail("Cannot create Vulkan command pool");
        VkCommandBufferAllocateInfo command_info{
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        command_info.commandPool = m->command_pool;
        command_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        command_info.commandBufferCount = 1;
        if (!vk_ok(vkAllocateCommandBuffers(m->device, &command_info,
                                             &m->command)))
                return m->fail("Cannot allocate Vulkan command buffer");
        return true;
}

bool
r12l_vaapi_vulkan::convert(const unsigned char *source,
                           std::size_t source_stride,
                           unsigned int va_surface)
{
        std::size_t row = static_cast<std::size_t>(m->width) * 9U / 2U;
        if (source_stride == row) {
                std::memcpy(m->input_mapping, source, m->source_size);
        } else {
                for (int y = 0; y < m->height; ++y)
                        std::memcpy(
                            static_cast<unsigned char *>(m->input_mapping) +
                                y * row,
                            source + y * source_stride, row);
        }
        if (!m->input_coherent) {
                VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
                range.memory = m->input_memory;
                range.size = VK_WHOLE_SIZE;
                if (!vk_ok(vkFlushMappedMemoryRanges(m->device, 1, &range)))
                        return m->fail("Cannot flush Vulkan input memory");
        }
        if (m->surface != va_surface &&
            !m->import_surface(static_cast<VASurfaceID>(va_surface)))
                return false;

        vkResetCommandBuffer(m->command, 0);
        VkCommandBufferBeginInfo begin{
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (!vk_ok(vkBeginCommandBuffer(m->command, &begin)))
                return m->fail("Cannot begin Vulkan conversion");
        VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        barrier.srcAccessMask =
            m->output_initialized ? VK_ACCESS_MEMORY_READ_BIT : 0;
        barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.oldLayout = m->output_initialized ? VK_IMAGE_LAYOUT_GENERAL
                                                   : VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = m->output;
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.layerCount = 1;
        vkCmdPipelineBarrier(m->command, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0,
                             nullptr, 0, nullptr, 1, &barrier);
        vkCmdBindPipeline(m->command, VK_PIPELINE_BIND_POINT_COMPUTE,
                          m->pipeline);
        vkCmdBindDescriptorSets(m->command, VK_PIPELINE_BIND_POINT_COMPUTE,
                                m->pipeline_layout, 0, 1,
                                &m->descriptor_set, 0, nullptr);
        uint32_t parameters[] = {
            static_cast<uint32_t>(m->width),
            static_cast<uint32_t>(row),
        };
        vkCmdPushConstants(m->command, m->pipeline_layout,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof parameters, parameters);
        vkCmdDispatch(m->command, (m->width + 15U) / 16U,
                      (m->height + 15U) / 16U, 1);
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        vkCmdPipelineBarrier(m->command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0,
                             nullptr, 0, nullptr, 1, &barrier);
        if (!vk_ok(vkEndCommandBuffer(m->command)))
                return m->fail("Cannot end Vulkan conversion");
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &m->command;
        if (!vk_ok(vkQueueSubmit(m->queue, 1, &submit, {})) ||
            !vk_ok(vkQueueWaitIdle(m->queue)))
                return m->fail("Vulkan conversion submission failed");
        m->output_initialized = true;
        return true;
}

const std::string &
r12l_vaapi_vulkan::error() const
{
        return m->error;
}
