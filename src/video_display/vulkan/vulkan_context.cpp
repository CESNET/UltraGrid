/**
 * @file   video_display/vulkan_context.cpp
 * @author Martin Bela      <492789@mail.muni.cz>
 */
/*
 * Copyright (c) 2021-2023 CESNET, z. s. p. o.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, is permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * 3. Neither the name of CESNET nor the names of its contributors may be
 *    used to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHORS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING,
 * BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
 * AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "vulkan_context.hpp"
#include <cassert>
#include <iostream>

using namespace vulkan_display_detail;
using namespace vulkan_display;

namespace {

VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(
        [[maybe_unused]] VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
        [[maybe_unused]] VkDebugUtilsMessageTypeFlagsEXT message_type,
        const VkDebugUtilsMessengerCallbackDataEXT* callback_data,
        [[maybe_unused]] void* user_data)
{
        LogLevel level = LogLevel::notice;
        if      (VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT & message_severity)   level = LogLevel::error;
        else if (VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT & message_severity) level = LogLevel::warning;
        else if (VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT & message_severity)    level = LogLevel::info;
        else if (VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT & message_severity) level = LogLevel::verbose;

        vulkan_log_msg(level, "validation layer: "s + callback_data->pMessage);

        if (message_type != VkDebugUtilsMessageTypeFlagBitsEXT::VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT){
                //assert(false);
        }

        return VK_FALSE;
}

void check_validation_layers(const std::vector<const char*>& required_layers) {
        std::vector<vk::LayerProperties>  layers = vk::enumerateInstanceLayerProperties();
        //for (auto& l : layers) puts(l.layerName);

        for (const auto& req_layer : required_layers) {
                auto layer_equals = [req_layer](auto const &layer) { return strcmp(req_layer, layer.layerName) == 0; };
                bool found = std::any_of(layers.begin(), layers.end(), layer_equals);
                if (!found) {
                        throw VulkanError{"Layer "s + req_layer + " is not supported."};
                }
        }
}

void check_instance_extensions(const std::vector<const char*>& required_extensions) {
        std::vector<vk::ExtensionProperties> extensions = vk::enumerateInstanceExtensionProperties(nullptr);

        for (const auto& req_exten : required_extensions) {
                auto extension_equals = [req_exten](auto const &exten) { return strcmp(req_exten, exten.extensionName) == 0; };
                bool found = std::any_of(extensions.begin(), extensions.end(), extension_equals);
                if (!found) {
                        throw VulkanError{"Instance extension "s + req_exten + " is not supported."};
                }
        }
}

bool check_device_extensions(bool propagate_error,
        const std::vector<const char*>& required_extensions, const vk::PhysicalDevice& device)
{
        std::vector<vk::ExtensionProperties> extensions = device.enumerateDeviceExtensionProperties(nullptr);

        for (const auto& req_exten : required_extensions) {
                auto extension_equals = [req_exten](auto const &exten) { return strcmp(req_exten, exten.extensionName) == 0; };
                bool found = std::any_of(extensions.begin(), extensions.end(), extension_equals);
                if (!found) {
                        if (propagate_error) {
                                throw VulkanError{"Device extension "s + req_exten + " is not supported."};
                        }
                        return false;
                }
        }
        return true;
}

uint32_t choose_queue_family_index(vk::PhysicalDevice gpu, vk::SurfaceKHR surface) {
        assert(gpu);

        std::vector<vk::QueueFamilyProperties> families = gpu.getQueueFamilyProperties();

        for (uint32_t i = 0; i < families.size(); i++) {
                VkBool32 surface_supported = true;
                if (surface) {
                        surface_supported = gpu.getSurfaceSupportKHR(i, surface);
                }

                if (surface_supported &&
                        (families[i].queueFlags & vk::QueueFlagBits::eGraphics) &&
                        (families[i].queueFlags & vk::QueueFlagBits::eCompute))
                {
                        return i;
                }
        }
        return no_queue_index_found;
}

const std::vector required_gpu_extensions = { "VK_KHR_swapchain" };

bool is_gpu_suitable(bool propagate_error, vk::PhysicalDevice gpu, vk::SurfaceKHR surface = nullptr) {
        bool result = check_device_extensions(propagate_error, required_gpu_extensions, gpu);
        if (!result) {
                return false;
        }
        uint32_t index = choose_queue_family_index(gpu, surface);
        return index != no_queue_index_found;
}

vk::PhysicalDevice choose_suitable_GPU(const std::vector<vk::PhysicalDevice>& gpus, vk::SurfaceKHR surface, uint32_t req_device_type) {
        assert(surface);
        for (const auto& gpu : gpus) {
                auto properties = gpu.getProperties();
                if (properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu) {
                        if (is_gpu_suitable(false, gpu, surface) && (req_device_type == vulkan_display::gpu_discrete
                                                || req_device_type == vulkan_display::no_gpu_selected)) {
                                return gpu;
                        }
                }
        }

        for (const auto& gpu : gpus) {
                auto properties = gpu.getProperties();
                if (properties.deviceType == vk::PhysicalDeviceType::eIntegratedGpu) {
                        if (is_gpu_suitable(false, gpu, surface) && (req_device_type == vulkan_display::gpu_integrated
                                                || req_device_type == vulkan_display::no_gpu_selected)) {
                                return gpu;
                        }
                }
        }

        for (const auto& gpu : gpus) {
                if (is_gpu_suitable(false, gpu, surface) && req_device_type == vulkan_display::no_gpu_selected) {
                        return gpu;
                }
        }

        throw VulkanError{"No suitable gpu found."};
}

vk::PhysicalDevice choose_gpu_by_index(const std::vector<vk::PhysicalDevice>& gpus, uint32_t gpu_index) {
        if (gpu_index >= gpus.size()) {
                throw VulkanError{"GPU index is not valid."};
        }
        std::vector<std::pair<std::string, vk::PhysicalDevice>> gpu_names;
        gpu_names.reserve(gpus.size());

        auto get_gpu_name = [](auto gpu) -> std::pair<std::string, vk::PhysicalDevice> {
                return { gpu.getProperties().deviceName, gpu };
        };

        std::transform(gpus.begin(), gpus.end(), std::back_inserter(gpu_names), get_gpu_name);

        std::sort(gpu_names.begin(), gpu_names.end());
        return gpu_names[gpu_index].second;
}

vk::CompositeAlphaFlagBitsKHR get_composite_alpha(vk::CompositeAlphaFlagsKHR capabilities) {
        uint32_t result = 1;
        while (!(result & static_cast<uint32_t>(capabilities))) {
                result <<= 1u;
        }
        return static_cast<vk::CompositeAlphaFlagBitsKHR>(result);
}

template<typename T>
bool contains(const std::vector<T>& vec, const T& key){
        return std::find(vec.begin(), vec.end(), key) != vec.end();
}

vk::SurfaceFormatKHR get_surface_format(vk::PhysicalDevice gpu, vk::SurfaceKHR surface) {
        std::vector<vk::SurfaceFormatKHR> available_formats = gpu.getSurfaceFormatsKHR(surface);

        std::array<vk::SurfaceFormatKHR, 5> preferred_formats {{
                {vk::Format::eA2B10G10R10UnormPack32, vk::ColorSpaceKHR::eSrgbNonlinear},
                {vk::Format::eB8G8R8A8Unorm, vk::ColorSpaceKHR::eSrgbNonlinear},
                {vk::Format::eR8G8B8A8Unorm, vk::ColorSpaceKHR::eSrgbNonlinear},
        }};

        for(auto& format: preferred_formats){
                if (contains(available_formats, format)) {
                        return format;
                }
        }

        return available_formats[0];
}

vk::PresentModeKHR get_present_mode(vk::PhysicalDevice gpu, vk::SurfaceKHR surface, vk::PresentModeKHR preferred_present_mode) {
        std::vector<vk::PresentModeKHR> modes = gpu.getSurfacePresentModesKHR(surface);

        vk::PresentModeKHR preferred = preferred_present_mode;
        if (contains(modes, preferred)) {
                return preferred;
        }
        
        // Mailbox is alternative to Immediate, Fifo to everything else
        auto alternative = (preferred == vk::PresentModeKHR::eImmediate 
                ? vk::PresentModeKHR::eMailbox
                : vk::PresentModeKHR::eFifo);

        if (contains(modes, alternative)) {
                return alternative;
        }

        return modes[0];
}

void log_gpu_info(vk::PhysicalDeviceProperties const &gpu_properties, uint32_t vulkan_version){
        vulkan_log_msg(LogLevel::info, "Vulkan uses GPU called: "s + &gpu_properties.deviceName[0]);
        std::string msg = concat(32, std::array{
                "Used Vulkan API: "s,
                std::to_string(VK_VERSION_MAJOR(vulkan_version)),
                "."s,
                std::to_string(VK_VERSION_MINOR(vulkan_version))
                });
        vulkan_log_msg(LogLevel::info, msg);
}

vk::PhysicalDevice create_physical_device(vk::Instance instance, vk::SurfaceKHR surface, uint32_t gpu_index) {
        assert(instance);
        assert(surface);
        std::vector<vk::PhysicalDevice> gpus = instance.enumeratePhysicalDevices();

        if (gpu_index >= vulkan_display::gpu_macro_min) {
                return choose_suitable_GPU(gpus, surface, gpu_index);
        } else {
                auto gpu = choose_gpu_by_index(gpus, gpu_index);
                is_gpu_suitable(true, gpu, surface);
                return gpu;
        }
}

void create_swapchain_views(vk::Device device, vk::SwapchainKHR swapchain, vk::Format format, std::vector<SwapchainImage>& swapchain_images) {
        std::vector<vk::Image> images = device.getSwapchainImagesKHR(swapchain);
        auto image_count = static_cast<uint32_t>(images.size());

        vk::ImageViewCreateInfo image_view_info = 
                default_image_view_create_info(format);

        swapchain_images.resize(image_count);
        for (uint32_t i = 0; i < image_count; i++) {
                SwapchainImage& swapchain_image = swapchain_images[i];
                swapchain_image.image = images[i];

                image_view_info.setImage(swapchain_image.image);
                swapchain_image.view = device.createImageView(image_view_info);
        }
}

} //namespace ------------------------------------------------------------------------


namespace vulkan_display {

void VulkanInstance::init(std::vector<const char*>& required_extensions, bool enable_validation, std::function<void(LogLevel, std::string_view sv)> logging_function) {
        vulkan_log_msg = std::move(logging_function);
        std::vector<const char*> validation_layers{};
        if (enable_validation) {
                validation_layers.push_back("VK_LAYER_KHRONOS_validation");
                check_validation_layers(validation_layers);

                const char* debug_extension = VK_EXT_DEBUG_UTILS_EXTENSION_NAME;
                required_extensions.push_back(debug_extension);
        }

        check_instance_extensions(required_extensions);

        vk::ApplicationInfo app_info{};
        app_info.setApiVersion(VK_API_VERSION_1_1);
        vulkan_version = VK_API_VERSION_1_1;

        vk::InstanceCreateInfo instance_info{};
        instance_info
                .setPApplicationInfo(&app_info)
                .setEnabledLayerCount(static_cast<uint32_t>(validation_layers.size()))
                .setPpEnabledLayerNames(validation_layers.data())
                .setEnabledExtensionCount(static_cast<uint32_t>(required_extensions.size()))
                .setPpEnabledExtensionNames(required_extensions.data());
        auto result = vk::createInstance(&instance_info, nullptr, &instance);
        
        switch (result) {
                case vk::Result::eSuccess:
                        break;
                case vk::Result::eErrorIncompatibleDriver:
                        app_info.apiVersion = VK_API_VERSION_1_0;
                        vulkan_version = VK_API_VERSION_1_0;
                        result = vk::createInstance(&instance_info, nullptr, &instance);
                        if(result != vk::Result::eSuccess){
                            throw VulkanError{"Vulkan instance cannot be created: "s + vk::to_string(result)};
                        }
                        break;
                default:
                        throw VulkanError{"Vulkan instance cannot be created: "s + vk::to_string(result)};
        }

        if (enable_validation) {
                dynamic_dispatcher = std::make_unique<vk::DispatchLoaderDynamic>(instance, vkGetInstanceProcAddr);
                init_validation_layers_error_messenger();
        }
}

void VulkanInstance::init_validation_layers_error_messenger() {
        vk::DebugUtilsMessengerCreateInfoEXT messenger_info{};
        using Severity = vk::DebugUtilsMessageSeverityFlagBitsEXT;
        using Type = vk::DebugUtilsMessageTypeFlagBitsEXT;
        messenger_info
                .setMessageSeverity(Severity::eError | Severity::eInfo | Severity::eWarning) // severity::eInfo |
                .setMessageType(Type::eGeneral | Type::ePerformance | Type::eValidation)
                .setPfnUserCallback(debug_callback)
                .setPUserData(nullptr);
        messenger = instance.createDebugUtilsMessengerEXT(messenger_info, nullptr, *dynamic_dispatcher);
}

void VulkanInstance::get_available_gpus(std::vector<std::pair<std::string, bool>>& gpus) {
        assert(instance);

        std::vector<vk::PhysicalDevice> physical_devices = instance.enumeratePhysicalDevices();
        gpus.clear();
        gpus.reserve(physical_devices.size());
        for (const auto& gpu : physical_devices) {
                auto properties = gpu.getProperties();
                gpus.emplace_back(properties.deviceName, true);
        }
        std::sort(gpus.begin(), gpus.end());
}

void VulkanInstance::destroy() {
        if (instance) {
                instance.destroy();
                if (messenger) {
                        instance.destroy(messenger, nullptr, *dynamic_dispatcher);
                }
                dynamic_dispatcher = nullptr;
                instance = nullptr;
        }
}


} // namespace vulkan_display ----------------------------------------------------------------------------


namespace vulkan_display_detail { //------------------------------------------------------------------------

void VulkanContext::create_logical_device() {
        assert(gpu);
        assert(queue_family_index != no_queue_index_found);

        constexpr std::array priorities = { 1.0f };
        vk::DeviceQueueCreateInfo queue_info{};
        queue_info
                .setQueueFamilyIndex(queue_family_index)
                .setPQueuePriorities(priorities.data())
                .setQueueCount(1);

        vk::DeviceCreateInfo device_info{};
        device_info
                .setPNext(nullptr)
                .setQueueCreateInfoCount(1)
                .setPQueueCreateInfos(&queue_info)
                .setEnabledExtensionCount(static_cast<uint32_t>(required_gpu_extensions.size()))
                .setPpEnabledExtensionNames(required_gpu_extensions.data());

        vk::PhysicalDeviceFeatures2 features2{};
        vk::PhysicalDeviceSamplerYcbcrConversionFeatures yCbCr_feature{};
        if (vulkan_version == VK_API_VERSION_1_1) {
                features2.setPNext(&yCbCr_feature);
                gpu.getFeatures2(&features2);
                if (yCbCr_feature.samplerYcbcrConversion) {
                        yCbCr_supported = true;
                        device_info.setPNext(&features2);
                        vulkan_log_msg(LogLevel::info, "yCbCr feature supported.");
                }
        }

        //reset features
        features2 = vk::PhysicalDeviceFeatures2{};
        features2.setPNext(&yCbCr_feature);

        device = gpu.createDevice(device_info);
}

void VulkanContext::create_swap_chain(vk::SwapchainKHR&& old_swapchain) {
        constexpr int initialization_attempts = 3;
        for (int attempt = 0; attempt < initialization_attempts; attempt++) {
                auto& capabilities = swapchain_atributes.capabilities;
                capabilities = gpu.getSurfaceCapabilitiesKHR(surface);

                swapchain_atributes.format = get_surface_format(gpu, surface);
                swapchain_atributes.mode = get_present_mode(gpu, surface, preferred_present_mode);

                vk::Extent2D swapchain_image_size;
                swapchain_image_size.width = std::clamp(window_parameters.width,
                        capabilities.minImageExtent.width,
                        capabilities.maxImageExtent.width);
                swapchain_image_size.height = std::clamp(window_parameters.height,
                        capabilities.minImageExtent.height,
                        capabilities.maxImageExtent.height);
                swapchain_atributes.image_size = swapchain_image_size;

                uint32_t image_count = std::max(uint32_t{2}, capabilities.minImageCount);
                if (capabilities.maxImageCount != 0) {
                        image_count = std::min(image_count, capabilities.maxImageCount);
                }

                auto msg = concat(64, std::array{
                        "Recreating swapchain, size: "s,
                        std::to_string(swapchain_image_size.width),
                        "x"s,
                        std::to_string(swapchain_image_size.height),
                        ", format: "s,
                        vk::to_string(swapchain_atributes.format.format)
                });
                vulkan_log_msg(LogLevel::info, msg);

                //assert(capabilities.supportedUsageFlags & vk::ImageUsageFlagBits::eTransferDst);
                vk::SwapchainCreateInfoKHR swapchain_info{};
                swapchain_info
                        .setSurface(surface)
                        .setImageFormat(swapchain_atributes.format.format)
                        .setImageColorSpace(swapchain_atributes.format.colorSpace)
                        .setPresentMode(swapchain_atributes.mode)
                        .setMinImageCount(image_count)
                        .setImageExtent(swapchain_image_size)
                        .setImageArrayLayers(1)
                        .setImageUsage(vk::ImageUsageFlagBits::eColorAttachment)
                        .setImageSharingMode(vk::SharingMode::eExclusive)
                        .setPreTransform(swapchain_atributes.capabilities.currentTransform)
                        .setCompositeAlpha(get_composite_alpha(swapchain_atributes.capabilities.supportedCompositeAlpha))
                        .setClipped(true)
                        .setOldSwapchain(old_swapchain);
                try{
                        swapchain = device.createSwapchainKHR(swapchain_info);
                        device.destroy(old_swapchain);
                        old_swapchain = nullptr;
                        return;
                }
                catch(std::exception& err){
                        vulkan_log_msg(LogLevel::info, "Recreation unsuccesful: "s + err.what());
                        device.destroy(old_swapchain);
                        old_swapchain = nullptr;
                        if(attempt + 1 == initialization_attempts){
                            throw err;
                        }
                }
        }
        assert(false);
}

void VulkanContext::init(vulkan_display::VulkanInstance&& instance, VkSurfaceKHR surface,
        WindowParameters parameters, uint32_t gpu_index, vk::PresentModeKHR preferredMode)
{
        assert(!this->instance);
        this->instance = instance.instance;
        this->dynamic_dispatcher = std::move(instance.dynamic_dispatcher);
        this->messenger = instance.messenger;
        this->vulkan_version = instance.vulkan_version;
        instance.instance = nullptr;
        instance.messenger = nullptr;

        this->surface = surface;
        this->preferred_present_mode = preferredMode;
        window_parameters = parameters;

        gpu = create_physical_device(this->instance, surface, gpu_index);

        queue_family_index = choose_queue_family_index(gpu, surface);
        assert(queue_family_index != no_queue_index_found);

        create_logical_device();
        auto properties = gpu.getProperties();
        if (properties.apiVersion < VK_API_VERSION_1_1) {
                vulkan_version = VK_API_VERSION_1_0;
        }

        log_gpu_info(properties, vulkan_version);
        
        queue = device.getQueue(queue_family_index, 0);
        create_swap_chain();
        create_swapchain_views(device, swapchain, swapchain_atributes.format.format, swapchain_images);
}

void VulkanContext::create_framebuffers(vk::RenderPass render_pass) {
        vk::FramebufferCreateInfo framebuffer_info;
        framebuffer_info
                .setRenderPass(render_pass)
                .setWidth(swapchain_atributes.image_size.width)
                .setHeight(swapchain_atributes.image_size.height)
                .setLayers(1);

        for(auto& swapchain_image : swapchain_images){
                framebuffer_info
                        .setAttachmentCount(1)
                        .setPAttachments(&swapchain_image.view);
                swapchain_image.framebuffer = device.createFramebuffer(framebuffer_info);
        }
}

void VulkanContext::recreate_swapchain(WindowParameters parameters, vk::RenderPass render_pass) {
        window_parameters = parameters;

        device.waitIdle();

        destroy_framebuffers();
        destroy_swapchain_views();
        vk::SwapchainKHR old_swap_chain = swapchain;
        create_swap_chain(std::move(old_swap_chain));
        create_swapchain_views(device, swapchain, swapchain_atributes.format.format, swapchain_images);
        create_framebuffers(render_pass);
}

uint32_t VulkanContext::acquire_next_swapchain_image(vk::Semaphore acquire_semaphore) const {
        constexpr uint64_t timeout = 1'000'000'000; // 1s = 1 000 000 000 nanoseconds
        uint32_t image_index;
        auto acquired = device.acquireNextImageKHR(swapchain, timeout, acquire_semaphore, nullptr, &image_index);
        switch (acquired) {
                case vk::Result::eSuccess:
                        break;
                case vk::Result::eSuboptimalKHR: [[fallthrough]];
                case vk::Result::eErrorOutOfDateKHR:
                        image_index = swapchain_image_out_of_date;
                        break;
                case vk::Result::eTimeout:
                        image_index = swapchain_image_timeout;
                        break;
                default:
                        throw VulkanError{"Next swapchain image cannot be acquired."s + vk::to_string(acquired)};
        }
        return image_index;
}

void VulkanContext::destroy() {
        if (device) {
                // static_cast to silence nodiscard warning
                device.waitIdle();
                destroy_framebuffers();
                destroy_swapchain_views();
                device.destroy(swapchain);
                device.destroy();
        }
        if (instance) {
                instance.destroy(surface);
                if (messenger) {
                        instance.destroy(messenger, nullptr, *dynamic_dispatcher);
                }
                instance.destroy();
        }
        dynamic_dispatcher = nullptr;
}


} //vulkan_display_detail

