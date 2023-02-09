/**
 * @file   video_display/vulkan_display.h
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

#pragma once

#include "vulkan_context.hpp"
#include "vulkan_transfer_image.hpp"
#include "vulkan_pipelines.hpp"
#include "concurrent_queue.hpp"

#include <array>
#include <deque>
#include <optional>
#include <queue>
#include <mutex>
#include <string>
#include <utility>

namespace vulkan_display_detail {

constexpr static unsigned filled_img_max_count = 1;
constexpr auto waiting_time_for_filled_image = 50ms;
constexpr size_t filled_image_queue_size = 1;

struct PerFrameResources{
        vk::CommandBuffer command_buffer;
        vk::Semaphore image_acquired_semaphore;
        vk::Semaphore image_rendered_semaphore;
        vk::DescriptorSet render_descriptor_set;

        Image2D converted_image;
        vk::DescriptorSet conversion_source_descriptor_set;
        vk::DescriptorSet conversion_destination_descriptor_set;
};


} // vulkan_display_detail

namespace vulkan_display {

namespace detail = vulkan_display_detail;

class WindowChangedCallback {
protected:
        ~WindowChangedCallback() = default;
public:
        virtual WindowParameters get_window_parameters() = 0;
};

class VulkanDisplay {
        std::string path_to_shaders;
        WindowChangedCallback* window = nullptr;
        detail::VulkanContext context;
        
        vk::Device device;
        std::mutex device_mutex{};

        bool format_conversion_enabled = false;
        detail::ConversionPipeline conversion_pipeline;

        detail::RenderPipeline render_pipeline;

        vk::DescriptorPool descriptor_pool;
        vk::CommandPool command_pool;

        std::array<detail::PerFrameResources, 3> frame_resources;
        std::vector<detail::PerFrameResources*> free_frame_resources;

        ImageDescription current_image_description;

        using TransferImageImpl = detail::TransferImageImpl;
        std::deque<TransferImageImpl> transfer_images{};

        /// available_img_queue - producer is the render thread, consumer is the provided thread
        detail::ConcurrentQueue<TransferImageImpl*> available_img_queue{};
        /// filled_img_queue - producer is the provider thread, consumer is the render thread
        detail::ConcurrentQueue<TransferImageImpl*, detail::filled_img_max_count> filled_img_queue{};
        /// local to provider thread
        std::vector<TransferImageImpl*> available_images;

        struct RenderedImage{
                TransferImageImpl* image;
                detail::PerFrameResources* gpu_commands;
        };
        std::queue<RenderedImage> rendered_images;

        bool destroyed = false;
private:
        void bind_transfer_image(TransferImageImpl& image, detail::PerFrameResources& resources);
        //void create_transfer_image(transfer_image*& result, image_description description);
        [[nodiscard]] TransferImageImpl& acquire_transfer_image();

        void record_graphics_commands(detail::PerFrameResources& commands, TransferImageImpl& transfer_image, uint32_t swapchain_image_id);

        void reconfigure(const TransferImageImpl& transfer_image);

        void destroy_format_dependent_resources();
public:
        /// TERMINOLOGY:
        /// render thread - thread which renders queued images on the screen 
        /// provider thread - thread which calls getf and putf and fills image queue with newly filled images

        VulkanDisplay() = default;

        VulkanDisplay(const VulkanDisplay& other) = delete;
        VulkanDisplay& operator=(const VulkanDisplay& other) = delete;
        VulkanDisplay(VulkanDisplay&& other) = delete;
        VulkanDisplay& operator=(VulkanDisplay&& other) = delete;

        ~VulkanDisplay() noexcept {
                if (!destroyed) {
                        try {
                                destroy();
                        } catch (vk::SystemError &e) {
                                std::string err = std::string("~VulkanDisplay vk::SystemError: ") + e.what() + "\n";
                                vulkan_display_detail::vulkan_log_msg(LogLevel::error, err);
                        }
                }
        }

        void init(VulkanInstance&& instance, VkSurfaceKHR surface, uint32_t transfer_image_count,
                WindowChangedCallback& window, uint32_t gpu_index = no_gpu_selected,
                std::string path_to_shaders = "./shaders", bool vsync = true, bool tearing_permitted = false);

        void destroy();

        /** Thread-safe */
        bool is_image_description_supported(ImageDescription description);

        /** Thread-safe to call from provider thread.*/
        TransferImage acquire_image(ImageDescription description);

        /** Thread-safe to call from provider thread.
         **
         ** @return true if image was discarded
         */
        bool queue_image(TransferImage img, bool discardable);

        /** Thread-safe to call from provider thread.*/
        void copy_and_queue_image(unsigned char* frame, ImageDescription description);

        /** Thread-safe to call from provider thread.*/
        void discard_image(TransferImage image) {
                auto* ptr = image.get_transfer_image();
                assert(ptr);
                available_images.push_back(ptr);
        }



        /** Thread-safe to call from render thread.
         **
         ** @return true if image was displayed
         */
        bool display_queued_image();

        /** Thread-safe*/
        uint32_t get_vulkan_version() const { return context.get_vulkan_version(); }
        
        /** Thread-safe*/
        bool is_yCbCr_supported() const { return context.is_yCbCr_supported(); }

        /**
         * @brief Hint to vulkan display that some window parameters spicified in struct WindowParameters changed.
         * Thread-safe.
         */
        void window_parameters_changed(WindowParameters new_parameters);

        
        /** Thread-safe */
        void window_parameters_changed() {
                window_parameters_changed(window->get_window_parameters());
        }
};

} //vulkan_display
