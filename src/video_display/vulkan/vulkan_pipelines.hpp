/**
 * @file   video_display/vulkan_pipelines.hpp
 * @author Martin Bela      <492789@mail.muni.cz>
 */
/*
 * Copyright (c) 2021-2022 CESNET, z. s. p. o.
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

#include <string>

namespace vulkan_display_detail{

struct ImageSize{
        uint32_t width;
        uint32_t height;

        static ImageSize fromExtent2D(vk::Extent2D extent){
               return ImageSize{extent.width, extent.height};
        }
};

class ConversionPipeline {
        bool valid = false;

        vk::ShaderModule compute_shader{};
        vk::PipelineLayout pipeline_layout{};
        vk::Pipeline pipeline{};

        vk::DescriptorSetLayout source_desc_set_layout{};
        vk::DescriptorSetLayout destination_desc_set_layout{};

        vk::SamplerYcbcrConversion yCbCr_conversion{};
        vk::Sampler sampler{};

public:
        void create(vk::Device device, const std::string& shader_path, vk::Format format);

        void destroy(vk::Device device);

        void record_commands(vk::CommandBuffer cmd_buffer, ImageSize image_size, std::array<vk::DescriptorSet, 2> descriptor_set);

        vk::DescriptorSetLayout get_source_image_desc_set_layout(){ return source_desc_set_layout; }

        vk::DescriptorSetLayout get_destination_image_desc_set_layout() { return destination_desc_set_layout; }

        vk::Sampler get_sampler(){ return sampler; }

        vk::SamplerYcbcrConversion get_yCbCr_conversion(){ return yCbCr_conversion; };
};

struct RenderArea {
        uint32_t x;
        uint32_t y;
        uint32_t width;
        uint32_t height;
};

class RenderPipeline {
        bool valid = false;

        vk::SamplerYcbcrConversion yCbCr_conversion{};
        vk::Sampler sampler{};

        RenderArea render_area{};
        vk::Extent2D render_area_siza{};
        vk::Viewport viewport{};
        vk::Rect2D scissor{};

        vk::ShaderModule vertex_shader{};
        vk::ShaderModule fragment_shader{};

        vk::RenderPass render_pass{};
        vk::ClearValue clear_color{};

        vk::DescriptorSetLayout image_desc_set_layout{};
        vk::PipelineLayout pipeline_layout{};
        vk::Pipeline pipeline{};

public:
        void create(VulkanContext& context, const std::string& path_to_shaders);

        void destroy(vk::Device device);

        void update_render_area(vk::Extent2D render_area_siza, vk::Extent2D image_size);

        /** Invalidates descriptor sets created from stored descriptor set layout**/
        void reconfigure(vk::Device device, vk::Format sampler);

        void record_commands(vk::CommandBuffer cmd_buffer, vk::DescriptorSet image, vk::Framebuffer framebuffer);

        vk::RenderPass get_render_pass(){ return render_pass; }

        vk::DescriptorSetLayout get_image_desc_set_layout(){ return image_desc_set_layout; }

        vk::Sampler get_sampler(){ return sampler; }

        vk::SamplerYcbcrConversion get_yCbCr_conversion(){ return yCbCr_conversion; };
};


} //vulkan_display_detail
