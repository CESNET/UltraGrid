/**
 * @file   video_display/vulkan_pipelines.cpp
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

#include "vulkan_pipelines.hpp"

#include <cstdint>
#include <cmath>
#include <fstream>
#include <vector>

using namespace vulkan_display;
using namespace vulkan_display_detail;

namespace{

vk::ShaderModule create_shader(

        const std::string& file_path,
        const vk::Device& device)
{
        std::ifstream file(file_path, std::ios::binary);
        if(!file.is_open()){
                throw VulkanError{"Failed to open file:"s + file_path};
        }
        file.seekg(0, std::ios_base::end);
        auto size = file.tellg();
        assert(size % 4 == 0);
        file.seekg(0);
        std::vector<std::uint32_t> shader_code(size / 4);
        file.read(reinterpret_cast<char*>(shader_code.data()), static_cast<std::streamsize>(size));

        if(!file.good()){
                throw VulkanError{"Error reading from file:"s + file_path};
        }

        vk::ShaderModuleCreateInfo shader_info;
        shader_info
                .setCodeSize(shader_code.size() * 4)
                .setPCode(shader_code.data());
        return device.createShaderModule(shader_info);
}

vk::SamplerYcbcrConversion createYCbCrConversion(vk::Device device, vk::Format format){
        vk::SamplerYcbcrConversion yCbCr_conversion = nullptr;
        if (is_yCbCr_format(format)) {
                vk::SamplerYcbcrConversionCreateInfo conversion_info;
                conversion_info
                        .setFormat(format)
                        .setYcbcrModel(vk::SamplerYcbcrModelConversion::eYcbcr709)
                        .setYcbcrRange(vk::SamplerYcbcrRange::eItuNarrow)
                        .setComponents({})
                        .setChromaFilter(vk::Filter::eLinear)
                        .setXChromaOffset(vk::ChromaLocation::eMidpoint)
                        .setYChromaOffset(vk::ChromaLocation::eMidpoint)
                        .setForceExplicitReconstruction(false);
                yCbCr_conversion = device.createSamplerYcbcrConversion(conversion_info);
        }
        return yCbCr_conversion;
}

vk::Sampler create_sampler(vk::Device device, vk::SamplerYcbcrConversion yCbCr_conversion, vk::Filter filter) {
        vk::SamplerYcbcrConversionInfo yCbCr_info{ yCbCr_conversion };
        
        vk::SamplerCreateInfo sampler_info;
        sampler_info
                .setAddressModeU(vk::SamplerAddressMode::eClampToEdge)
                .setAddressModeV(vk::SamplerAddressMode::eClampToEdge)
                .setAddressModeW(vk::SamplerAddressMode::eClampToEdge)
                .setMagFilter(filter)
                .setMinFilter(filter)
                .setAnisotropyEnable(false)
                .setUnnormalizedCoordinates(false)
                .setPNext(yCbCr_conversion ? &yCbCr_info : nullptr);
        return device.createSampler(sampler_info);
}

struct Binding{
        vk::DescriptorType type;
        vk::ShaderStageFlags stages;
        vk::Sampler sampler = nullptr;
};

vk::DescriptorSetLayout create_descriptor_set_layout(vk::Device device, uint32_t first_binding, std::vector<Binding> bindings) {
        std::vector<vk::DescriptorSetLayoutBinding> descriptor_set_layout_bindings(bindings.size());
        for(size_t i = 0; i < bindings.size(); i++){
            descriptor_set_layout_bindings[i]
                .setBinding(first_binding + i)
                .setDescriptorCount(1)
                .setDescriptorType(bindings[i].type)
                .setStageFlags(bindings[i].stages)
                .setPImmutableSamplers(&bindings[i].sampler);
        }
        vk::DescriptorSetLayoutCreateInfo descriptor_set_layout_info{};
        descriptor_set_layout_info
                .setBindingCount(descriptor_set_layout_bindings.size())
                .setPBindings(descriptor_set_layout_bindings.data());

        return device.createDescriptorSetLayout(descriptor_set_layout_info);
}

vk::PipelineLayout create_compute_pipeline_layout(vk::Device device, const std::vector<vk::DescriptorSetLayout>& descriptor_set_layout){
        vk::PushConstantRange push_constant_range{};
        push_constant_range
                .setOffset(0)
                .setSize(sizeof(ImageSize))
                .setStageFlags(vk::ShaderStageFlagBits::eCompute);

        vk::PipelineLayoutCreateInfo pipeline_layout_info;
        pipeline_layout_info
                .setSetLayoutCount(descriptor_set_layout.size())
                .setPSetLayouts(descriptor_set_layout.data())
                .setPushConstantRangeCount(1)
                .setPPushConstantRanges(&push_constant_range);

        return device.createPipelineLayout(pipeline_layout_info);
}

vk::Pipeline create_compute_pipeline(vk::Device device, vk::PipelineLayout pipeline_layout, vk::ShaderModule shader){
        vk::PipelineShaderStageCreateInfo shader_stage_info;
        shader_stage_info
                .setModule(shader)
                .setPName("main")
                .setStage(vk::ShaderStageFlagBits::eCompute);

        vk::ComputePipelineCreateInfo pipeline_info{};
        pipeline_info
                .setStage(shader_stage_info)
                .setLayout(pipeline_layout);

        vk::Pipeline compute_pipeline;
        auto result =  device.createComputePipelines(nullptr, 1, &pipeline_info, nullptr, &compute_pipeline);
        if(result != vk::Result::eSuccess){
                throw VulkanError{"Pipeline cannot be created."};
        }
        return compute_pipeline;
}

} //namespace

namespace vulkan_display_detail {

void ConversionPipeline::create(vk::Device device, const std::string& shader_path, vk::Format format){
        assert(!valid);
        valid = true;

        yCbCr_conversion = createYCbCrConversion(device, format);
        sampler = create_sampler(device, yCbCr_conversion, vk::Filter::eNearest);

        compute_shader = create_shader(shader_path, device);
        source_desc_set_layout = create_descriptor_set_layout(device, 0, {
               {vk::DescriptorType::eCombinedImageSampler, vk::ShaderStageFlagBits::eCompute, sampler}
        });
        destination_desc_set_layout = create_descriptor_set_layout(device, 1, {
               {vk::DescriptorType::eStorageImage, vk::ShaderStageFlagBits::eCompute}
        });
        pipeline_layout = create_compute_pipeline_layout(device,
                {source_desc_set_layout, destination_desc_set_layout});
        pipeline = create_compute_pipeline(device, pipeline_layout, compute_shader);
}

void ConversionPipeline::destroy(vk::Device device){
        if(valid){
                valid = false;
                if (yCbCr_conversion){
                        device.destroy(yCbCr_conversion);
                }
                device.destroy(sampler);
                device.destroy(compute_shader);
                device.destroy(pipeline_layout);
                device.destroy(pipeline);
                device.destroy(source_desc_set_layout);
                device.destroy(destination_desc_set_layout);
        }
}

void ConversionPipeline::record_commands(vk::CommandBuffer cmd_buffer, ImageSize image_size,
        std::array<vk::DescriptorSet, 2> descriptor_sets)
{
        cmd_buffer.pushConstants(pipeline_layout, vk::ShaderStageFlagBits::eCompute,
                0, sizeof(image_size), &image_size);

        cmd_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
        cmd_buffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipeline_layout, 0, descriptor_sets, nullptr);
        cmd_buffer.dispatch(image_size.width / 16, image_size.height / 16, 1);
}

} //namespace vulkan_display_detail

namespace{

vk::RenderPass create_render_pass(vk::Device device, vk::Format swapchain_image_format) {
        vk::RenderPassCreateInfo render_pass_info;

        vk::AttachmentDescription color_attachment;
        color_attachment
                .setFormat(swapchain_image_format)
                .setSamples(vk::SampleCountFlagBits::e1)
                .setLoadOp(vk::AttachmentLoadOp::eClear)
                .setStoreOp(vk::AttachmentStoreOp::eStore)
                .setStencilLoadOp(vk::AttachmentLoadOp::eDontCare)
                .setStencilStoreOp(vk::AttachmentStoreOp::eDontCare)
                .setInitialLayout(vk::ImageLayout::eUndefined)
                .setFinalLayout(vk::ImageLayout::ePresentSrcKHR);
        render_pass_info
                .setAttachmentCount(1)
                .setPAttachments(&color_attachment);

        vk::AttachmentReference attachment_reference;
        attachment_reference
                .setAttachment(0)
                .setLayout(vk::ImageLayout::eColorAttachmentOptimal);
        vk::SubpassDescription subpass;
        subpass
                .setPipelineBindPoint(vk::PipelineBindPoint::eGraphics)
                .setColorAttachmentCount(1)
                .setPColorAttachments(&attachment_reference);
        render_pass_info
                .setSubpassCount(1)
                .setPSubpasses(&subpass);

        vk::SubpassDependency subpass_dependency{};
        subpass_dependency
                .setSrcSubpass(VK_SUBPASS_EXTERNAL)
                .setDstSubpass(0)
                .setSrcStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput)
                .setDstStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput)
                .setDstAccessMask(vk::AccessFlagBits::eColorAttachmentWrite);
        render_pass_info
                .setDependencyCount(1)
                .setPDependencies(&subpass_dependency);

        return device.createRenderPass(render_pass_info);
}

vk::PipelineLayout create_render_pipeline_layout(vk::Device device, vk::DescriptorSetLayout descriptor_set_layout){
        vk::PipelineLayoutCreateInfo pipeline_layout_info{};

        vk::PushConstantRange push_constants;
        push_constants
                .setOffset(0)
                .setSize(sizeof(RenderArea))
                .setStageFlags(vk::ShaderStageFlagBits::eFragment);
        pipeline_layout_info
                .setPushConstantRangeCount(1)
                .setPPushConstantRanges(&push_constants)
                .setSetLayoutCount(1)
                .setPSetLayouts(&descriptor_set_layout);
        return device.createPipelineLayout(pipeline_layout_info);
}


vk::Pipeline create_render_pipeline(vk::Device device, vk::PipelineLayout render_pipeline_layout, vk::RenderPass render_pass,
        vk::ShaderModule vertex_shader, vk::ShaderModule fragment_shader)
{
        vk::GraphicsPipelineCreateInfo pipeline_info{};

        std::array<vk::PipelineShaderStageCreateInfo, 2> shader_stages_infos;
        shader_stages_infos[0]
                .setModule(vertex_shader)
                .setPName("main")
                .setStage(vk::ShaderStageFlagBits::eVertex);
        shader_stages_infos[1]
                .setModule(fragment_shader)
                .setPName("main")
                .setStage(vk::ShaderStageFlagBits::eFragment);
        pipeline_info
                .setStageCount(static_cast<uint32_t>(shader_stages_infos.size()))
                .setPStages(shader_stages_infos.data());

        vk::PipelineVertexInputStateCreateInfo vertex_input_state_info{};
        pipeline_info.setPVertexInputState(&vertex_input_state_info);

        vk::PipelineInputAssemblyStateCreateInfo input_assembly_state_info{};
        input_assembly_state_info.setTopology(vk::PrimitiveTopology::eTriangleList);
        pipeline_info.setPInputAssemblyState(&input_assembly_state_info);

        vk::PipelineViewportStateCreateInfo viewport_state_info;
        viewport_state_info
                .setScissorCount(1)
                .setViewportCount(1);
        pipeline_info.setPViewportState(&viewport_state_info);

        vk::PipelineRasterizationStateCreateInfo rasterization_info{};
        rasterization_info
                .setPolygonMode(vk::PolygonMode::eFill)
                .setLineWidth(1.f);
        pipeline_info.setPRasterizationState(&rasterization_info);

        vk::PipelineMultisampleStateCreateInfo multisample_info;
        multisample_info
                .setSampleShadingEnable(false)
                .setRasterizationSamples(vk::SampleCountFlagBits::e1);
        pipeline_info.setPMultisampleState(&multisample_info);

        using ColorFlags = vk::ColorComponentFlagBits;
        vk::PipelineColorBlendAttachmentState color_blend_attachment{};
        color_blend_attachment
                .setBlendEnable(false)
                .setColorWriteMask(ColorFlags::eR | ColorFlags::eG | ColorFlags::eB | ColorFlags::eA);
        vk::PipelineColorBlendStateCreateInfo color_blend_info{};
        color_blend_info
                .setAttachmentCount(1)
                .setPAttachments(&color_blend_attachment);
        pipeline_info.setPColorBlendState(&color_blend_info);

        std::array dynamic_states{ vk::DynamicState::eViewport, vk::DynamicState::eScissor };
        vk::PipelineDynamicStateCreateInfo dynamic_state_info{};
        dynamic_state_info
                .setDynamicStateCount(static_cast<uint32_t>(dynamic_states.size()))
                .setPDynamicStates(dynamic_states.data());
        pipeline_info.setPDynamicState(&dynamic_state_info);

        pipeline_info
                .setLayout(render_pipeline_layout)
                .setRenderPass(render_pass);

        vk::Pipeline render_pipeline;
        auto result = device.createGraphicsPipelines(nullptr, 1, &pipeline_info, nullptr, &render_pipeline);
        if(result != vk::Result::eSuccess){
                throw VulkanError{"Pipeline cannot be created."};
        }
        return render_pipeline;
}

} //namespace

namespace vulkan_display_detail {

void RenderPipeline::create(VulkanContext& context, const std::string& path_to_shaders){
        assert(!valid);
        valid = true;

        auto device = context.get_device();
        vertex_shader = create_shader(path_to_shaders + "/render.vert.spv", device);
        fragment_shader = create_shader(path_to_shaders + "/render.frag.spv", device);
        render_pass = create_render_pass(device, context.get_swapchain_image_format());

        vk::ClearColorValue clear_color_value{};
        clear_color_value.setFloat32({ 0.01f, 0.01f, 0.01f, 1.0f });
        clear_color.setColor(clear_color_value);
}

void RenderPipeline::destroy(vk::Device device){
        if(valid){
                valid = false;
                device.destroy(pipeline);
                device.destroy(pipeline_layout);
                device.destroy(image_desc_set_layout);
                device.destroy(render_pass);
                device.destroy(fragment_shader);
                device.destroy(vertex_shader);
                device.destroy(sampler);
                if(yCbCr_conversion){
                        device.destroy(yCbCr_conversion);
                }
        }
}

void RenderPipeline::update_render_area(vk::Extent2D render_area_size, vk::Extent2D image_size){
        double wnd_aspect = static_cast<double>(render_area_size.width) / render_area_size.height;
        double img_aspect = static_cast<double>(image_size.width) / image_size.height;

        this->render_area_siza = render_area_size;

        if (wnd_aspect > img_aspect) {
                render_area.height = render_area_size.height;
                render_area.width = static_cast<uint32_t>(std::round(render_area_size.height * img_aspect));
                render_area.x = (render_area_size.width - render_area.width) / 2;
                render_area.y = 0;
        } else {
                render_area.width = render_area_size.width;
                render_area.height = static_cast<uint32_t>(std::round(render_area_size.width / img_aspect));
                render_area.x = 0;
                render_area.y = (render_area_size.height - render_area.height) / 2;
        }

        viewport
                .setX(static_cast<float>(render_area.x))
                .setY(static_cast<float>(render_area.y))
                .setWidth(static_cast<float>(render_area.width))
                .setHeight(static_cast<float>(render_area.height))
                .setMinDepth(0.f)
                .setMaxDepth(1.f);
        scissor
                .setOffset({ static_cast<int32_t>(render_area.x), static_cast<int32_t>(render_area.y) })
                .setExtent({ render_area.width, render_area.height });
}

void RenderPipeline::reconfigure(vk::Device device, vk::Format format){
        device.destroy(pipeline);
        device.destroy(pipeline_layout);
        device.destroy(image_desc_set_layout);
        device.destroy(sampler);
        if (yCbCr_conversion){
                device.destroy(yCbCr_conversion);
        }

        yCbCr_conversion = createYCbCrConversion(device, format);
        sampler = create_sampler(device, yCbCr_conversion, vk::Filter::eLinear);
        image_desc_set_layout = create_descriptor_set_layout(device, 0, {
            {vk::DescriptorType::eCombinedImageSampler, vk::ShaderStageFlagBits::eFragment, sampler}
        });
        pipeline_layout = create_render_pipeline_layout(device, image_desc_set_layout);
        pipeline = create_render_pipeline(device, pipeline_layout, render_pass, vertex_shader, fragment_shader);
}

void RenderPipeline::record_commands(vk::CommandBuffer cmd_buffer, vk::DescriptorSet image, vk::Framebuffer framebuffer){
        vk::RenderPassBeginInfo render_pass_begin_info;
        render_pass_begin_info
                .setRenderPass(render_pass)
                .setRenderArea(vk::Rect2D{ {0,0}, render_area_siza })
                .setClearValueCount(1)
                .setPClearValues(&clear_color)
                .setFramebuffer(framebuffer);
        cmd_buffer.beginRenderPass(render_pass_begin_info, vk::SubpassContents::eInline);

        cmd_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);

        cmd_buffer.setScissor(0, scissor);
        cmd_buffer.setViewport(0, viewport);
        cmd_buffer.pushConstants(pipeline_layout, vk::ShaderStageFlagBits::eFragment, 0, sizeof(render_area), &render_area);
        cmd_buffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                pipeline_layout, 0, image, nullptr);
        cmd_buffer.draw(3, 1, 0, 0);

        cmd_buffer.endRenderPass();
}

} //namespace vulkan_display_detail
