/**
 * @file   video_display/vulkan_transfer_image.cpp
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

#include "vulkan_transfer_image.hpp"

using namespace vulkan_display_detail;
using namespace vulkan_display;

namespace {

/*constexpr vk::DeviceSize add_padding(vk::DeviceSize size, vk::DeviceSize allignment) {
        vk::DeviceSize remainder = size % allignment;
        if (remainder == 0) {
                return size;
        }
        return size + allignment - remainder;
}*/


/**
 * Check if the required flags are present among the provided flags
 */
template<typename T>
constexpr bool flags_present(T provided_flags, T required_flags) {
        return (provided_flags & required_flags) == required_flags;
}

uint32_t get_memory_type(
        uint32_t memory_type_bits,
        vk::MemoryPropertyFlags requested_properties, vk::MemoryPropertyFlags optional_properties,
        vk::PhysicalDevice gpu)
{
        uint32_t possible_memory_type = UINT32_MAX;
        auto supported_properties = gpu.getMemoryProperties();
        for (uint32_t i = 0; i < supported_properties.memoryTypeCount; i++) {
                // if i-th bit in memory_type_bits is set, than i-th memory type can be used
                bool is_type_usable = (1u << i) & memory_type_bits;
                auto& mem_type = supported_properties.memoryTypes[i];
                if (flags_present(mem_type.propertyFlags, requested_properties) && is_type_usable) {
                        if (flags_present(mem_type.propertyFlags, optional_properties)) {
                                return i;
                        }
                        possible_memory_type = i;
                }
        }
        if (possible_memory_type != UINT32_MAX) {
                return possible_memory_type;
        }
        throw VulkanError{"No available memory for transfer images found."};
}

} //namespace -------------------------------------------------------------

namespace vulkan_display_detail{

vk::Extent2D get_buffer_size(const vulkan_display::ImageDescription& description){
        if (description.format == vulkan_display::Format::UYVY8_422_conv){
                return { description.size.width / 2, description.size.height };
        }
        return description.size;
}

void TransferImageImpl::init(vk::Device device, uint32_t id) {
        this->id = id;
        vk::FenceCreateInfo fence_info{};
        is_available_fence = device.createFence(fence_info);
}

void Image2D::init(VulkanContext& context,
        vk::Extent2D size, vk::Format format, vk::ImageUsageFlags usage, 
        vk::AccessFlags initial_access, InitialImageData preinitialised, MemoryLocation memory_location)
{
        vk::ImageTiling tiling;
        vk::MemoryPropertyFlags requested_properties;
        vk::MemoryPropertyFlags optional_properties;
        
        using MemBits = vk::MemoryPropertyFlagBits;
        if (memory_location == MemoryLocation::host_local){
                tiling = vk::ImageTiling::eLinear;
                requested_properties = MemBits::eHostVisible | MemBits::eHostCoherent;
                optional_properties = MemBits::eHostCached;
        }
        else{
                tiling = vk::ImageTiling::eOptimal;
                requested_properties = {};
                optional_properties = MemBits::eDeviceLocal;
        }
        this->init(context, size, format, usage, initial_access, preinitialised, tiling, requested_properties, optional_properties);
}

void Image2D::init(VulkanContext& context, vk::Extent2D size, vk::Format format, vk::ImageUsageFlags usage,
        vk::AccessFlags initial_access, InitialImageData preinitialised, vk::ImageTiling tiling,
        vk::MemoryPropertyFlags requested_properties, vk::MemoryPropertyFlags optional_properties)
{
        this->format = format;
        this->size = size;
        this->access = initial_access;
        this->layout = preinitialised == InitialImageData::preinitialised ? 
                vk::ImageLayout::ePreinitialized : 
                vk::ImageLayout::eUndefined;
        this->view = nullptr;

        vk::Device device = context.get_device();
        vk::ImageCreateInfo image_info{};
        image_info
                .setImageType(vk::ImageType::e2D)
                .setExtent(vk::Extent3D{ size, 1 })
                .setMipLevels(1)
                .setArrayLayers(1)
                .setFormat(format)
                .setTiling(tiling)
                .setInitialLayout(layout)
                .setUsage(usage)
                .setSharingMode(vk::SharingMode::eExclusive)
                .setSamples(vk::SampleCountFlagBits::e1);
        image = device.createImage(image_info);

        vk::MemoryRequirements memory_requirements = device.getImageMemoryRequirements(image);
        byte_size = memory_requirements.size;

        uint32_t memory_type = get_memory_type(memory_requirements.memoryTypeBits,
                requested_properties, optional_properties, context.get_gpu());

        vk::MemoryAllocateInfo allocInfo{ byte_size , memory_type };
        memory = device.allocateMemory(allocInfo);

        device.bindImageMemory(image, memory, 0);
}

vk::ImageView Image2D::get_image_view(vk::Device device, vk::SamplerYcbcrConversion conversion) {
        if(!view){
                assert(image);
                vk::ImageViewCreateInfo view_info =
                        default_image_view_create_info(format);
                view_info.setImage(image);

                vk::SamplerYcbcrConversionInfo yCbCr_info{ conversion };
                view_info.setPNext(conversion ? &yCbCr_info : nullptr);
                view = device.createImageView(view_info);
        }
        return view;
}

void Image2D::destroy(vk::Device device) {
        device.destroy(view);
        view = nullptr;
        device.destroy(image);
        image = nullptr;

        if (memory) {
                device.freeMemory(memory);
        }
}

void TransferImageImpl::recreate(VulkanContext& context, ImageDescription description) {
        assert(id != NO_ID);
        buffer.destroy(context.get_device());
        
        auto device = context.get_device();

        buffer.init(context, get_buffer_size(description), description.format_info().buffer_format, vk::ImageUsageFlagBits::eSampled, vk::AccessFlagBits::eHostWrite,
                InitialImageData::preinitialised, MemoryLocation::host_local);
        
        void* void_ptr = device.mapMemory(buffer.memory, 0, buffer.byte_size);
        if (void_ptr == nullptr) {
                throw VulkanError{"Image memory cannot be mapped."};
        }
        ptr = reinterpret_cast<std::byte*>(void_ptr);

        vk::ImageSubresource subresource{ vk::ImageAspectFlagBits::eColor, 0, 0 };
        row_pitch = device.getImageSubresourceLayout(buffer.image, subresource).rowPitch;

        image_description = description;
}

vk::ImageMemoryBarrier  Image2D::create_memory_barrier(
        vk::ImageLayout new_layout, vk::AccessFlags new_access_mask,
        uint32_t src_queue_family_index, uint32_t dst_queue_family_index)
{
        vk::ImageMemoryBarrier memory_barrier{};
        memory_barrier
                .setImage(image)
                .setOldLayout(layout)
                .setNewLayout(new_layout)
                .setSrcAccessMask(access)
                .setDstAccessMask(new_access_mask)
                .setSrcQueueFamilyIndex(src_queue_family_index)
                .setDstQueueFamilyIndex(dst_queue_family_index);
        memory_barrier.subresourceRange
                .setAspectMask(vk::ImageAspectFlagBits::eColor)
                .setLayerCount(1)
                .setLevelCount(1);

        layout = new_layout;
        access = new_access_mask;
        return memory_barrier;
}

void TransferImageImpl::preprocess() {
        if (preprocess_fun) {
                vulkan_display::TransferImage img{ *this };
                preprocess_fun(img);
                img.set_process_function(nullptr);
        }
}

void TransferImageImpl::destroy(vk::Device device) {
        device.unmapMemory(buffer.memory);
        buffer.destroy(device);
        device.destroy(is_available_fence);
}

} //vulkan_display_detail
