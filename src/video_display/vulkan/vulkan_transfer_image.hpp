/**
 * @file   video_display/vulkan_transfer_image.cpp
 * @author Martin Bela      <492789@mail.muni.cz>
 */
/*
 * Copyright (c) 2021-2025 CESNET, zajmové sdružení právnických osob
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
#include <functional>

namespace vulkan_display {

enum class Format{
        uninitialized,
        RGBA8,
        RGB8,
        UYVY8_422,
        UYVY8_422_conv,
        YUYV8_422,
        VUYA8_4444_conv,
        YUYV16_422,
        UYVA16_4444_conv,
        RGB10A2_conv,
        RGB16
};

struct ImageDescription;

} // vulkan_display-----------------------------------------

namespace vulkan_display_detail {

struct FormatInfo{
        vk::Format buffer_format;
        const char *conversion_shader = "";
        vk::Format conversion_image_format{};
};

inline FormatInfo format_info(vulkan_display::Format format){

        using F = vulkan_display::Format;
        using VkF = vk::Format;

        switch (format) {
        case F::uninitialized:    return { VkF::eUndefined,            };
        case F::RGBA8:            return { VkF::eR8G8B8A8Unorm,        };
        case F::RGB8:             return { VkF::eR8G8B8Srgb,           };
        case F::UYVY8_422:        return { VkF::eB8G8R8G8422Unorm,     };
        case F::UYVY8_422_conv:   return { VkF::eR8G8B8A8Unorm,        "UYVY8_conv", VkF::eR8G8B8A8Unorm };
        case F::YUYV8_422:        return { VkF::eG8B8G8R8422Unorm,     };
        case F::VUYA8_4444_conv:  return { VkF::eR8G8B8A8Unorm,        "VUYA8_conv", VkF::eR8G8B8A8Unorm };
        case F::YUYV16_422:       return { VkF::eG16B16G16R16422Unorm, };
        case F::UYVA16_4444_conv: return { VkF::eR16G16B16A16Uint,     "UYVA16_conv", VkF::eR16G16B16A16Sfloat };
        case F::RGB10A2_conv:     return { VkF::eR8G8B8A8Uint,         "RGB10A2_conv", VkF::eA2B10G10R10UnormPack32 };
        case F::RGB16:            return { VkF::eR16G16B16Unorm        };
        };
        abort();
}

vk::Extent2D get_buffer_size(const vulkan_display::ImageDescription& description);

} //vulkan_display_detail

namespace vulkan_display{

struct ImageDescription {
        vk::Extent2D size{};
        Format format = Format::uninitialized;

        ImageDescription() = default;
        ImageDescription(vk::Extent2D size, Format format) :
                size{ size }, format{ format } { }
        ImageDescription(uint32_t width, uint32_t height, Format format) :
                ImageDescription{ vk::Extent2D{width, height}, format } { }

        bool operator==(const ImageDescription& other) const {
                return size.width == other.size.width 
                        && size.height == other.size.height 
                        && format == other.format;
        }

        bool operator!=(const ImageDescription& other) const {
                return !(*this == other);
        }

        detail::FormatInfo format_info() const { return detail::format_info(format); }
};

class TransferImage;

} //vulkan_display

namespace vulkan_display_detail{

enum class MemoryLocation{
        device_local, host_local        
};

enum class InitialImageData{
        preinitialised, undefined
};

struct Image2D{
        void init(VulkanContext& context,
                vk::Extent2D size, vk::Format format, vk::ImageUsageFlags usage,
                vk::AccessFlags initial_access, InitialImageData preinitialised, MemoryLocation memory_location);

        void init(VulkanContext& context,
                vk::Extent2D size, vk::Format format, vk::ImageUsageFlags usage,
                vk::AccessFlags initial_access, InitialImageData preinitialised, vk::ImageTiling tiling,
                vk::MemoryPropertyFlags requested_properties, vk::MemoryPropertyFlags optional_properties);

        vk::ImageView get_image_view(vk::Device device, vk::SamplerYcbcrConversion conversion);
        
        void destroy(vk::Device device);

        vk::ImageMemoryBarrier create_memory_barrier(vk::ImageLayout new_layout, vk::AccessFlags new_access_mask,
                uint32_t src_queue_family_index = VK_QUEUE_FAMILY_IGNORED,
                uint32_t dst_queue_family_index = VK_QUEUE_FAMILY_IGNORED);

        vk::DeviceMemory memory{};
        vk::Image image{};
        vk::ImageLayout layout{};
        vk::AccessFlags access{};

        vk::ImageView view{};

        size_t byte_size{};

        vk::Extent2D size{};
        vk::Format format{};
};

class TransferImageImpl {
public:
        TransferImageImpl() = default;
        TransferImageImpl(vk::Device device, uint32_t id) {
                init(device, id);
        }

        void init(vk::Device device, uint32_t id);
        void recreate(VulkanContext& context, vulkan_display::ImageDescription description);
        void destroy(vk::Device device);

        void preprocess();

        uint32_t get_id() const { return id; }

        vulkan_display::ImageDescription get_image_description() const {
                return image_description;
        }
        
        Image2D& get_buffer() {
                return buffer;        
        }

        vk::ImageView get_image_view(vk::Device device, vk::SamplerYcbcrConversion conversion) {
            return buffer.get_image_view(device, conversion);
        }

        friend class vulkan_display::TransferImage;
        static constexpr uint32_t NO_ID = UINT32_MAX;

        vk::Fence is_available_fence; // is_available_fence becomes signalled when gpu releases the image

private:
        Image2D buffer;
        uint32_t id = NO_ID;
        unsigned char* ptr = nullptr;

        vk::DeviceSize row_pitch = 0;

        using PreprocessFunction = std::function<void(vulkan_display::TransferImage& image)>;
        PreprocessFunction preprocess_fun{ nullptr };

        vulkan_display::ImageDescription image_description;
};

} // vulkan_display_detail


namespace vulkan_display {

class TransferImage {
public:
        TransferImage() = default;
        TransferImage(std::nullptr_t){}

        explicit TransferImage(detail::TransferImageImpl& image) :
                transfer_image{ &image }
        { 
                assert(image.id != detail::TransferImageImpl::NO_ID);
        }

        uint32_t get_id() const {
                assert(transfer_image); 
                return transfer_image->id;
        }

        unsigned char* get_memory_ptr() {
                assert(transfer_image);
                return transfer_image->ptr;
        }

        ImageDescription get_description(){
                return transfer_image->image_description;
        }

        vk::DeviceSize get_row_pitch() const {
                assert(transfer_image);
                return transfer_image->row_pitch;
        }

        vk::Extent2D get_size() const {
                return transfer_image->image_description.size;
        }

        vulkan_display_detail::TransferImageImpl* get_transfer_image() {
                return transfer_image;
        }

        void set_process_function(std::function<void(TransferImage& image)> function) {
                transfer_image->preprocess_fun = std::move(function);
        }

        bool operator==(TransferImage other) const {
                return transfer_image == other.transfer_image;
        }

private:
        detail::TransferImageImpl* transfer_image = nullptr;
};

} //namespace vulkan_display

