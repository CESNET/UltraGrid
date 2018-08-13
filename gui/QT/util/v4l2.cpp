#ifdef __linux__
#include <unistd.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <linux/videodev2.h>

#include <cstring>

#include "v4l2.hpp"

std::vector<Camera> getCameras(){
	const int max = 64; //There can be only 64 /dev/videoN devices

	std::vector<Camera> result;

	int fd;
	struct v4l2_capability video_cap;

	for(int i = 0; i < max; i++){
		std::string path = "/dev/video" + std::to_string(i);
		if((fd = open(path.c_str(), O_RDONLY)) == -1){
			continue;
		}

		if(ioctl(fd, VIDIOC_QUERYCAP, &video_cap) == -1){
			continue;
		}

		Camera c;
		c.name = reinterpret_cast<const char*>(video_cap.card);
		c.path = path;

		result.push_back(c);

		close(fd);
	}

	return result;
}

std::vector<Mode> getModes(const std::string& path){
	std::vector<Mode> result;

	struct v4l2_fmtdesc fmt;
	std::memset(&fmt, 0, sizeof(fmt));
	fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	fmt.index = 0;

	int fd;
	if((fd = open(path.c_str(), O_RDONLY)) == -1){
		return result;
	}

	while(ioctl(fd, VIDIOC_ENUM_FMT, &fmt) == 0){
		std::string codec;
		codec = (char *) &fmt.pixelformat;

		struct v4l2_frmsizeenum size;
		std::memset(&size, 0, sizeof(size));
		size.pixel_format = fmt.pixelformat;

		int res = ioctl(fd, VIDIOC_ENUM_FRAMESIZES, &size);

		if(res == -1){
			close(fd);
			return result;
		}

		struct v4l2_frmivalenum frame_int;
		std::memset(&frame_int, 0, sizeof(frame_int));
		frame_int.index = 0;
		frame_int.pixel_format = fmt.pixelformat;

		if(size.type == V4L2_FRMSIZE_TYPE_DISCRETE){
			while(ioctl(fd, VIDIOC_ENUM_FRAMESIZES, &size) == 0){
				frame_int.width = size.discrete.width;
				frame_int.height = size.discrete.height;
				frame_int.index = 0;

				res = ioctl(fd, VIDIOC_ENUM_FRAMEINTERVALS, &frame_int);

				if(res == -1){
					close(fd);
					return result;
				}

				if(frame_int.type != V4L2_FRMIVAL_TYPE_DISCRETE){
					continue;
				}

				Mode m;
				m.codec = codec;
				m.width = frame_int.width;
				m.height = frame_int.height;
				m.tpf_numerator = frame_int.discrete.numerator;
				m.tpf_denominator = frame_int.discrete.denominator;

				result.push_back(m);

				size.index++;
			}
		}
		fmt.index++;
	}

	close(fd);
	return result;
}
#endif
