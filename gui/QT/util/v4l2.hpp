#ifndef V4L2_HPP
#define V4L2_HPP

#include <vector>
#include <string>

struct Camera{
	std::string name;
	std::string path;
};

struct Mode{
	int tpf_numerator;
	int tpf_denominator;

	int width;
	int height;

	std::string codec;
};

std::vector<Camera> getCameras();
std::vector<Mode> getModes(const std::string& path);



#endif
