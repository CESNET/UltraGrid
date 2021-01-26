#pragma once

#if defined _MSC_VER || defined __MINGW32__
#define DLLIMPORT __declspec(dllimport)
#else
#define DLLIMPORT
#endif

#ifdef __cplusplus
#define VRG_STREAM_API  extern "C" DLLIMPORT
#else
#define VRG_STREAM_API
#endif

struct ProjectionFovTan {
	float left;
	float right;
	float top;
	float bottom;
};

struct Vector3 {
	float x;
	float y;
	float z;
};

struct Quaternion {
	float x;
	float y;
	float z;
	float w;
};

struct Pose {
	struct Vector3 position;
	struct Quaternion orientation;
};

struct RenderPacket {
	struct ProjectionFovTan left_projection_fov;
	struct Pose left_view_pose;
	struct ProjectionFovTan right_projection_fov;
	struct Pose right_view_pose;
	int pix_width_eye;
	int pix_height_eye;
	unsigned long long timepoint;
	unsigned int frame;
};

enum VrgStreamApiError {
	HmdTestAppNotResponds = -8,
	NoSupportedGPUFound = -7,
	GPUFnNotSupported = -6,
	FrameNotRequested = -5,
	HmdTesterNotInited = -4,
	GPUError = -3,
	NotInited = -2,
	InitFail = -1,
	Ok = 0
};

enum VrgInputFormat {
	RGBA = 0,
	YUV420 = 1,
	NV12 = 2
};

enum VrgMemory {
	CPU = 0,
	DX11 = 1,
	CUDA = 2,
	GL = 3
};

VRG_STREAM_API enum VrgStreamApiError vrgStreamInit(enum VrgInputFormat inputFormat);

VRG_STREAM_API enum VrgStreamApiError vrgStreamRenderFrame(struct RenderPacket *packet);

VRG_STREAM_API enum VrgStreamApiError vrgStreamSubmitFrame(struct RenderPacket* packet, void* sbs_image_data, enum VrgMemory api);

