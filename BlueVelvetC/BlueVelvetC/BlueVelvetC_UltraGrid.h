#ifndef HG_BLUEVELVET_C
#define HG_BLUEVELVET_C

#include <objbase.h>

#include "BlueVelvet4.h"
#include "BlueHancUtils.h"

#ifdef BLUEVELVETC_EXPORTS
	#define BLUEVELVETC_API __declspec(dllexport)
#else
	#define BLUEVELVETC_API __declspec(dllimport)
#endif


typedef void* BLUEVELVETC_HANDLE;


extern "C"
{
	BLUEVELVETC_API const char* bfcGetVersion();
	BLUEVELVETC_API BLUEVELVETC_HANDLE bfcFactory();
	BLUEVELVETC_API void bfcDestroy(BLUEVELVETC_HANDLE pHandle);

	BLUEVELVETC_API int bfcEnumerate(BLUEVELVETC_HANDLE pHandle, int& iDevices);
	BLUEVELVETC_API int bfcQueryCardType(BLUEVELVETC_HANDLE pHandle, int iDeviceID=0);
	BLUEVELVETC_API int bfcAttach(BLUEVELVETC_HANDLE pHandle, int& iDeviceId);
	BLUEVELVETC_API int bfcDetach(BLUEVELVETC_HANDLE pHandle);

	BLUEVELVETC_API int bfcQueryCardProperty32(BLUEVELVETC_HANDLE pHandle, int iProperty, unsigned int& nValue);
	BLUEVELVETC_API int bfcSetCardProperty32(BLUEVELVETC_HANDLE pHandle, int iProperty, unsigned int& nValue);
	BLUEVELVETC_API int bfcQueryCardProperty64(BLUEVELVETC_HANDLE pHandle, int iProperty, unsigned long long& nValue);
	BLUEVELVETC_API int bfcSetCardProperty64(BLUEVELVETC_HANDLE pHandle, int iProperty, unsigned long long& nValue);

	BLUEVELVETC_API int bfcGetCardSerialNumber(BLUEVELVETC_HANDLE pHandle, char* pSerialNumber, unsigned int nStringSize); //nStringSize must be at least 20

	BLUEVELVETC_API int bfcVideoCaptureStart(BLUEVELVETC_HANDLE pHandle);
	BLUEVELVETC_API int bfcVideoCaptureStop(BLUEVELVETC_HANDLE pHandle);

	BLUEVELVETC_API int bfcVideoPlaybackStart(BLUEVELVETC_HANDLE pHandle, int iStep, int iLoop);
	BLUEVELVETC_API int bfcVideoPlaybackStop(BLUEVELVETC_HANDLE pHandle, int iWait, int iFlush);

	BLUEVELVETC_API int bfcWaitVideoInputSync(BLUEVELVETC_HANDLE pHandle, unsigned long ulUpdateType, unsigned long& ulFieldCount);
	BLUEVELVETC_API int bfcWaitVideoOutputSync(BLUEVELVETC_HANDLE pHandle, unsigned long ulUpdateType, unsigned long& ulFieldCount);

	BLUEVELVETC_API int bfcSystemBufferReadAsync(BLUEVELVETC_HANDLE pHandle, unsigned char* pPixels, unsigned long ulSize, OVERLAPPED* pOverlap, unsigned long ulBufferID, unsigned long ulOffset=0);
	BLUEVELVETC_API int bfcSystemBufferWriteAsync(BLUEVELVETC_HANDLE pHandle, unsigned char* pPixels, unsigned long ulSize, OVERLAPPED* pOverlap, unsigned long ulBufferID, unsigned long ulOffset=0);

	BLUEVELVETC_API int bfcGetCaptureVideoFrameInfoEx(BLUEVELVETC_HANDLE pHandle, OVERLAPPED* pOverlap, struct blue_videoframe_info_ex& VideoFrameInfo, int iCompostLater, unsigned int* nCaptureFifoSize);
	BLUEVELVETC_API int bfcVideoPlaybackAllocate(BLUEVELVETC_HANDLE pHandle, void** pAddress, unsigned long& ulBufferID, unsigned long& ulUnderrun);
	BLUEVELVETC_API int bfcVideoPlaybackPresent(BLUEVELVETC_HANDLE pHandle, unsigned long& ulUniqueID, unsigned long ulBufferID, unsigned long ulCount, int iKeep, int iOdd=0);
	BLUEVELVETC_API int bfcVideoPlaybackRelease(BLUEVELVETC_HANDLE pHandle, unsigned long ulBufferID);

	BLUEVELVETC_API int bfcRenderBufferCapture(BLUEVELVETC_HANDLE pHandle, unsigned long ulBufferID);
	BLUEVELVETC_API int bfcRenderBufferUpdate(BLUEVELVETC_HANDLE pHandle, unsigned long ulBufferID);

	//AUDIO Helper functions (BlueHancUtils)
	BLUEVELVETC_API int bfcEncodeHancFrameEx(BLUEVELVETC_HANDLE pHandle, unsigned int nCardType, struct hanc_stream_info_struct* pHancEncodeInfo, void* pAudioBuffer, unsigned int nAudioChannels, unsigned int nAudioSamples, unsigned int nSampleType, unsigned int nAudioFlags);
	BLUEVELVETC_API int bfcDecodeHancFrameEx(BLUEVELVETC_HANDLE pHandle, unsigned int nCardType, unsigned int* pHancBuffer, struct hanc_decode_struct* pHancDecodeInfo);

	//Miscellaneous functions
	BLUEVELVETC_API int bfcGetReferenceClockPhaseSettings(BLUEVELVETC_HANDLE pHandle, unsigned int& nHPhase, unsigned int& nVPhase, unsigned int& nHPhaseMax, unsigned int& nVPhaseMax);
	BLUEVELVETC_API int bfcLoadOutputLUT1D(BLUEVELVETC_HANDLE pHandle, struct blue_1d_lookup_table_struct* pLutData);
	BLUEVELVETC_API int bfcControlVideoScaler(BLUEVELVETC_HANDLE pHandle,	unsigned int nScalerId,
																			bool bOnlyReadValue,
																			float* pSrcVideoHeight,
																			float* pSrcVideoWidth,
																			float* pSrcVideoYPos,
																			float* pSrcVideoXPos,
																			float* pDestVideoHeight,
																			float* pDestVideoWidth,
																			float* pDestVideoYPos,
																			float* pDestVideoXPos);

	// Custom functions
	BLUEVELVETC_API blue_video_sync_struct* bfcNewVideoSyncStruct(BLUEVELVETC_HANDLE pHandle);
	BLUEVELVETC_API void bfcVideoSyncStructSet(BLUEVELVETC_HANDLE pHandle, blue_video_sync_struct* pIrqInfo, unsigned int uiVideoChannel, unsigned int uiWaitType, unsigned int uiTimeoutMs);
	BLUEVELVETC_API void bfcWaitVideoSyncAsync(BLUEVELVETC_HANDLE pHandle, OVERLAPPED* pOverlap, blue_video_sync_struct *pIrqInfo);
	BLUEVELVETC_API HANDLE bfcGetHandle(BLUEVELVETC_HANDLE pHandle);
	BLUEVELVETC_API void bfcVideoSyncStructGet(BLUEVELVETC_HANDLE pHandle, blue_video_sync_struct* pIrqInfo, unsigned int &VideoMsc, unsigned int &SubfieldInterrupt);

} //extern "C"


#endif //HG_BLUEVELVET_C
