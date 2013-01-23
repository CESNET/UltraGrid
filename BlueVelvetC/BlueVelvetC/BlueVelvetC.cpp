#ifndef UUID_94639C46_79BD_11E2_9DBA_F0DEF1A0ACC9
#define UUID_94639C46_79BD_11E2_9DBA_F0DEF1A0ACC9

#include "BlueVelvetC_UltraGrid.h"

BLUEVELVETC_API BLUEVELVETC_HANDLE bfcFactory()
{
	return BlueVelvetFactory4();
}

void bfcDestroy(BLUEVELVETC_HANDLE pSDK)
{
        BlueVelvetDestroy((CBlueVelvet4 *) pSDK);
}

int bfcQueryCardProperty32(BLUEVELVETC_HANDLE pSDK, int property, unsigned int &value)
{
        VARIANT varVal;
        varVal.vt = VT_UI4;

        BErr err = ((CBlueVelvet4 *) pSDK)->QueryCardProperty(property, varVal);
        value = varVal.ulVal;

        return err;
}

int bfcSetCardProperty32(BLUEVELVETC_HANDLE pSDK, int property, unsigned int &value)
{
        VARIANT varVal;
        varVal.vt = VT_UI4;
        varVal.ulVal = value;

        BErr err = ((CBlueVelvet4 *) pSDK)->SetCardProperty(property, varVal);

        return err;
}

int bfcEnumerate(BLUEVELVETC_HANDLE pSDK, int &iDevices)
{
        return ((CBlueVelvet4 *) pSDK)->device_enumerate(iDevices);
}

int bfcAttach(BLUEVELVETC_HANDLE pSDK, int &iDeviceId)
{
        return ((CBlueVelvet4 *) pSDK)->device_attach(iDeviceId, 0);
}

int bfcDetach(BLUEVELVETC_HANDLE pSDK)
{
        return ((CBlueVelvet4 *) pSDK)->device_detach();
}

int bfcVideoCaptureStart(BLUEVELVETC_HANDLE pSDK)
{
        return ((CBlueVelvet4 *) pSDK)->video_capture_start(0);
}

int bfcVideoCaptureStop(BLUEVELVETC_HANDLE pSDK)
{
        return ((CBlueVelvet4 *) pSDK)->video_capture_stop();
}

int bfcVideoPlaybackStart(BLUEVELVETC_HANDLE  pSDK, int iStep, int iLoop)
{
		return ((CBlueVelvet4 *) pSDK)->video_playback_start(iStep, iLoop);
}

int bfcVideoPlaybackStop(BLUEVELVETC_HANDLE pSDK, int iWait, int iFlush)
{
        return ((CBlueVelvet4 *) pSDK)->video_playback_stop(iWait, iFlush);
}

int bfcWaitVideoInputSync(BLUEVELVETC_HANDLE pSDK, unsigned long ulUpdateType, unsigned long& ulFieldCount) {
        return ((CBlueVelvet4 *) pSDK)->wait_input_video_synch(ulUpdateType, ulFieldCount);
}

int bfcWaitVideoOutputSync(BLUEVELVETC_HANDLE pSDK, unsigned long ulUpdateType, unsigned long& ulFieldCount) {
        return ((CBlueVelvet4 *) pSDK)->wait_output_video_synch(ulUpdateType, ulFieldCount);
}

int bfcQueryCardType(BLUEVELVETC_HANDLE pSDK)
{
        return ((CBlueVelvet4 *) pSDK)->has_video_cardtype();
};

int bfcDecodeHancFrameEx(BLUEVELVETC_HANDLE pHandle, unsigned int nCardType, unsigned int* pHancBuffer, struct hanc_decode_struct* pHancDecodeInfo)
{
        return hanc_decoder_ex(nCardType, pHancBuffer, pHancDecodeInfo);
}

int bfcEncodeHancFrameEx(BLUEVELVETC_HANDLE pHandle, unsigned int nCardType, struct hanc_stream_info_struct* pHancEncodeInfo, void *pAudioBuffer, unsigned int nAudioChannels, unsigned int nAudioSamples, unsigned int nSampleType, unsigned int nAudioFlags)
{
        return encode_hanc_frame_ex(nCardType, pHancEncodeInfo, pAudioBuffer, nAudioChannels, nAudioSamples, nSampleType, nAudioFlags);
}

int bfcSystemBufferWriteAsync(BLUEVELVETC_HANDLE pSDK, unsigned char *pPixels, unsigned long ulSize, OVERLAPPED *pOverlap, unsigned long ulBufferID, unsigned long ulOffset)
{
        return ((CBlueVelvet4 *) pSDK)->system_buffer_write_async(pPixels, ulSize, pOverlap, ulBufferID, ulOffset);
}

int bfcRenderBufferCapture(BLUEVELVETC_HANDLE pSDK, unsigned long ulBufferID)
{
        return ((CBlueVelvet4 *) pSDK)->render_buffer_capture(ulBufferID, 0);
}

int bfcRenderBufferUpdate(BLUEVELVETC_HANDLE pSDK, unsigned long ulBufferID)
{
        return ((CBlueVelvet4 *) pSDK)->render_buffer_update(ulBufferID);
}

int bfcGetCaptureVideoFrameInfoEx(BLUEVELVETC_HANDLE pHandle, OVERLAPPED* pOverlap, struct blue_videoframe_info_ex& VideoFrameInfo, int iCompostLater, unsigned int* nCaptureFifoSize)
{
	return GetVideo_CaptureFrameInfoEx((CBlueVelvet4 *) pHandle, pOverlap, VideoFrameInfo, iCompostLater, nCaptureFifoSize);
}

int bfcSystemBufferReadAsync(BLUEVELVETC_HANDLE pHandle, unsigned char* pPixels, unsigned long ulSize, OVERLAPPED* pOverlap, unsigned long ulBufferID, unsigned long ulOffset)
{
	return ((CBlueVelvet4 *) pHandle)->system_buffer_read_async(pPixels, ulSize, pOverlap, ulBufferID, ulOffset);
}

int bfcQueryCardType(BLUEVELVETC_HANDLE pHandle, int iDeviceID)
{
	return ((CBlueVelvet4 *) pHandle)->has_video_cardtype();
}

blue_video_sync_struct* bfcNewVideoSyncStruct(BLUEVELVETC_HANDLE pHandle)
{
	blue_video_sync_struct* pIrqInfo = new blue_video_sync_struct;
	memset(pIrqInfo, 0, sizeof(blue_video_sync_struct));
	return pIrqInfo;
}

void bfcVideoSyncStructSet(BLUEVELVETC_HANDLE pHandle, blue_video_sync_struct* pIrqInfo, unsigned int uiVideoChannel, unsigned int uiWaitType, unsigned int uiTimeoutMs)
{
	pIrqInfo->video_channel = uiVideoChannel;
	pIrqInfo->sync_wait_type = uiWaitType;
	pIrqInfo->timeout_video_msc = uiTimeoutMs;
}

void bfcWaitVideoSyncAsync(BLUEVELVETC_HANDLE pHandle, OVERLAPPED* pOverlap, blue_video_sync_struct *pIrqInfo)
{
	blue_wait_video_sync_async((CBlueVelvet4 *) pHandle, pOverlap, pIrqInfo);
}

HANDLE bfcGetHandle(BLUEVELVETC_HANDLE pHandle)
{
	return ((CBlueVelvet4 *) pHandle)->m_hDevice;
}

void bfcVideoSyncStructGet(BLUEVELVETC_HANDLE pHandle, blue_video_sync_struct* pIrqInfo, unsigned int &VideoMsc, unsigned int &SubfieldInterrupt)
{
	VideoMsc = pIrqInfo->video_msc;
	SubfieldInterrupt = pIrqInfo->subfield_interrupt;
}

#endif // defined UUID_94639C46_79BD_11E2_9DBA_F0DEF1A0ACC9

