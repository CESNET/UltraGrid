/*
 * FILE:     auddev_maxosx.c
 * PROGRAM:  RAT 4
 * AUTHOR:   Juraj Sucik <juraj.sucik@cnl.tuke.sk>
 * This code was inspired by audio implementation in VAT from OpenMash.
 */

#include "config.h"
#include "config_unix.h"
#include "debug.h"
#include "memory.h"
#include "audio_types.h"
#include "audio_fmt.h"
#include "audio_hw/macosx.h"
#include "decim_3528kHz_8kHz.h"

#include <AudioToolbox/DefaultAudioOutput.h>
#include <AudioToolbox/AudioConverter.h>
#include <CoreAudio/CoreAudio.h>

struct device
{
	char *name;							// The input device name
	AudioUnit outputUnit_;						// The output audio unit
	AudioStreamBasicDescription inputStreamBasicDescription_;	// The input stream description
	AudioStreamBasicDescription mashStreamBasicDescription_;	// The Mash stream description
	AudioUnitInputCallback input;					// callback function to privde output data
	Float32 inputGain_, outputGain_;				// Input and output volume gain values. 
	audio_format suported_formats[14];
	int num_supported_format;
	AudioDeviceID inputDeviceID_;					// The input device ID.
};

// used for testing
//static float kilo[] = { 0, 0.70710678118655, 1.00000000000000, 0.70710678118655, 0.00000000000000,  -0.70710678118655,  -1.00000000000000, -0.70710678118655, };

enum { AUDIO_IN, AUDIO_OUT};

static audio_port_details_t iport = { AUDIO_IN, "audio_in"};
static audio_port_details_t oport = { AUDIO_OUT, "audio_out"};

static AudioStreamBasicDescription streamdesc_;	// The Mash stream description

static struct device devices[3];

//// The read and write function buffers.
//
// Reading and writing from the buffers is done by two separate threads.
static const int ringBufferFactor_ = 32;
static int readBufferSize_, writeBufferSize_;
static u_char* readBuffer_;
static int inputReadIndex_, inputWriteIndex_;
static SInt16* writeBuffer_;
static int outputReadIndex_, outputWriteIndex_;
static int availableInput_;
// The decimation Z delay line and the current FIR filter phase.
double* zLine_;
int currentPhase_;
//static u_int blksize_ = 160;
//u_int blksize_ = 80;
Boolean input_, output_;
Boolean muteInput_, muteOutput_;
Boolean obtained_;
static int add = 0;
// used for testing
//static int k=0;

// Iowegian's dspGuru resampling function.
static void 
resamp(int interp_factor_L, int decim_factor_M, int num_taps_per_phase,
       int *p_current_phase, const double *const p_H,
       double *const p_Z, int num_inp, const double *p_inp,
       double *p_out, int *p_num_out)
{
    int tap, num_out, num_new_samples, phase_num = *p_current_phase;
    const double *p_coeff;
    double sum;

    num_out = 0;
    while (num_inp > 0) {

        /* figure out how many new samples to shift into Z delay line */
        num_new_samples = 0;
        while (phase_num >= interp_factor_L) {
            /* decrease phase number by interpolation factor L */
            phase_num -= interp_factor_L;
            num_new_samples++;
            if (--num_inp == 0) {
                break;
            }
        }

        if (num_new_samples >= num_taps_per_phase) {
            /* the new samples are bigger than the size of Z:
            fill the entire Z with the tail of new inputs */
            p_inp += (num_new_samples - num_taps_per_phase);
            num_new_samples = num_taps_per_phase;
        }

        /* copy new samples into Z */

        /* shift Z delay line up to make room for next samples */
        for (tap = num_taps_per_phase - 1; tap >= num_new_samples; tap--) {
            p_Z[tap] = p_Z[tap - num_new_samples];
        }

        /* copy next samples from input buffer to bottom of Z */
        for (tap = num_new_samples - 1; tap >= 0; tap--) {
            p_Z[tap] = *p_inp++;
        }

        /* calculate outputs */
        while (phase_num < interp_factor_L) {
            /* point to the current polyphase filter */
            p_coeff = p_H + phase_num;

            /* calculate FIR sum */
            sum = 0.0;
            for (tap = 0; tap < num_taps_per_phase; tap++) {
                sum += *p_coeff * p_Z[tap];
                p_coeff += interp_factor_L;   /* point to next coefficient */
            }
            *p_out++ = sum;     /* store sum and point to next output */
            num_out++;

            /* decrease phase number by decimation factor M */
            phase_num += decim_factor_M;
        }
    }

    /* pass back to caller phase number (for next call) and number of
        outputs */
    *p_current_phase = phase_num;
    *p_num_out = num_out;
}
static OSStatus 
audioIOProc(AudioDeviceID inDevice, const AudioTimeStamp* inNow, 
            const AudioBufferList* inInputData, const AudioTimeStamp* inInputTime, 
	    AudioBufferList* outOutputData, const AudioTimeStamp* inOutputTime, 
	    void* inClientData)
{
    UNUSED(inDevice);
    UNUSED(inNow);
    UNUSED(inInputTime);
    UNUSED(outOutputData);
    UNUSED(inOutputTime);
    UNUSED(inClientData);

    //If input is provided, copy it into the read buffer.
    if (inInputData != NULL && inInputData->mNumberBuffers > 0) {
        // Get the input data.
        Float32* ip = (Float32*)inInputData->mBuffers[0].mData;
	int providedFrames = inInputData->mBuffers[0].mDataByteSize / devices[add].inputStreamBasicDescription_.mBytesPerFrame;
	//debug_msg("providedFrames  -%d\n",providedFrames );
	//debug_msg("inInputData->mBuffers[0].mDataByteSize-%d\n",inInputData->mBuffers[0].mDataByteSize);
	//printf("read mSampleRate: %f\n",devices[add].inputStreamBasicDescription_.mSampleRate);

	//struct device * dev = (struct device *) inClientData;
	//debug_msg("dev->inputStreamBasicDescription_.mBytesPerFrame-%d\n", devices[add].inputStreamBasicDescription_.mBytesPerFrame);
        // Convert the audio to mono.
        double* monoInput = malloc(sizeof(double)*providedFrames);
	int sample;
        for (sample = 0; sample < providedFrames; sample++)
	 {
        	monoInput[sample] = *ip;
           	ip += inInputData->mBuffers[0].mNumberChannels;
        };
        // Resample the raw data.
        double* monoOutput = malloc(sizeof(double)*providedFrames);
        int numOutput = 0;
        int factorL = 80, factorM = 441;
        resamp(factorL, factorM, DECIM441_LENGTH / factorL, &(currentPhase_), decim441, zLine_, providedFrames, monoInput, monoOutput, &numOutput);
        // Convert the output to mulaw and store it in the read buffer.
        for (sample = 0; sample < numOutput; sample++) {
         //   if (muteInput_) {
          //      readBuffer_[inputWriteIndex_++] = lintomulaw[0];
                //readBuffer_[inputWriteIndex_++] = 0;
           // } else {
			//debug_msg("audioIOProc1\n");
			// Apply the audio gain.
			monoOutput[sample] *= devices[add].inputGain_;
			// Clip the audio data.
            		if (monoOutput[sample] > 1.0) monoOutput[sample] = 1.0;
            		else if (monoOutput[sample] < -1.0) monoOutput[sample] = -1.0;
			// Convert to signed 16-bit.
			// poslem 1 kHz ton
			//monoOutput[sample] = kilo[k++];
			//if (k == 8) k = 0;

			//SInt16 si = (monoOutput[sample] >= 0) ? (SInt16)(monoOutput[sample] * 32767.0) : (SInt16)(monoOutput[sample] * 32768.0);
			SInt16 si = 32767*monoOutput[sample];

			//printf("%d\n", si);
			// Convert to 8-bit mulaw and store in the read buffer.
			//readBuffer_[inputWriteIndex_++] = lintomulaw[si & 0xFFFF];
			//readBuffer_[inputWriteIndex_++] = si;
			//rb = readBuffer_ + inputWriteIndex_;
			
			memcpy(readBuffer_ + inputWriteIndex_, &si, 2);
			inputWriteIndex_ += 2;
	   // }
	    		if (inputWriteIndex_ == readBufferSize_) inputWriteIndex_ = 0;
	    //availableInput_++;
	    		availableInput_ += 2;
	}
	//printf("inputWriteIndex_: %d\n", inputWriteIndex_);
	// Clean up.
	free(monoInput); monoInput = NULL;
	free(monoOutput); monoOutput = NULL;
     };

	// Return success.*/
    return noErr;
};

static OSStatus 
outputRenderer(void *inRefCon, AudioUnitRenderActionFlags inActionFlags, const AudioTimeStamp *inTimeStamp, 
               UInt32 inBusNumber, AudioBuffer *ioData) 
{
	int i;
	SInt16* ip = (SInt16*)ioData->mData;

	UNUSED(inRefCon);
	UNUSED(inActionFlags);
	UNUSED(inTimeStamp);
	UNUSED(inBusNumber);

	int requestedFrames = ioData->mDataByteSize / devices[add].mashStreamBasicDescription_.mBytesPerFrame;
       	//int requestedFrames = ioData->mDataByteSize / 2;
	int zeroIndex = outputReadIndex_ - 1;

	//printf("req frames: %d\tDataByteSize %d\ndev: %d\n",requestedFrames, ioData->mDataByteSize, devices[add].mashStreamBasicDescription_.mBytesPerFrame);
	//printf("outputRender:\n");
	for (i = 0; i < requestedFrames; i++)
	{
		// Copy the requested amount of data into the output buffer.
		//if (muteOutput_)
		//{	
			//ip[i] = 0;
			//outputReadIndex_++;
		//}
		//else
		//{
			//*ip++ = 10000;
			ip[i] = writeBuffer_[outputReadIndex_];
			//memcpy(ip + i, writeBuffer_ + outputReadIndex_, 2);
			outputReadIndex_++;

			//printf("%d\n", ip[i]);
			//printf("outputReadIndex: %d\n",outputReadIndex_);
			//printf("%d",*(ip + i));
		//};
	//	 Zero out the previous frames to avoid replaying the contents of the ring buffer.
		 writeBuffer_[zeroIndex--] = 0;
	//	 Wrap around the indices if necessary.
		if (zeroIndex == -1) zeroIndex = writeBufferSize_ - 1;
		if (outputReadIndex_ == writeBufferSize_) outputReadIndex_ = 0;
	};
	//printf("write data: \n");
	//for (i = 0; i < blksize_; i++)
	//printf("\n");	


    // Return success.
    return noErr;
};




int macosx_audio_init(void)/* Test and initialize audio interface */
{
	int num_devices;
	num_devices = macosx_audio_device_count();
	if (num_devices < 1) {
		return 0;
	}
	return 1; 
};

int  macosx_audio_open(audio_desc_t ad, audio_format* ifmt, audio_format *ofmt)
{
	//return 0;
	OSStatus err = noErr;
	UInt32   propertySize;
	Boolean  writable;
	obtained_ = false;
	add = ad;
	//dev[0] = devices[ad];
	UNUSED(ofmt);

	// Get the default input device ID. 
	err = AudioHardwareGetPropertyInfo(kAudioHardwarePropertyDefaultInputDevice, &propertySize, &writable);              
	if (err != noErr)
	{
		return 0;
	};
	err = AudioHardwareGetProperty(kAudioHardwarePropertyDefaultInputDevice, &propertySize, &(devices[ad].inputDeviceID_));
	if (err != noErr)
	{
		debug_msg("error kAudioHardwarePropertyDefaultInputDevice");
		return 0;
	};
	if (devices[ad].inputDeviceID_ == kAudioDeviceUnknown) {
		debug_msg("error kAudioDeviceUnknown");
		return 0;
	}
	// Get the input stream description.
	err = AudioDeviceGetPropertyInfo(devices[ad].inputDeviceID_, 0, true, kAudioDevicePropertyStreamFormat, &propertySize, &writable);
	if (err != noErr)
	{
		debug_msg("error AudioDeviceGetPropertyInfo");
		return 0;
	};
	err = AudioDeviceGetProperty(devices[ad].inputDeviceID_, 0, true, kAudioDevicePropertyStreamFormat, &propertySize, &(devices[ad].inputStreamBasicDescription_));
	//printf("inputStreamBasicDescription_.mBytesPerFrame %d\n", devices[add].inputStreamBasicDescription_);
	if (err != noErr)
	{
		debug_msg("error AudioDeviceGetProperty");
		return 0;
	};

	// nastavime maly endian
	devices[ad].inputStreamBasicDescription_.mFormatFlags &= (kAudioFormatFlagIsBigEndian & 0);

	if (writable) {
	        err = AudioDeviceSetProperty(devices[ad].inputDeviceID_, NULL, 0, true, kAudioDevicePropertyStreamFormat, sizeof(AudioStreamBasicDescription), &(devices[ad].inputStreamBasicDescription_));
	        if (err != noErr) printf("err: AudioDeviceSetProperty: kAudioDevicePropertyStreamFormat\n");
	}
	
	/* set the buffer size of the device */
	
	/*
	int bufferByteSize = 8192;
	propertySize = sizeof(bufferByteSize);
	err = AudioDeviceSetProperty(devices[ad].inputDeviceID_, NULL, 0, true, kAudioDevicePropertyBufferSize, propertySize, &bufferByteSize);
	if (err != noErr) debug_msg("err: Set kAudioDevicePropertyBufferSize to %d\n", bufferByteSize);
	else debug_msg("sucessfully set kAudioDevicePropertyBufferSize to %d\n", bufferByteSize);
	*/
	
	// Register the AudioDeviceIOProc.
	err = AudioDeviceAddIOProc(devices[ad].inputDeviceID_, audioIOProc,NULL);
	if (err != noErr)
	{
		debug_msg("error AudioDeviceAddIOProc");
		return 0;
	};
	err = OpenDefaultAudioOutput(&(devices[ad].outputUnit_));
	if (err != noErr)
	{
		debug_msg("error OpenDefaultAudioOutput");
		return 0;
	};
	err = AudioUnitInitialize(devices[ad].outputUnit_);
	if (err != noErr)
	{
		debug_msg("error AudioUnitInitialize");
		return 0;
	};
	// Register a callback function to provide output data to the unit.
	devices[ad].input.inputProc = outputRenderer;
	devices[ad].input.inputProcRefCon = 0;
	err = AudioUnitSetProperty(devices[ad].outputUnit_, kAudioUnitProperty_SetInputCallback, kAudioUnitScope_Global, 0, &(devices[ad].input), sizeof(devices[ad].input));

	if (err != noErr)
	{
		debug_msg("error AudioUnitSetProperty1");
		return 0;
	};
	// Define the Mash stream description. Mash puts 20ms of data into each read
	// and write call. 20ms at 8000Hz equals 160 samples. Each sample is a u_char,
	// so that's 160 bytes. Mash uses 8-bit mu-law internally, so we need to convert
	// to 16-bit linear before using the audio data.
	devices[ad].mashStreamBasicDescription_.mSampleRate = 8000.0;
	devices[ad].mashStreamBasicDescription_.mFormatID = kAudioFormatLinearPCM;
	devices[ad].mashStreamBasicDescription_.mFormatFlags =kLinearPCMFormatFlagIsSignedInteger | kLinearPCMFormatFlagIsBigEndian |kLinearPCMFormatFlagIsPacked;
	devices[ad].mashStreamBasicDescription_.mBytesPerPacket = 2;
	devices[ad].mashStreamBasicDescription_.mFramesPerPacket = 1;
	devices[ad].mashStreamBasicDescription_.mBytesPerFrame = 2;
	devices[ad].mashStreamBasicDescription_.mChannelsPerFrame = 1;
	devices[ad].mashStreamBasicDescription_.mBitsPerChannel = 16;

	// Inform the default output unit of our source format.
	err = AudioUnitSetProperty(devices[ad].outputUnit_, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Input, 0, &(devices[ad].mashStreamBasicDescription_), sizeof(AudioStreamBasicDescription));
	if (err != noErr)
	{
		debug_msg("error AudioUnitSetProperty2");
		printf("error setting output unit source format\n");
		return 0;
	};

	// check the stream format
	err = AudioUnitGetPropertyInfo(devices[ad].outputUnit_, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Input, 0, &propertySize, &writable);
	if (err != noErr) debug_msg("err getting propert info for kAudioUnitProperty_StreamFormat\n");

	err = AudioUnitGetProperty(devices[ad].outputUnit_, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Input, 0, &streamdesc_, &propertySize);
	if (err != noErr) debug_msg("err getting values for kAudioUnitProperty_StreamFormat\n");
	
	char name[128];
	audio_format_name(ifmt, name, 128);
	debug_msg("Requested ifmt %s\n",name);
	debug_msg("ifmt bytes pre block: %d\n",ifmt->bytes_per_block);

	// handle the requested format
	if (ifmt->encoding != DEV_S16) {
		audio_format_change_encoding(ifmt, DEV_S16);
		debug_msg("Requested ifmt changed to %s\n",name);
		debug_msg("ifmt bytes pre block: %d\n",ifmt->bytes_per_block);
	}

	audio_format_name(ofmt, name, 128);
	debug_msg("Requested ofmt %s\n",name);
	debug_msg("ofmt bytes pre block: %d\n",ofmt->bytes_per_block);
	
	// Allocate the read buffer and Z delay line.
	//readBufferSize_ = 8192;
	readBufferSize_ = ifmt->bytes_per_block * ringBufferFactor_;
	//readBufferSize_ = 320;
	//printf("readBufferSize_ %d\n", readBufferSize_);
	readBuffer_ = malloc(sizeof(u_char)*readBufferSize_);
	bzero(readBuffer_, readBufferSize_ * sizeof(u_char));
	//memset(readBuffer_, PCMU_AUDIO_ZERO, readBufferSize_);
	//inputReadIndex_ = -1; 
	inputReadIndex_ = 0; inputWriteIndex_ = 0;
	zLine_ = malloc(sizeof(double)*DECIM441_LENGTH / 80);
	availableInput_ = 0;

	// Allocate the write buffer.
	//writeBufferSize_ = 8000;
	writeBufferSize_ = ofmt->bytes_per_block * ringBufferFactor_;
	writeBuffer_ = malloc(sizeof(SInt16)*writeBufferSize_);
	bzero(writeBuffer_, writeBufferSize_ * sizeof(SInt16));
	outputReadIndex_ = 0; outputWriteIndex_ = 0;
	//outputWriteIndex_ = -1;
    	// Start audio processing.
	err = AudioDeviceStart(devices[ad].inputDeviceID_, audioIOProc);
	if (err != noErr) {
		fprintf(stderr, "Input device error: AudioDeviceStart\n");
		return 0;
	}
	err = AudioOutputUnitStart(devices[ad].outputUnit_);
	if (err != noErr) {
		fprintf(stderr, "Output device error: AudioOutputUnitStart\n");
		return 0;
	}
	// Inform the default output unit of our source format.
	/*
	err = AudioUnitSetProperty(devices[ad].outputUnit_, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Input, 0, &(devices[ad].mashStreamBasicDescription_), sizeof(AudioStreamBasicDescription));
	if (err != noErr)
	{
		debug_msg("error AudioUnitSetProperty3");
		return 0;
	};
	*/
	return 1;
};

void macosx_audio_close(audio_desc_t ad)
{
	OSStatus err = noErr;
	// Stop the audio devices.
	err = AudioDeviceStop(devices[ad].inputDeviceID_, audioIOProc);
	if (err != noErr) fprintf(stderr, "Input device error: AudioDeviceStop\n");
	err = AudioOutputUnitStop(devices[ad].outputUnit_);
	if (err != noErr) fprintf(stderr, "Output device error: AudioOutputUnitStop\n");
	// Unregister the AudioDeviceIOProc.
	err = AudioDeviceRemoveIOProc(devices[ad].inputDeviceID_,audioIOProc);
	if (err != noErr) fprintf(stderr, "Input device error: AudioDeviceRemoveIOProc\n");
	CloseComponent(devices[ad].outputUnit_);
	if (readBuffer_ != NULL) free(readBuffer_); readBuffer_ = NULL;
	if (writeBuffer_ != NULL) free(writeBuffer_); writeBuffer_ = NULL;
	if (zLine_ != NULL) free(zLine_); zLine_ = NULL;
};

void macosx_audio_drain(audio_desc_t ad)
{
	UNUSED(ad);
	macosx_audio_read(ad, NULL, 10000000);
};

int  macosx_audio_duplex(audio_desc_t ad)
{
	UNUSED(ad);
	return 1;
}

void macosx_audio_set_igain(audio_desc_t ad,int level)
{
	UNUSED(ad);
	// Remember the record gain.
	//devices[ad].rgain_ = level;
	devices[ad].inputGain_ = level; /// inputGainDivisor_;
}

void macosx_audio_set_ogain(audio_desc_t ad,int vol)
{
	UNUSED(ad);
	OSStatus err = noErr;
	// Remember the playback gain.
	//devices[ad].pgain_ = level;
	devices[ad].outputGain_ = vol; /// outputGainDivisor_;
	//float og_scaled = devices[ad].outputGain_ / 16;

	// Set the volume.
	err = AudioUnitSetParameter(devices[ad].outputUnit_, kAudioUnitParameterUnit_LinearGain, kAudioUnitScope_Global, 0, devices[ad].outputGain_, 0);
	if (err != noErr) fprintf(stderr, "Input device error: set_ogain");
};

int  macosx_audio_get_igain(audio_desc_t ad)
{
	UNUSED(ad);
	return devices[ad].inputGain_;
};

int  macosx_audio_get_ogain(audio_desc_t ad)
{
	UNUSED(ad);
	return devices[ad].outputGain_;
}

void macosx_audio_loopback(audio_desc_t ad, int gain)
{
	UNUSED(ad);
	UNUSED(gain);
};

int  macosx_audio_read(audio_desc_t ad, u_char *buf, int read_bytes)
{
	UNUSED(ad);
	int done = 0;
	int this_read = 0;
//	SInt16 i;
	
//        debug_msg("macosx_audio_read\n");
	//printf("audio_read read_bytes: %d\n",read_bytes);	
	// Initialize the read index if this is the first time Read has been called.
	//if (inputReadIndex_ == -1) inputReadIndex_ = 0;
	// Point to the next blksize_ of data in the read buffer.
	//for (i = 0; i < availableInput_; i++) {
	//printf("inputReadIndex_ pred: %d\n", inputReadIndex_);
	//printf("availableInput: %d, requested read_bytes: %d, ", availableInput_, read_bytes);
	if (availableInput_ > 0 && done < read_bytes) {
		// kolko budeme citat
		//if (read_bytes > 320) read_bytes = 320;
		this_read = min(availableInput_, read_bytes - done);
		if (inputReadIndex_ + this_read > readBufferSize_)
			this_read = readBufferSize_ - inputReadIndex_;
		// kopirovanie
		memcpy(buf + done, readBuffer_ + inputReadIndex_, this_read);
		//memset(readBuffer_ + inputReadIndex_, 0, this_read);
		// upravime indexy
		done += this_read;
		availableInput_ -= this_read;
		inputReadIndex_ += this_read;

		//printf("read_bytes: %d\t this_read: %d\t done: %d\t ava: %d\tiidx: %d\tsize:%d\n",read_bytes, this_read, done, availableInput_, inputReadIndex_, readBufferSize_);
		//memcpy(buf + len, readBuffer_ + inputReadIndex_, 2);
		//buf[len] = readBuffer_[inputReadIndex_++];
		//len ++;
		//buf[len] = readBuffer_[inputReadIndex_++];
		//readBuffer_[inputReadIndex_++] = PCMU_AUDIO_ZERO;
		//readBuffer_[inputReadIndex_] = L16_AUDIO_ZERO;

		//buf[len] = (u_char) 127*kilo[k];
		//i = 32676*kilo[k];
		
		//memcpy(buf + len, &i, 2);
		//k++;
		//if (k == 8) k = 0;
		
		//inputReadIndex_ += 2;
		//printf("%d\n", buf);

		if (inputReadIndex_ == readBufferSize_) inputReadIndex_ = 0;
	}
	return done;
	//buf = readBuffer_ + inputReadIndex_;
	//inputReadIndex_ += blksize_;
	//availableInput_ -= blksize_;
	// Wrap around the input read index if necessary.
	// Return the data pointer.
	//return rp;
};

int macosx_audio_write(audio_desc_t ad, u_char* data, int write_bytes)
{
	UNUSED(ad);
	// Don't do anything if output is not supported.
	//return write_bytes;

/*
	int i = 0;
	//SInt16 j = 0;

    // Initialize the write index if this is the first time Write has been called.
    //if (outputWriteIndex_ == -1) outputWriteIndex_ = 0;
    //printf("write_bytes: %d\nwriteBufferSize_: %d\n", write_bytes, writeBufferSize_); 
    // Convert the 8-bit mulaw audio to 16-bit linear, only if necessary.
	//printf("audio_write data: %d\n", write_bytes);
	for (i = 0; i < write_bytes; i += 2)
	{
	    //writeBuffer_[outputWriteIndex_++] = data[i];
	//	printf("%d ", data[i]);
	    //writeBuffer_[outputWriteIndex_] = 32766*kilo[k++];
	    //j = 32766*kilo[k];
	    //k++;
	    memcpy(writeBuffer_ + outputWriteIndex_, data + i, 2);
	   // printf("%d\n", *(writeBuffer_ + outputWriteIndex_));

	    outputWriteIndex_ ++;
	    //if (k == 8) k = 0;
	    if (outputWriteIndex_ == writeBufferSize_) outputWriteIndex_ = 0;
	}
	*/
	register int first_write;
	register int second_write;
        first_write  = min((writeBufferSize_ - outputWriteIndex_) << 1, write_bytes);
        second_write = write_bytes - first_write;
        memcpy(writeBuffer_ + outputWriteIndex_, data, first_write);
        if (second_write) {
                memcpy(writeBuffer_, data + first_write, second_write);
                outputWriteIndex_ = second_write >> 1;
                return write_bytes;
        }
        outputWriteIndex_ += first_write >> 1;

	return write_bytes;
};

void macosx_audio_non_block(audio_desc_t ad)
{
    UNUSED(ad);
};

void macosx_audio_block(audio_desc_t ad)
{
	UNUSED(ad);
};

void macosx_audio_oport_set(audio_desc_t ad, audio_port_t port)
{
	UNUSED(ad);
	UNUSED(port);
};

audio_port_t macosx_audio_oport_get(audio_desc_t ad) 
{
	UNUSED(ad);
	return AUDIO_OUT;
};

int  macosx_audio_oport_count(audio_desc_t ad)
{
	UNUSED(ad);
	return 1;
};

const audio_port_details_t* macosx_audio_oport_details(audio_desc_t ad, int idx)
{
	UNUSED(ad);
	UNUSED(idx);
	return &oport;
};

void macosx_audio_iport_set(audio_desc_t ad, audio_port_t port)
{
	UNUSED(ad);
	UNUSED(port);
};

audio_port_t macosx_audio_iport_get(audio_desc_t ad)
{
	UNUSED(ad);
	return AUDIO_IN;
}

int  macosx_audio_iport_count(audio_desc_t ad)
{
	UNUSED(ad);
	return 1;
};

const audio_port_details_t* macosx_audio_iport_details(audio_desc_t ad, int idx)
{
	UNUSED(ad);
	UNUSED(idx);
	return &iport;
};

int  macosx_audio_is_ready(audio_desc_t ad)
{
	UNUSED(ad);
	return (availableInput_ != NULL);
}

void macosx_audio_wait_for(audio_desc_t ad, int delay_ms) {
	struct timeval tv;
	
	UNUSED(ad);

	tv.tv_sec = 0;
	tv.tv_usec = delay_ms * 1000;

	select(0, NULL, NULL, NULL, &tv);
}

int  macosx_audio_supports(audio_desc_t ad, audio_format *fmt)
{
	UNUSED(ad);
	if (fmt->encoding != DEV_S16) {
		return 0;
	}
	return 1;
}

int macosx_audio_device_count(void)
/* Then this one tells us the number of 'em */
{

    OSStatus err = noErr;
    UInt32 theSize;
    err = AudioHardwareGetPropertyInfo ( kAudioHardwarePropertyDevices, &theSize, NULL );
    int num_devices;
    num_devices = theSize / sizeof(AudioDeviceID);
    //return num_devices;
    return 1;
};

char *macosx_audio_device_name(audio_desc_t idx)	/* Then this one tells us the name          */
{
	UNUSED(idx);
    char *name;
    name = (char *) malloc(sizeof(char)*128);
    /*
    OSStatus theStatus;
    UInt32 theSize;
    theStatus = AudioHardwareGetPropertyInfo ( kAudioHardwarePropertyDevices, &theSize, NULL );
    int theNumberDevices;
    theNumberDevices = theSize / sizeof(AudioDeviceID);
    AudioDeviceID * theDeviceList, theDevice;
    theDeviceList = (AudioDeviceID*) malloc ( theNumberDevices * sizeof(AudioDeviceID) );
    theStatus = AudioHardwareGetProperty ( kAudioHardwarePropertyDevices, &theSize, theDeviceList ) ;
    theDevice = theDeviceList[0];
    UInt32 s = sizeof(char)*128;
    AudioDeviceGetProperty( theDevice, 0, 0, kAudioDevicePropertyDeviceName, &s, name);
    */
    name = "Default Sound Device";
    return name;
};

