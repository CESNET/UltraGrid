/*
 * FILE: audio_hw/win32.c
 *
 * Reworked by Orion Hodson from RAT 3.0 code.
 *
 * Assorted fixes and multilingual comments from Michael Wallbaum 
 * <wallbaum@informatik.rwth-aachen.de>
 *
 * Copyright (c)      2003 University of Southern California
 * Copyright (c) 1995-2001 University College London
 * All rights reserved.
 *
 * $Revision: 1.1 $
 * $Date: 2007/11/08 09:48:59 $
 */
 
#ifdef WIN32

#include "config_win32.h"
#include "audio.h"
#include "debug.h"
#include "memory.h"
#include "audio_hw/win32.h"
#include "audio_types.h"
#include "audio_fmt.h"
#include "util.h"
#include "mmsystem.h"

#define MAX_DEVICE_GAIN 0xffff
#define rat_to_device(x)	(((x) * MAX_DEVICE_GAIN / MAX_AMP) << 16 | ((x) * MAX_DEVICE_GAIN / MAX_AMP))
#define device_to_rat(x)	((x & 0xffff) * MAX_AMP / MAX_DEVICE_GAIN)

#define W32SDK_MAX_DEVICES 5
static  int have_probed[W32SDK_MAX_DEVICES];
static  int w32sdk_probe_formats(audio_desc_t ad);

static int  error = 0;
static char errorText[MAXERRORLENGTH];
static int  nLoopGain = 100;
#define     MAX_DEV_NAME 64

static UINT mapAudioDescToMixerID(audio_desc_t ad);

/* mcd_elem_t is a node used to store control state so 
 * we can restore mixer controls when device closes.
 */

typedef struct s_mcd_elem {
        MIXERCONTROLDETAILS *pmcd;
        struct s_mcd_elem   *next;
} mcd_elem_t;

static mcd_elem_t *control_list;

#define MIX_ERR_LEN 32
#define MIX_MAX_CTLS 8
#define MIX_MAX_GAIN 100

static int32_t	play_vol, rec_vol;
static HMIXER   hMixer;

static DWORD    dwRecLineID, dwVolLineID; 

static audio_port_details_t *input_ports, *loop_ports;
static int                   n_input_ports, n_loop_ports;
static int iport; /* Current input port */

/* Macro to convert macro name to string so we diagnose controls and error  */
/* codes.                                                                   */ 
#define CASE_STRING(x) case x: return #x

/* DEBUGGING FUNCTIONS ******************************************************/

static const char *
mixGetErrorText(MMRESULT mmr)
{
#ifndef NDEBUG
        switch (mmr) {
        CASE_STRING(MMSYSERR_NOERROR);     
        CASE_STRING(MIXERR_INVALLINE);     
        CASE_STRING(MIXERR_INVALCONTROL);  
        CASE_STRING(MIXERR_INVALVALUE);    
        CASE_STRING(WAVERR_BADFORMAT);     
        CASE_STRING(MMSYSERR_BADDEVICEID); 
        CASE_STRING(MMSYSERR_INVALFLAG);   
        CASE_STRING(MMSYSERR_INVALHANDLE); 
        CASE_STRING(MMSYSERR_INVALPARAM);  
        CASE_STRING(MMSYSERR_NODRIVER);
        default:
                return "Undefined Error";
        }
#endif /* NDEBUG */
        return "Mixer Error.";
}

static const char *
mixGetControlType(DWORD dwCtlType)
{
#ifndef NDEBUG
        switch(dwCtlType) {
        CASE_STRING(MIXERCONTROL_CONTROLTYPE_CUSTOM);        
        CASE_STRING(MIXERCONTROL_CONTROLTYPE_BOOLEANMETER);  
        CASE_STRING(MIXERCONTROL_CONTROLTYPE_SIGNEDMETER);   
        CASE_STRING(MIXERCONTROL_CONTROLTYPE_PEAKMETER);     
        CASE_STRING(MIXERCONTROL_CONTROLTYPE_UNSIGNEDMETER); 
        CASE_STRING(MIXERCONTROL_CONTROLTYPE_BOOLEAN);       
        CASE_STRING(MIXERCONTROL_CONTROLTYPE_ONOFF);         
        CASE_STRING(MIXERCONTROL_CONTROLTYPE_MUTE);          
        CASE_STRING(MIXERCONTROL_CONTROLTYPE_MONO);          
        CASE_STRING(MIXERCONTROL_CONTROLTYPE_LOUDNESS);      
        CASE_STRING(MIXERCONTROL_CONTROLTYPE_STEREOENH);     
        CASE_STRING(MIXERCONTROL_CONTROLTYPE_BUTTON);        
        CASE_STRING(MIXERCONTROL_CONTROLTYPE_DECIBELS);      
        CASE_STRING(MIXERCONTROL_CONTROLTYPE_SIGNED);        
        CASE_STRING(MIXERCONTROL_CONTROLTYPE_UNSIGNED);      
        CASE_STRING(MIXERCONTROL_CONTROLTYPE_PERCENT);       
        CASE_STRING(MIXERCONTROL_CONTROLTYPE_SLIDER);        
        CASE_STRING(MIXERCONTROL_CONTROLTYPE_PAN);           
        CASE_STRING(MIXERCONTROL_CONTROLTYPE_QSOUNDPAN);     
        CASE_STRING(MIXERCONTROL_CONTROLTYPE_FADER);         
        CASE_STRING(MIXERCONTROL_CONTROLTYPE_VOLUME);        
        CASE_STRING(MIXERCONTROL_CONTROLTYPE_BASS);          
        CASE_STRING(MIXERCONTROL_CONTROLTYPE_TREBLE);        
        CASE_STRING(MIXERCONTROL_CONTROLTYPE_EQUALIZER);     
        CASE_STRING(MIXERCONTROL_CONTROLTYPE_SINGLESELECT);  
        CASE_STRING(MIXERCONTROL_CONTROLTYPE_MUX);           
        CASE_STRING(MIXERCONTROL_CONTROLTYPE_MULTIPLESELECT);
        CASE_STRING(MIXERCONTROL_CONTROLTYPE_MIXER);         
        CASE_STRING(MIXERCONTROL_CONTROLTYPE_MICROTIME);     
        CASE_STRING(MIXERCONTROL_CONTROLTYPE_MILLITIME);    
        }
#endif /* NDEBUG */
        return "Unknown";
}

static void
mixerDumpLineInfo(HMIXEROBJ hMix, DWORD dwLineID)
{
        MIXERLINECONTROLS mlc;
        LPMIXERCONTROL    pmc;
        MIXERLINE ml;
        MMRESULT  mmr;
        UINT      i;
        
        /* Determine number of controls */
        ml.cbStruct = sizeof(ml);
        ml.dwLineID = dwLineID;
        
        mmr = mixerGetLineInfo((HMIXEROBJ)hMix, &ml, MIXER_GETLINEINFOF_LINEID | MIXER_OBJECTF_HMIXER);
        if (mmr != MMSYSERR_NOERROR) {
                debug_msg(mixGetErrorText(mmr));
                return;
        }
        
        pmc = (LPMIXERCONTROL)xmalloc(sizeof(MIXERCONTROL)*ml.cControls);
        mlc.cbStruct  = sizeof(MIXERLINECONTROLS);
        mlc.cbmxctrl  = sizeof(MIXERCONTROL);
        mlc.pamxctrl  = pmc;
        mlc.cControls = ml.cControls;
        mlc.dwLineID  = dwLineID;
        
        mmr = mixerGetLineControls((HMIXEROBJ)hMix, &mlc, MIXER_GETLINECONTROLSF_ALL | MIXER_OBJECTF_HMIXER);
        if (mmr != MMSYSERR_NOERROR) {
                debug_msg(mixGetErrorText(mmr));
                xfree(pmc);
                return;
        }
        
        for(i = 0; i < ml.cControls; i++) {
                debug_msg("- %u %s\t\t %s\n", i, pmc[i].szName, mixGetControlType(pmc[i].dwControlType));
        }
        xfree(pmc);
}

/* Code for saving control states when claiming device, so we can restore the 
 * config when we release the device.  Lots of request for this 
 */

int
mcd_elem_add_control(mcd_elem_t **pplist, MIXERCONTROLDETAILS *pmcd)
{
        mcd_elem_t *elem;
        
        elem = (mcd_elem_t*)xmalloc(sizeof(mcd_elem_t));
        if (elem) {
                elem->pmcd = pmcd;
                elem->next = *pplist;
                *pplist    = elem;
                return TRUE;
        }
        return FALSE;
}

MIXERCONTROLDETAILS*
mcd_elem_get_control(mcd_elem_t **pplist)
{
        MIXERCONTROLDETAILS *pmcd;
        mcd_elem_t *elem;
        
        elem = *pplist;
        if (elem) {
                pmcd    = elem->pmcd;
                *pplist = elem->next;
                xfree(elem);
                return pmcd;
        }
        return NULL;
}

void
mixRestoreControls(UINT uMix, mcd_elem_t **pplist)
{
        MIXERCONTROLDETAILS *pmcd;
        MMRESULT mmr;

        return;

        while((pmcd = mcd_elem_get_control(pplist)) != NULL) {
                mmr = mixerSetControlDetails((HMIXEROBJ)uMix, pmcd, MIXER_OBJECTF_MIXER);
                xfree(pmcd->paDetails);
                xfree(pmcd);
                if (mmr != MMSYSERR_NOERROR) {
                        debug_msg("mixerSetControlDetails: %s\n", mixGetErrorText(mmr));
                        continue;
                }
        }
        assert(*pplist == NULL);
}

void
mixSaveLine(UINT uMix, MIXERLINE *pml, mcd_elem_t **pplist)
{
        MIXERCONTROLDETAILS *pmcd;
        MIXERLINECONTROLS mlc;
        MIXERCONTROL     *pmc;
        MMRESULT          mmr;
        UINT              i;
        
        /* Retrieve control types */
        pmc = (MIXERCONTROL*)xmalloc(sizeof(MIXERCONTROL)*pml->cControls);
        
        mlc.cbStruct  = sizeof(mlc);
        mlc.dwLineID  = pml->dwLineID;
        mlc.cControls = pml->cControls;
        mlc.pamxctrl  = pmc;
        mlc.cbmxctrl  = sizeof(MIXERCONTROL);
        
        debug_msg("Saving %s\n", pml->szName);

        mmr = mixerGetLineControls((HMIXEROBJ)uMix, &mlc, MIXER_GETLINECONTROLSF_ALL | MIXER_OBJECTF_MIXER);
        if (mmr != MMSYSERR_NOERROR) {
                debug_msg("mixerGetLineControls: %s\n", mixGetErrorText(mmr));
                xfree(pmc);
                return;
        }
        
        for(i = 0; i < pml->cControls; i++) {
                DWORD itemCnt, itemSz;
                if (pmc[i].cMultipleItems == 0) {
                        itemCnt = 1;	
                } else {
                        itemCnt = pmc[i].cMultipleItems;
                } 
                
                switch(pmc[i].dwControlType & MIXERCONTROL_CT_UNITS_MASK) {
                        /* Our application on affects boolean types (mute, on/off) and unsigned (vol) */
                case MIXERCONTROL_CT_UNITS_BOOLEAN:
                        itemSz = sizeof(MIXERCONTROLDETAILS_BOOLEAN);
                        break;
                case MIXERCONTROL_CT_UNITS_UNSIGNED:
                        itemSz = sizeof(MIXERCONTROLDETAILS_UNSIGNED);
                        break;
                default:
                        debug_msg("not done %s\n", pmc[i].szName);
                        continue;
                }
                pmcd = (MIXERCONTROLDETAILS*)xmalloc(sizeof(MIXERCONTROLDETAILS));
                pmcd->cbStruct       = sizeof(MIXERCONTROLDETAILS);
                pmcd->cMultipleItems = pmc[i].cMultipleItems;
                pmcd->dwControlID    = pmc[i].dwControlID;
                pmcd->cChannels      = 1;
                pmcd->paDetails      = (void*)xmalloc(itemSz * itemCnt);
                pmcd->cbDetails      = itemSz;
                
                mmr = mixerGetControlDetails((HMIXEROBJ)uMix, pmcd, MIXER_GETCONTROLDETAILSF_VALUE | MIXER_OBJECTF_MIXER);
                if (mmr != MMSYSERR_NOERROR) {
                        debug_msg("mixerGetControlDetails: %s\n", mixGetErrorText(mmr));
                        continue;
                }
                mcd_elem_add_control(pplist, pmcd);
        }
        xfree(pmc);
}


void
mixSaveControls(UINT uMix, mcd_elem_t **pplist)
{
        MIXERLINE ml, sml;
        MIXERCAPS mc;
        MMRESULT  mmr;
        UINT i,j;
        
        mmr = mixerGetDevCaps(uMix, &mc, sizeof(mc));
        if (mmr != MMSYSERR_NOERROR) {
                debug_msg("mixerGetDevCaps: %s\n", mixGetErrorText(mmr));
                return;
        }
        
        for(i = 0; i < mc.cDestinations; i++) {
                memset(&ml, 0, sizeof(ml));
                ml.cbStruct      = sizeof(ml);
                ml.dwDestination = i;
                mmr = mixerGetLineInfo((HMIXEROBJ)uMix, &ml, MIXER_OBJECTF_MIXER | MIXER_GETLINEINFOF_DESTINATION); 
                if (mmr != MMSYSERR_NOERROR) {
                        debug_msg("mixerGetLineInfo: %s\n", mixGetErrorText(mmr));
                        continue;
                }
                mixSaveLine(uMix, &ml, pplist);
                for (j = 0; j < ml.cConnections; j++) {
                        memset(&sml, 0, sizeof(sml));
                        sml.cbStruct = sizeof(sml);
                        sml.dwSource = j;
                        mmr = mixerGetLineInfo((HMIXEROBJ)uMix, &sml, MIXER_OBJECTF_MIXER | MIXER_GETLINEINFOF_SOURCE); 
                        if (mmr != MMSYSERR_NOERROR) {
                                debug_msg("mixerGetLineInfo: %s\n", mixGetErrorText(mmr));
                                continue;
                        }
                        mixSaveLine(uMix, &sml, pplist);
                }
        }
}

/* CODE FOR CONTROLLING INPUT AND OUTPUT (LOOPBACK) LINES *******************
 * NOTE: the control of input lines and output lines is slightly different
 * because most card manufacturers put the volume and mute controls for output
 * as controls on the same output line.  The selection of the input lines is
 * controlled on the MUX control actually on the recording source, and the
 * volume control is on a line for the the input port.  To match the input 
 * select and the volume control we use the name of the line the volume
 * control is assigned to, and this ties in with the names on the MUX.  This
 * seems to be the only sensible way to correlate the two and it isn't in
 * the msdn library documentation.  I wasted a fair amount of time, trying
 * to match the name of the volume control and names in the MUX list, and
 * got this to work for all but one card.
 */

/* mixGetInputInfo - attempt to find corresponding wavein index
* for mixer uMix and corresponding destination line of mixer.  
* Returns TRUE if successful.
*/

int mixGetInputInfo(UINT uMix, UINT *puWavIn, DWORD *pdwLineID)
{
        UINT i, nWavIn;
        MIXERLINE  ml;
        MMRESULT   mmr;
        WAVEINCAPS wic;
        MIXERCAPS  mc;
        
        mmr = mixerGetDevCaps(uMix, &mc, sizeof(mc));
        if (mmr != MMSYSERR_NOERROR) {
                debug_msg("mixerGetDevCaps: %s\n", mixGetErrorText(mmr));
                return FALSE;
        }
        
        nWavIn = waveInGetNumDevs();
        for(i = 0; i < nWavIn; i++) {
                mmr = waveInGetDevCaps(i, &wic, sizeof(wic));
                if (mmr != MMSYSERR_NOERROR) {
                        debug_msg("waveInGetDevCaps: %s\n", mixGetErrorText(mmr));
                        continue;
                }
                
                ml.cbStruct       = sizeof(ml);
                ml.Target.dwType  = MIXERLINE_TARGETTYPE_WAVEIN;
                strncpy(ml.Target.szPname, wic.szPname, MAXPNAMELEN);
                ml.Target.vDriverVersion = wic.vDriverVersion;
                ml.Target.wMid    = wic.wMid;
                ml.Target.wPid    = wic.wPid;
                
                mmr = mixerGetLineInfo((HMIXEROBJ)uMix, &ml, MIXER_OBJECTF_MIXER | MIXER_GETLINEINFOF_TARGETTYPE);
                if (mmr == MMSYSERR_NOERROR) {
                        *puWavIn          = i;
                        *pdwLineID = ml.dwLineID;
                        debug_msg("Input: %s(%d - %d)\n", ml.szName, ml.dwDestination, ml.dwLineID);
                        return TRUE;
                } else {
                        debug_msg("mixerGetLineInfo (ignore this error): %s\n", mixGetErrorText(mmr));
                }
        }
        return FALSE;
}

/* mixGetOutputInfo - attempt to find corresponding waveout index
 * and corresponding destination line of mixer.  Returns TRUE if
 * successful.
 */
int 
mixGetOutputInfo(UINT uMix, UINT *puWavOut, DWORD *pdwLineID)
{
        UINT i, nWavOut;
        MIXERLINE  ml;
        MMRESULT   mmr;
        WAVEOUTCAPS woc;
        MIXERCAPS  mc;
        
        mmr = mixerGetDevCaps(uMix, &mc, sizeof(mc));
        if (mmr != MMSYSERR_NOERROR) {
                debug_msg("mixerGetDevCaps: %s\n", mixGetErrorText(mmr));
                return FALSE;
        }
        
        nWavOut = waveOutGetNumDevs();
        for(i = 0; i < nWavOut; i++) {
                mmr = waveOutGetDevCaps(i, &woc, sizeof(woc));
                if (mmr != MMSYSERR_NOERROR) {
                        debug_msg("waveOutGetDevCaps: %s\n", mixGetErrorText(mmr));
                        continue;
                }
                ml.cbStruct       = sizeof(ml);
                ml.Target.dwType  = MIXERLINE_TARGETTYPE_WAVEOUT;
                strncpy(ml.Target.szPname, woc.szPname, MAXPNAMELEN);
                ml.Target.vDriverVersion = woc.vDriverVersion;
                ml.Target.wMid    = woc.wMid;
                ml.Target.wPid    = woc.wPid;
                
                mmr = mixerGetLineInfo((HMIXEROBJ)uMix, &ml, MIXER_OBJECTF_MIXER | MIXER_GETLINEINFOF_TARGETTYPE);
                if (mmr == MMSYSERR_NOERROR) {
                        *puWavOut  = i;
                        *pdwLineID = ml.dwLineID;
                        debug_msg("Output: %s(%d - %d)\n", ml.szName, ml.dwDestination, ml.dwLineID);
                        return TRUE;
                } 
        }
        return FALSE;
}

/* mixerEnableInputLine - enables the input line whose name starts with beginning of portname.  
 * We cannot just use the port index like we do for volume because the mute controls are
 * not necessarily in the same order as the volume controls (grrr!).  The only card
 * that we have seen where this is necessary is the Winnov Videum AV, but there are
 * bound to be others.
 * Muting for input lines on the toplevel control (Rec, or whatever driver happens to call it).
 * It usually has a single control a MUX/Mixer that has "multiple items", one mute for
 * each input line.  Depending on the control type it may be legal to have multiple input
 * lines enabled, or just one.  So mixerEnableInputLine disables all lines other than
 * one selected.
 */

static int
mixerEnableInputLine(HMIXEROBJ hMix, char *portname)
{
        MIXERCONTROLDETAILS_BOOLEAN *mcdbState;
        MIXERCONTROLDETAILS_LISTTEXT *mcdlText;
        MIXERCONTROLDETAILS mcd;
        MIXERLINECONTROLS mlc;
        MIXERCONTROL mc;
        MIXERLINE ml;
        MMRESULT  mmr;
        UINT      i, matchLine;
        
        ml.cbStruct = sizeof(ml);
        ml.dwLineID = dwRecLineID;
        
        mmr = mixerGetLineInfo(hMix, &ml, MIXER_GETLINEINFOF_LINEID|MIXER_OBJECTF_HMIXER);
        if (mmr != MMSYSERR_NOERROR) {
                debug_msg("mixerGetLineInfo: %s\n", mixGetErrorText(mmr));
        }
        
        /* Get Mixer/MUX control information (need control id to set and get control details) */
        mlc.cbStruct      = sizeof(mlc);
        mlc.dwLineID      = ml.dwLineID;
        mlc.pamxctrl      = &mc;
        mlc.cbmxctrl      = sizeof(mc);
        
        mlc.dwControlType = MIXERCONTROL_CONTROLTYPE_MUX; /* Single Select */
        mmr = mixerGetLineControls(hMix, &mlc, MIXER_GETLINECONTROLSF_ONEBYTYPE|MIXER_OBJECTF_HMIXER);
        if (mmr != MMSYSERR_NOERROR) {
                mlc.dwControlType = MIXERCONTROL_CONTROLTYPE_MIXER; /* Multiple Select */
                mmr = mixerGetLineControls(hMix, &mlc, MIXER_GETLINECONTROLSF_ONEBYTYPE|MIXER_OBJECTF_HMIXER);
                if (mmr != MMSYSERR_NOERROR) {
                        debug_msg("mixerGetLineControls: %s\n", mixGetErrorText(mmr));
                        return FALSE;
                }
        }
        
        mcd.cbStruct    = sizeof(mcd);
        mcd.dwControlID = mc.dwControlID;
        mcd.cChannels   = 1;
        mcd.cMultipleItems = mc.cMultipleItems;
        mcdlText = (MIXERCONTROLDETAILS_LISTTEXT*)xmalloc(sizeof(MIXERCONTROLDETAILS_LISTTEXT)*mc.cMultipleItems);        
        mcd.paDetails = mcdlText;
        mcd.cbDetails = sizeof(MIXERCONTROLDETAILS_LISTTEXT);
        mmr = mixerGetControlDetails(hMix, &mcd, MIXER_GETCONTROLDETAILSF_LISTTEXT | MIXER_OBJECTF_MIXER);
        
        matchLine = 0;
        for(i = 0; i < mcd.cMultipleItems; i++) {
                if (!strcmp(mcdlText[i].szName, portname)) {
                        matchLine = i;
                        break;
                }
        }
        xfree(mcdlText);

        /* Now get control itself */
        mcd.cbStruct    = sizeof(mcd);
        mcd.dwControlID = mc.dwControlID;
        mcd.cChannels   = 1;
        mcd.cMultipleItems = mc.cMultipleItems;
        mcdbState = (MIXERCONTROLDETAILS_BOOLEAN*)xmalloc(sizeof(MIXERCONTROLDETAILS_BOOLEAN)*mc.cMultipleItems);        
        mcd.paDetails = mcdbState;
        mcd.cbDetails = sizeof(MIXERCONTROLDETAILS_BOOLEAN);
        
        mmr = mixerGetControlDetails(hMix, &mcd, MIXER_GETCONTROLDETAILSF_VALUE|MIXER_OBJECTF_MIXER);
        if (mmr != MMSYSERR_NOERROR) {
                debug_msg("mixerGetControlDetails: %s\n", mixGetErrorText(mmr));
                xfree(mcdbState);
                return FALSE;
        }
        
        for(i = 0; i < mcd.cMultipleItems; i++) {
                if (i == matchLine) {
                        mcdbState[i].fValue = TRUE;
                } else {
                        mcdbState[i].fValue = FALSE;
                }
        }
        
        mmr = mixerSetControlDetails(hMix, &mcd, MIXER_OBJECTF_MIXER);
        if (mmr != MMSYSERR_NOERROR) {
                debug_msg("mixerSetControlDetails: %s\n", mixGetErrorText(mmr));
                xfree(mcdbState);
                return FALSE;
        }
        
        xfree(mcdbState);
        return TRUE;
}

static int
mixerEnableOutputLine(HMIXEROBJ hMix, DWORD dwLineID, int state)
{
        MIXERCONTROLDETAILS_BOOLEAN mcdbState;
        MIXERCONTROLDETAILS mcd;
        MIXERLINECONTROLS mlc;
        MIXERCONTROL      mc;
        MMRESULT          mmr;
        
        mlc.cbStruct      = sizeof(mlc);
        mlc.pamxctrl      = &mc;
        mlc.cbmxctrl      = sizeof(MIXERCONTROL);
        mlc.dwLineID      = dwLineID;
        mlc.dwControlType = MIXERCONTROL_CONTROLTYPE_MUTE;
        
        mmr = mixerGetLineControls(hMix, &mlc, MIXER_GETLINECONTROLSF_ONEBYTYPE | MIXER_OBJECTF_HMIXER);
        if (mmr != MMSYSERR_NOERROR) {
                mlc.cbStruct      = sizeof(mlc);
                mlc.pamxctrl      = &mc;
                mlc.cbmxctrl      = sizeof(MIXERCONTROL);
                mlc.dwLineID      = dwLineID;
                mlc.dwControlType = MIXERCONTROL_CONTROLTYPE_ONOFF;
                mmr = mixerGetLineControls(hMix, &mlc, MIXER_GETLINECONTROLSF_ONEBYTYPE | MIXER_OBJECTF_HMIXER);
                if (mmr != MMSYSERR_NOERROR) {
                        debug_msg("Could not get mute control for line 0x%08x: %s\n", 
                                dwLineID,
                                mixGetErrorText(mmr));
                        mixerDumpLineInfo(hMix, dwLineID);
                        return FALSE;
                }
        }
        
        mcd.cbStruct       = sizeof(mcd);
        mcd.dwControlID    = mc.dwControlID;
        mcd.cChannels      = 1;
        mcd.cMultipleItems = mc.cMultipleItems;
        mcd.cbDetails      = sizeof(MIXERCONTROLDETAILS_BOOLEAN);
        mcd.paDetails      = &mcdbState;
        mcdbState.fValue   = !((UINT)state);
        
        mmr = mixerSetControlDetails((HMIXEROBJ)hMix, &mcd, MIXER_OBJECTF_HMIXER);
        if (mmr != MMSYSERR_NOERROR) {
                debug_msg("Could not set mute state for line 0x%08x\n", dwLineID);
                return FALSE;
        }
        return TRUE;
}

/* MixerSetLineGain - sets gain of line (range 0-MIX_MAX_GAIN) */
static int
mixerSetLineGain(HMIXEROBJ hMix, DWORD dwLineID, int gain)
{
        MIXERCONTROLDETAILS_UNSIGNED mcduGain;
        MIXERCONTROLDETAILS mcd;
        MIXERLINECONTROLS mlc;
        MIXERCONTROL      mc;
        MMRESULT          mmr;
        
        mlc.cbStruct      = sizeof(mlc);
        mlc.pamxctrl      = &mc;
        mlc.cbmxctrl      = sizeof(MIXERCONTROL);
        mlc.dwLineID      = dwLineID;
        mlc.dwControlType = MIXERCONTROL_CONTROLTYPE_VOLUME;
        
        mmr = mixerGetLineControls(hMix, &mlc, MIXER_GETLINECONTROLSF_ONEBYTYPE | MIXER_OBJECTF_HMIXER);
        if (mmr != MMSYSERR_NOERROR) {
                debug_msg("Could not volume control for line 0x%08x: %s\n", 
                        dwLineID,
                        mixGetErrorText(mmr));
                return FALSE;        
        }
        
        mcd.cbStruct       = sizeof(mcd);
        mcd.dwControlID    = mc.dwControlID;
        mcd.cChannels      = 1;
        mcd.cMultipleItems = mc.cMultipleItems;
        mcd.cbDetails      = sizeof(MIXERCONTROLDETAILS_UNSIGNED);
        mcd.paDetails      = &mcduGain;
        mcduGain.dwValue   = ((mc.Bounds.dwMaximum - mc.Bounds.dwMinimum) * gain)/MIX_MAX_GAIN;
        
        mmr = mixerSetControlDetails((HMIXEROBJ)hMix, &mcd, MIXER_OBJECTF_HMIXER);
        if (mmr != MMSYSERR_NOERROR) {
                debug_msg("Could not set gain for line 0x%08x: %s\n", dwLineID, mixGetErrorText(mmr));
                return FALSE;
        }
        return TRUE;
}

/* MixerGetLineGain - returns gain of line (range 0-MIX_MAX_GAIN) */
static int
mixerGetLineGain(HMIXEROBJ hMix, DWORD dwLineID)
{
        MIXERCONTROLDETAILS_UNSIGNED mcduGain;
        MIXERCONTROLDETAILS mcd;
        MIXERLINECONTROLS mlc;
        MIXERCONTROL      mc;
        MMRESULT          mmr;
        
        mlc.cbStruct      = sizeof(mlc);
        mlc.pamxctrl      = &mc;
        mlc.cbmxctrl      = sizeof(MIXERCONTROL);
        mlc.dwLineID      = dwLineID;
        mlc.dwControlType = MIXERCONTROL_CONTROLTYPE_VOLUME;
        
        mmr = mixerGetLineControls(hMix, &mlc, MIXER_GETLINECONTROLSF_ONEBYTYPE | MIXER_OBJECTF_HMIXER);
        if (mmr != MMSYSERR_NOERROR) {
                debug_msg("Could not find volume control for line 0x%08x\n", dwLineID);
                return 0;        
        }
        
        mcd.cbStruct       = sizeof(mcd);
        mcd.dwControlID    = mc.dwControlID;
        mcd.cChannels      = 1;
        mcd.cMultipleItems = mc.cMultipleItems;
        mcd.cbDetails      = sizeof(MIXERCONTROLDETAILS_UNSIGNED);
        mcd.paDetails      = &mcduGain;
        
        mmr = mixerGetControlDetails((HMIXEROBJ)hMix, &mcd, MIXER_OBJECTF_HMIXER);
        if (mmr != MMSYSERR_NOERROR) {
                debug_msg("Could not get gain for line 0x%08x\n", dwLineID);
                return 0;
        }
        return (int)(mcduGain.dwValue * MIX_MAX_GAIN / (mc.Bounds.dwMaximum - mc.Bounds.dwMinimum));
}

static int
mixerGetLineName(HMIXEROBJ hMix, DWORD dwLineID, char *szName, UINT uLen)
{
        MIXERLINE           ml;
        MIXERLINECONTROLS   mlc;
        MIXERCONTROL        mc;
        MMRESULT            mmr;
        
        mlc.cbStruct      = sizeof(mlc);
        mlc.pamxctrl      = &mc;
        mlc.cbmxctrl      = sizeof(MIXERCONTROL);
        mlc.dwLineID      = dwLineID;
        mlc.dwControlType = MIXERCONTROL_CONTROLTYPE_VOLUME;
/*        
        mmr = mixerGetLineControls(hMix, &mlc, MIXER_GETLINECONTROLSF_ONEBYTYPE | MIXER_OBJECTF_HMIXER);
        if (mmr != MMSYSERR_NOERROR) {
                debug_msg("Could not find volume control for line 0x%08x: %s\n", dwLineID, mixGetErrorText(mmr));
                return FALSE;        
        }
*/
        memset(&ml,0, sizeof(MIXERLINE));
        ml.cbStruct = sizeof(MIXERLINE);
        ml.dwLineID = dwLineID;
        mmr = mixerGetLineInfo(hMix, &ml, MIXER_GETLINEINFOF_LINEID);
        if (mmr != MMSYSERR_NOERROR) {
                debug_msg("poo");
        }
        debug_msg("line name %s\n", ml.szName);
        strncpy(szName, ml.szName, uLen);
        return TRUE;
}

/* MixQueryControls: Get all line names and id's, fill into ppapd, and return number of lines */
static int
mixQueryControls(HMIXEROBJ hMix, DWORD dwLineID, audio_port_details_t** ppapd)
{
        MIXERCAPS mc;
        MIXERLINE mlt, mlc;
        audio_port_details_t *papd;
        MMRESULT mmr;
        UINT     i;
        
        /* Videum bug work around - Videum does not fill in number of connections if
         * called using MIXER_GETLINEINFOF_LINEID.  This is the only driver with this problem, 
         * but this seems to work in all cases.
         */        
        mixerGetDevCaps((UINT)hMix, &mc, sizeof(mc));
        for(i = 0; i < mc.cDestinations; i++) {
                memset(&mlt, 0, sizeof(mlt));
                mlt.cbStruct = sizeof(mlt);
                mlt.dwDestination = i;
                mmr = mixerGetLineInfo(hMix, &mlt, MIXER_GETLINEINFOF_DESTINATION);
                if (mmr != MMSYSERR_NOERROR) {
                        debug_msg("mixerGetLineInfo: %s\n", mixGetErrorText(mmr));
                        continue;
                }
                if (mlt.dwLineID == dwLineID) {
                        break;
                }
        }
        
        papd = (audio_port_details_t*)xmalloc(sizeof(audio_port_details_t)*mlt.cConnections);
        if (papd == NULL) {
                return 0;
        }
        
        mixerDumpLineInfo((HMIXEROBJ)hMix, mlt.dwLineID);
        
        for(i = 0; i < mlt.cConnections; i++) {
                memcpy(&mlc, &mlt, sizeof(mlc));
                mlc.dwSource = i;
                mmr = mixerGetLineInfo((HMIXEROBJ)hMixer, &mlc, MIXER_GETLINEINFOF_SOURCE|MIXER_OBJECTF_HMIXER);
                if (mmr != MMSYSERR_NOERROR) {
                        xfree(papd);
                        return 0;
                }
                strncpy(papd[i].name, mlc.szName, AUDIO_PORT_NAME_LENGTH);
                papd[i].port = mlc.dwLineID;
        }
        
        *ppapd = papd;
        return (int)mlt.cConnections;
}

/* XXX make_microphone_first_port is a hack to make microphone 
 * the first (default) port.  Of course this only works for
 * english language drivers...
 */

static int
make_microphone_first_port(audio_port_details_t *ports, int n_ports)
{
        audio_port_details_t tmp;
        int i;
        
        for(i = 1; i < n_ports; i++) {
                if (!strncasecmp("mic", ports[i].name, 3) ||
                    !strncasecmp("mik", ports[i].name, 3)) {
                        memcpy(&tmp, ports + i, sizeof(tmp));
                        memcpy(ports + i , ports, sizeof(ports[0]));
                        memcpy(ports, &tmp, sizeof(ports[0]));
                        return TRUE;
                }
        }

        return FALSE;
}

static int 
mixSetup(UINT uMixer)
{
        MIXERCAPS mc;
        MMRESULT  res;
        
        if (hMixer)  {mixerClose(hMixer);  hMixer  = 0;}
        
        res = mixerOpen(&hMixer, uMixer, (unsigned int)NULL, (unsigned long)NULL, MIXER_OBJECTF_MIXER);
        if (res != MMSYSERR_NOERROR) {
                debug_msg("mixerOpen failed: %s\n", mixGetErrorText(res));
                return FALSE;
        }
        
        res = mixerGetDevCaps((UINT)hMixer, &mc, sizeof(mc));
        if (res != MMSYSERR_NOERROR) {
                debug_msg("mixerGetDevCaps failed: %s\n", mixGetErrorText(res));
                return FALSE;
        }
        
        if (mc.cDestinations < 2) {
                debug_msg("mixer does not have 2 destinations?\n");
                return FALSE;
        }
        
        if (input_ports != NULL) {
                xfree(input_ports);
                input_ports   = NULL;
                n_input_ports = 0;
        }
        
        n_input_ports = mixQueryControls((HMIXEROBJ)hMixer, dwRecLineID, &input_ports);
        debug_msg("Input ports %d\n", n_input_ports);
        if (n_input_ports == 0) {
                return FALSE;
        }
        
        make_microphone_first_port(input_ports, n_input_ports);

        if (loop_ports != NULL) {
                xfree(loop_ports);
                loop_ports   = NULL;
                n_loop_ports = 0;
        }
        
        n_loop_ports = mixQueryControls((HMIXEROBJ)hMixer, dwVolLineID, &loop_ports);
        debug_msg("Loop ports %d\n", n_loop_ports);
        if (n_loop_ports == 0) {
                return 0;
        }
        return TRUE;
}

/* Global variables used by read and write processing                                    */
static int blksz;
static int nblks;
static int smplsz;

/* AUDIO OUTPUT RELATED FN's                                                             */
static HWAVEOUT	shWaveOut;         /* Handle for wave output                             */
static WAVEHDR *whWriteHdrs;       /* Pointer to blovk of wavehdr's alloced for writing  */
static u_char  *lpWriteData;       /* Pointer to raw audio data buffer                   */

static int
w32sdk_audio_open_out(UINT uId, WAVEFORMATEX *pwfx)
{
        MMRESULT        mmr;
        int		i;
        
        if (shWaveOut) {
                return (TRUE);
        }
        
        mmr = waveOutOpen(&shWaveOut, uId, pwfx, 0, 0, CALLBACK_NULL);
        if (mmr != MMSYSERR_NOERROR) {
                waveOutGetErrorText(mmr, errorText, sizeof(errorText));
                debug_msg("waveOutOpen: (%d) %s\n", mmr, errorText);
                return (FALSE);
        }
        
        if (lpWriteData != NULL) {
                xfree(lpWriteData);
        }
        lpWriteData = (u_char*)xmalloc(nblks * blksz);
        memset(lpWriteData, 0, nblks * blksz);
        
        if (whWriteHdrs != NULL) {
                xfree(whWriteHdrs);
        }
        whWriteHdrs = (WAVEHDR*)xmalloc(sizeof(WAVEHDR)*nblks);
        memset(whWriteHdrs, 0, sizeof(WAVEHDR)*nblks);

        for (i = 0; i < nblks; i++) {
                whWriteHdrs[i].dwFlags        = 0;
                whWriteHdrs[i].dwBufferLength = blksz;
                whWriteHdrs[i].lpData         = lpWriteData + i * blksz;
                whWriteHdrs[i].dwUser         = i; /* For debugging purposes */
                mmr = waveOutPrepareHeader(shWaveOut, &whWriteHdrs[i], sizeof(WAVEHDR));
                whWriteHdrs[i].dwFlags |= WHDR_DONE; /* Mark buffer as done - used to find free buffers */
		assert(mmr == MMSYSERR_NOERROR);
        }

        return (TRUE);
}

static void
w32sdk_audio_close_out()
{
        int i;
        
        if (shWaveOut == 0) {
                return;
        }
        
	waveOutReset(shWaveOut);
        
	for (i = 0; i < nblks; i++) {
                if (whWriteHdrs[i].dwFlags & WHDR_PREPARED) {
                        waveOutUnprepareHeader(shWaveOut, &whWriteHdrs[i], sizeof(WAVEHDR));
                }
        }
        
	waveOutClose(shWaveOut);
        
	xfree(whWriteHdrs); whWriteHdrs = NULL;
        xfree(lpWriteData); lpWriteData  = NULL;
      
        xmemchk();
        shWaveOut = 0;
}


#define WRITE_ERROR_STILL_PLAYING 33

const char *waveOutError(MMRESULT mmr)
{
	switch (mmr){
		CASE_STRING(MMSYSERR_NOERROR);
		CASE_STRING(MMSYSERR_INVALHANDLE);
		CASE_STRING(MMSYSERR_NODRIVER);
		CASE_STRING(MMSYSERR_NOMEM);
		CASE_STRING(WAVERR_UNPREPARED);
		CASE_STRING(WRITE_ERROR_STILL_PLAYING);
	default:
		return "Unknown";
	}
}

WAVEHDR *
w32sdk_audio_write_get_buffer()
{
	int	 i;

	for (i = 0; i < nblks; i++) {
		assert(whWriteHdrs[i].dwFlags & WHDR_PREPARED);
		if (whWriteHdrs[i].dwFlags & WHDR_DONE) {
			whWriteHdrs[i].dwFlags &= WHDR_PREPARED;
			return &whWriteHdrs[i];
		}
	}
	return NULL;
}

int
w32sdk_audio_write(audio_desc_t ad, u_char *buf , int buf_bytes)
{
        WAVEHDR   *whCur;
        MMRESULT   mmr;
        int        done, this_write;
        
        /* THis is slightly ugly because we handle writes of any size, not just */
        /* multiples of blksz. RAT likes to write multiples of the cushion step */
        /* size which is usually 1/2 device blksz.                              */
               
        done = 0;
        while(done < buf_bytes) {
                whCur = w32sdk_audio_write_get_buffer();
		if (whCur == NULL) {
			debug_msg("Write/Right out of buffers ???\n");
			break;
		}
		this_write = min(buf_bytes - done, (int)blksz);
		whCur->dwBufferLength = this_write;
                memcpy(whCur->lpData, 
                        buf + done,
                        this_write);
                done  += this_write;
                mmr    = waveOutWrite(shWaveOut, whCur, sizeof(WAVEHDR));
                if (mmr == WRITE_ERROR_STILL_PLAYING) {
                        debug_msg("Device filled\n");
                        break;
                }
                assert(mmr == MMSYSERR_NOERROR);
        }

        assert(done <= buf_bytes);
       
        return done;
}

/* AUDIO INPUT RELATED FN's *********************************/

static HWAVEIN	shWaveIn;               /* Handle for wave input                                */
static WAVEHDR	*whReadHdrs;            /* Pointer to block of wavehdr's allocated for reading  */
static u_char	*lpReadData;            /* Pointer to raw audio data buffer                     */
static WAVEHDR  *whReadList;            /* List of wave headers that have been read but not     */
                                        /* given to the application.                            */ 
static DWORD     dwBytesUsedAtReadHead; /* Number of bytes that have already been read at head  */
static HANDLE    hAudioReady;           /* Audio Ready Event */

int
w32sdk_audio_is_ready(audio_desc_t ad)
{
        UNUSED(ad);
        return (whReadList != NULL);
}

static void CALLBACK
waveInProc(HWAVEIN hwi,
           UINT    uMsg,
           DWORD   dwInstance,
           DWORD   dwParam1,
           DWORD   dwParam2)
{
        WAVEHDR *whRead, **whInsert;

        switch(uMsg) {
        case WIM_DATA:
                whRead = (WAVEHDR*)dwParam1;       
		/* Insert block at the available list */
                whRead->lpNext   = NULL;  
		whInsert = &whReadList;
                while(*whInsert != NULL) {
                        whInsert = &((*whInsert)->lpNext);
                }
                *whInsert = whRead;
                SetEvent(hAudioReady);
		break;
        default:
                ;  /* nothing to do currently */
        }
        UNUSED(dwInstance);
        UNUSED(dwParam2);
        UNUSED(hwi);
        
        return;
}

static int
w32sdk_audio_open_in(UINT uId, WAVEFORMATEX *pwfx)
{
        MMRESULT mmr;
        int      i;
        
        if (shWaveIn) {
		return (TRUE);
	}
        
        if (lpReadData != NULL) {
		xfree(lpReadData);
	}
        lpReadData = (u_char*)xmalloc(nblks * blksz);
        
        if (whReadHdrs != NULL) {
		xfree(whReadHdrs);
	}
        whReadHdrs = (WAVEHDR*)xmalloc(sizeof(WAVEHDR)*nblks); 
        
        mmr = waveInOpen(&shWaveIn, 
                         uId, 
                         pwfx,
                         (DWORD)waveInProc,
                         0,
                         CALLBACK_FUNCTION);
        
	if (mmr != MMSYSERR_NOERROR) {
                waveInGetErrorText(mmr, errorText, sizeof(errorText));
                debug_msg("waveInOpen: (%d) %s\n", mmr, errorText);
                return (FALSE);
        }

        /* Initialize wave headers */
        for (i = 0; i < nblks; i++) {
                whReadHdrs[i].lpData         = lpReadData + i * blksz;
                whReadHdrs[i].dwBufferLength = blksz;
                whReadHdrs[i].dwFlags        = 0;
                mmr = waveInPrepareHeader(shWaveIn, &whReadHdrs[i], sizeof(WAVEHDR));
                assert(mmr == MMSYSERR_NOERROR);               
                mmr = waveInAddBuffer(shWaveIn, &whReadHdrs[i], sizeof(WAVEHDR));
                assert(mmr == MMSYSERR_NOERROR);
        }

        whReadList           = NULL;
        dwBytesUsedAtReadHead = 0;

        error = waveInStart(shWaveIn);
        if (error) {
                waveInGetErrorText(error, errorText, sizeof(errorText));
                debug_msg("Win32Audio: waveInStart: (%d) %s\n", error, errorText);
                exit(1);
        }
        hAudioReady = CreateEvent(NULL, TRUE, FALSE, "RAT Audio Ready");
        
	return (TRUE);
}

static void
w32sdk_audio_close_in()
{
        int		i;
        
        if (shWaveIn == 0)
                return;
        
        waveInStop(shWaveIn);
        waveInReset(shWaveIn);
        
        for (i = 0; i < nblks; i++) {
                if (whReadHdrs[i].dwFlags & WHDR_PREPARED) {
                        waveInUnprepareHeader(shWaveIn, &whReadHdrs[i], sizeof(WAVEHDR));
                }
        }
        whReadList = NULL;

        waveInClose(shWaveIn);           
	shWaveIn = 0;
        
	xfree(whReadHdrs);               
	whReadHdrs = NULL;
        
	xfree(lpReadData);               
	lpReadData  = NULL;
        
        xmemchk();
}

int
w32sdk_audio_read(audio_desc_t ad, u_char *buf, int buf_bytes)
{
        WAVEHDR *whCur;
	MMRESULT mmr;
        int done = 0, this_read;
        static int added;
        
	/* This is slightly ugle because we want to be able to operate when     */
        /* buf_bytes has any value, not just a multiple of blksz.  In principle */
        /* we do this so the device blksz does not have to match application    */
        /* blksz.  I.e. can reduce process usage by using larger blocks at      */
        /* device whilst the app operates on smaller blocks.                    */

        while(whReadList != NULL && done < buf_bytes) {
		whCur = whReadList;
		this_read = min((int)(whCur->dwBytesRecorded - dwBytesUsedAtReadHead), buf_bytes - done);
                if (buf) {
			memcpy(buf + done, 
			       whCur->lpData + dwBytesUsedAtReadHead,
			       this_read);
		}
                done                  += this_read;
                dwBytesUsedAtReadHead += this_read;
                if (dwBytesUsedAtReadHead == whCur->dwBytesRecorded) {
                        whReadList = whReadList->lpNext;
			/* Finished with the block give it device */
			assert(whCur->dwFlags & WHDR_DONE);
			assert(whCur->dwFlags & ~WHDR_INQUEUE);
                   	whCur->lpNext          = NULL; 
			whCur->dwBytesRecorded = 0; 
			whCur->dwFlags        &= ~WHDR_DONE;
			mmr = waveInAddBuffer(shWaveIn, whCur, sizeof(WAVEHDR)); 
                        assert(mmr == MMSYSERR_NOERROR);
			assert(whCur->dwFlags & WHDR_INQUEUE);
                        dwBytesUsedAtReadHead = 0;
			added++;
                }
		assert((int)dwBytesUsedAtReadHead < blksz);
        }
	
        assert(done <= buf_bytes);
	UNUSED(ad);
        return done;
}

void
w32sdk_audio_drain(audio_desc_t ad)
{
	waveInStop(shWaveIn);
        w32sdk_audio_read(ad, NULL, 10000000);
	waveInStart(shWaveIn);
}

static void dumpReadHdrStats()
{
	WAVEHDR *whp;
	int i, done, inqueue, prepared, ready;
	
	done = inqueue = prepared = 0;
	for(i = 0; i < nblks; i++) {
		if (whReadHdrs[i].dwFlags & WHDR_DONE) {
			done++;
		}
		if (whReadHdrs[i].dwFlags & WHDR_INQUEUE) {
			inqueue++;
		}
		if (whReadHdrs[i].dwFlags & WHDR_PREPARED) {
			prepared++;
		}
	}

	ready = 0;
	whp = whReadList;
	while (whp != NULL) {
		ready++;
		whp = whp->lpNext;
	}
	debug_msg("done %d inqueue %d prepared %d ready %d\n",
		done, inqueue, prepared, ready);
}

void
w32sdk_audio_wait_for(audio_desc_t ad, int delay_ms)
{        
        if (whReadList == NULL) {
                DWORD dwRes;
		dwRes = WaitForSingleObject(hAudioReady, delay_ms);
		switch(dwRes) {
		case WAIT_TIMEOUT:
			debug_msg("No audio (%d ms wait timeout)\n", delay_ms);
			dumpReadHdrStats();
			break;
		case WAIT_FAILED:
			debug_msg("Wait failed (error %u)\n", GetLastError());
			break;
		}
                ResetEvent(hAudioReady);
        }
	waveInStart(shWaveIn);
        UNUSED(ad);
}

static int audio_dev_open = 0;

static int
w32sdk_audio_open_mixer(audio_desc_t ad, audio_format *fmt, audio_format *ofmt)
{
        static int virgin;
        WAVEFORMATEX owfx, wfx;
        UINT uWavIn, uWavOut;

        if (audio_dev_open) {
                debug_msg("Device not closed! Fix immediately");
                w32sdk_audio_close(ad);
        }
        
        assert(audio_format_match(fmt, ofmt));
        if (fmt->encoding != DEV_S16) {
                return FALSE; /* Only support L16 for time being */
        }
        
        if (mixGetInputInfo(ad, &uWavIn, &dwRecLineID) != TRUE) {
                debug_msg("Could not get wave in or mixer destination for mix %u\n", ad);
                return FALSE;
        }

        if (mixGetOutputInfo(ad, &uWavOut, &dwVolLineID) != TRUE) {
                debug_msg("Could not get wave out or mixer destination for mix %u\n", ad);
                return FALSE;
        }
        
        if (mixSetup(ad) == FALSE) {
                return FALSE; /* Could not secure mixer */
        }

        mixSaveControls(ad, &control_list);

        wfx.wFormatTag      = WAVE_FORMAT_PCM;
        wfx.nChannels       = (WORD)fmt->channels;
        wfx.nSamplesPerSec  = fmt->sample_rate;
        wfx.wBitsPerSample  = (WORD)fmt->bits_per_sample;
        smplsz              = wfx.wBitsPerSample / 8;
        wfx.nAvgBytesPerSec = wfx.nChannels * wfx.nSamplesPerSec * smplsz;
        wfx.nBlockAlign     = (WORD)(wfx.nChannels * smplsz);
        wfx.cbSize          = 0;
        
        memcpy(&owfx, &wfx, sizeof(wfx));
        
        /* Use 1 sec device buffer */	
        blksz  = fmt->bytes_per_block;
        nblks  = wfx.nAvgBytesPerSec / blksz;
        
        if (w32sdk_audio_open_in(uWavIn, &wfx) == FALSE){
                debug_msg("Open input failed\n");
                return FALSE;
        }
        
        assert(memcmp(&owfx, &wfx, sizeof(WAVEFORMATEX)) == 0);
        
        if (w32sdk_audio_open_out(uWavOut, &wfx) == FALSE) {
                debug_msg("Open output failed\n");
                w32sdk_audio_close_in();
                return FALSE;
        }

        /* because these get can corrupted... */
        assert(memcmp(&owfx, &wfx, sizeof(WAVEFORMATEX)) == 0);
    
        /* Set process priority as high as we can go without special permissions on */
        /* on NT.  Although this priority may seem anti-social, it's not that bad   */
        /* since we block whilst waiting for audio events.                          */
        SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
        
        /* SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);  */

        if (!have_probed[ad]) {
                have_probed[ad] = w32sdk_probe_formats(ad);
        }
 
        audio_dev_open = TRUE;
        return TRUE;
}

int 
w32sdk_audio_open(audio_desc_t ad, audio_format *ifmt, audio_format *ofmt)
{
        ad = mapAudioDescToMixerID(ad);
        return w32sdk_audio_open_mixer(ad, ifmt, ofmt);
}

static void
w32sdk_audio_close_mixer(audio_desc_t ad)
{
        MMRESULT mmr;

        debug_msg("Closing input device.\n");
        w32sdk_audio_close_in();
        
        debug_msg("Closing output device.\n");
        w32sdk_audio_close_out();
        
        if (input_ports != NULL) {
                xfree(input_ports);
                input_ports = NULL;
        }
        
        if (loop_ports != NULL) {
                xfree(loop_ports);
                loop_ports = NULL;
        }
        
        mixRestoreControls(ad, &control_list);
        mmr = mixerClose(hMixer); hMixer = 0;
        if (mmr != MMSYSERR_NOERROR) {
                debug_msg("mixerClose failed: %s\n", mixGetErrorText(mmr));
        }
        
        audio_dev_open = FALSE;
        UNUSED(ad);
}

void
w32sdk_audio_close(audio_desc_t ad)
{
        w32sdk_audio_close_mixer(ad);
}

int
w32sdk_audio_duplex(audio_desc_t ad)
{
        UNUSED(ad);
        return (TRUE);
}


void
w32sdk_audio_non_block(audio_desc_t ad)
{
        UNUSED(ad);
        debug_msg("Windows audio interface is asynchronous!\n");
}

void
w32sdk_audio_block(audio_desc_t ad)
{
        UNUSED(ad);
        debug_msg("Windows audio interface is asynchronous!\n");
}

void
w32sdk_audio_set_ogain(audio_desc_t ad, int level)
{
        DWORD	vol;
        
        UNUSED(ad);
        
        play_vol = level;
        
        if (shWaveOut == 0)
                return;
        
        vol = rat_to_device(level);
        error = waveOutSetVolume(shWaveOut, vol);
        if (error) {
                waveOutGetErrorText(error, errorText, sizeof(errorText));
                debug_msg("Win32Audio: waveOutSetVolume: %s\n", errorText);
        }
}

int
w32sdk_audio_get_ogain(audio_desc_t ad)
{
        DWORD	vol;
        
        UNUSED(ad);
        
        if (shWaveOut == 0) {
                return play_vol;
	}
        
        error = waveOutGetVolume(shWaveOut, &vol);
        if (error) {
                waveOutGetErrorText(error, errorText, sizeof(errorText));
                debug_msg("Win32Audio: waveOutGetVolume Error: %s\n", errorText);
                return 0;
        } else {
                return (device_to_rat(vol));
        }
}

void
w32sdk_audio_loopback(audio_desc_t ad, int gain)
{
        UNUSED(ad);
        
        nLoopGain = gain;
}

#define WIN32_SPEAKER 0x101010
static audio_port_details_t outports[] = {
        { WIN32_SPEAKER, AUDIO_PORT_SPEAKER}
};
#define NUM_OPORTS (sizeof(outports)/sizeof(outports[0]))

void
w32sdk_audio_oport_set(audio_desc_t ad, audio_port_t port)
{
        UNUSED(ad);
        UNUSED(port);
}

/* Return selected output port */
audio_port_t 
w32sdk_audio_oport_get(audio_desc_t ad)
{
        UNUSED(ad);
        return (WIN32_SPEAKER);
}

int
w32sdk_audio_oport_count(audio_desc_t ad)
{
        UNUSED(ad);
        return (int)NUM_OPORTS;
}

const audio_port_details_t*
w32sdk_audio_oport_details(audio_desc_t ad, int idx)
{
        UNUSED(ad);
        assert(idx >= 0 && idx < NUM_OPORTS);
        return &outports[idx];
}

void 
w32sdk_audio_iport_set(audio_desc_t ad, audio_port_t port)
{
        char portname[MIXER_LONG_NAME_CHARS+1];
        int i, j, gain;
        UNUSED(ad);
        
        for(i = 0; i < n_input_ports; i++) {
                if (input_ports[i].port == port) {
                        /* save gain */
                        gain = mixerGetLineGain((HMIXEROBJ)hMixer, input_ports[iport].port);
                        debug_msg("Gain %d\n", gain);
                        if (mixerGetLineName((HMIXEROBJ)hMixer, input_ports[i].port, portname, MIXER_LONG_NAME_CHARS)) {
                                mixerEnableInputLine((HMIXEROBJ)hMixer, portname);
                        }
                        mixerSetLineGain((HMIXEROBJ)hMixer, input_ports[i].port, gain);
                        
                        /* Do loopback */
                        for(j = 0; j < n_loop_ports && nLoopGain != 0; j++) {
                                if (strcmp(loop_ports[j].name, input_ports[i].name) == 0) {
                                        mixerEnableOutputLine((HMIXEROBJ)hMixer, loop_ports[j].port, 1);
                                        /* mixerSetLineGain((HMIXEROBJ)hMixer, loop_ports[j].port, nLoopGain); */
                                }
                        }
                        iport = i;
                        return;
                }
        }
        debug_msg("Port %d not found\n", port);
}

/* Return selected input port */
audio_port_t
w32sdk_audio_iport_get(audio_desc_t ad)
{
        UNUSED(ad);
        return input_ports[iport].port;
}

int
w32sdk_audio_iport_count(audio_desc_t ad)
{
        UNUSED(ad);
        return n_input_ports;
}

const audio_port_details_t*
w32sdk_audio_iport_details(audio_desc_t ad, int idx)
{
        UNUSED(ad);
        assert(idx >= 0 && idx < n_input_ports);
        return &input_ports[idx];
}

void
w32sdk_audio_set_igain(audio_desc_t ad, int level)
{
        UNUSED(ad);
        assert(iport >= 0 && iport < n_input_ports);
        mixerSetLineGain((HMIXEROBJ)hMixer, input_ports[iport].port, level);
}

int
w32sdk_audio_get_igain(audio_desc_t ad)
{
        UNUSED(ad);
        assert(iport >= 0 && iport < n_input_ports);
        return mixerGetLineGain((HMIXEROBJ)hMixer, input_ports[iport].port);
}

/* Probing support */

static audio_format af_sup[W32SDK_MAX_DEVICES][10];
static int          n_af_sup[W32SDK_MAX_DEVICES];

int
w32sdk_probe_format(int rate, int channels)
{
        WAVEFORMATEX wfx;
        
        wfx.cbSize = 0; /* PCM format */
        wfx.wFormatTag      = WAVE_FORMAT_PCM;
        wfx.wBitsPerSample  = 16; /* 16 bit linear */
        wfx.nChannels       = channels;
        wfx.nSamplesPerSec  = rate;
        wfx.nBlockAlign     = wfx.wBitsPerSample / 8 * wfx.nChannels;
        wfx.nAvgBytesPerSec = wfx.nSamplesPerSec * wfx.nBlockAlign;
        
        if (waveInOpen(NULL, (UINT)shWaveIn, &wfx, (UINT)NULL, (UINT)NULL, WAVE_FORMAT_QUERY)) {
                debug_msg("%d %d supported\n", rate, channels);
                return TRUE;
        }
        
        debug_msg("%d %d not supported\n", rate, channels);
        return FALSE;      
}

int 
w32sdk_probe_formats(audio_desc_t ad) 
{
        int rate, channels;

        for (rate = 8000; rate <= 48000; rate+=8000) {
                if (rate == 24000 || rate == 40000) continue;
                for(channels = 1; channels <= 2; channels++) {
                        if (w32sdk_probe_format(rate, channels)) {
                                af_sup[ad][n_af_sup[ad]].sample_rate = rate;
                                af_sup[ad][n_af_sup[ad]].channels    = channels;
                                n_af_sup[ad]++;
                        }
                }
        }
        return (n_af_sup[ad] ? TRUE : FALSE); /* Managed to find at least 1 we support    */
                                              /* We have this test since if we cannot get */
                                              /* device now (because in use elsewhere)    */
                                              /* we will want to test it later            */
}

int
w32sdk_audio_supports(audio_desc_t ad, audio_format *paf)
{
        int i;
        ad = mapAudioDescToMixerID(ad);
        for(i = 0; i < n_af_sup[ad]; i++) {
                if (af_sup[ad][i].sample_rate  == paf->sample_rate &&
                        af_sup[ad][i].channels == paf->channels) {
                        return TRUE;
                }
        }
        return FALSE;
}

static int   nMixersWithFullDuplex = 0;
static UINT *mixerIdMap;

static UINT 
mapAudioDescToMixerID(audio_desc_t ad)
{
        return mixerIdMap[ad];
}

int
w32sdk_audio_init(void)
{
        audio_format af;
        unsigned int i;

        mixerIdMap = (UINT*)xmalloc(sizeof(UINT) * mixerGetNumDevs());

        af.bits_per_sample = 16;
        af.bytes_per_block = 320;
        af.channels        = 1;
        af.encoding        = DEV_S16;
        af.sample_rate     = 8000;
        
        for(i = 0; i < mixerGetNumDevs(); i++) {
                if (w32sdk_audio_open_mixer(i, &af, &af)) {
                        w32sdk_audio_close_mixer(i);
                        mixerIdMap[nMixersWithFullDuplex] = i;
                        nMixersWithFullDuplex++;
                }
        }

        return nMixersWithFullDuplex;
}

int 
w32sdk_audio_free(void)
{
	xfree(mixerIdMap);
	return TRUE;
}

int
w32sdk_get_device_count(void)
{
        /* We are only interested in devices with mixers */
        return (int)nMixersWithFullDuplex;
}

static char tmpname[MAXPNAMELEN];

char *
w32sdk_get_device_name(int idx)
{
        MIXERCAPS mc;
        idx = mapAudioDescToMixerID(idx);
        if ((UINT)idx < mixerGetNumDevs()) {
                mixerGetDevCaps((UINT)idx, &mc, sizeof(mc));
                strncpy(tmpname, mc.szPname, MAXPNAMELEN);
                return tmpname;
        }
        return NULL;
}
#endif /* WIN32 */
