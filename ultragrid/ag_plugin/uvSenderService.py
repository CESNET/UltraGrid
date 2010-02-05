#-----------------------------------------------------------------------------
# Name:        uvSenderService.py
# Purpose:     Start the UltraGrid video client
#
# Author:      Ladan Gharai
#              Colin Perkins
#
# Created:     13 August 2004
# Copyright:   (c) 2004 University of Southern California
#              (c) 2004 University of Glasgow
#
# $Revision: 1.1 $
# $Date: 2007/11/08 09:48:58 $
#
#-----------------------------------------------------------------------------
import sys, os
try:    import _winreg
except: pass

from AccessGrid.Types           import Capability
from AccessGrid.AGService       import AGService
from AccessGrid.AGParameter     import ValueParameter, OptionSetParameter, RangeParameter, TextParameter
from AccessGrid                 import Platform
from AccessGrid.Platform.Config import AGTkConfig, UserConfig
from AccessGrid.NetworkLocation import MulticastNetworkLocation

class uvSenderService( AGService ):
    def __init__( self ):
        AGService.__init__( self )
        self.capabilities = [ Capability( Capability.PRODUCER, Capability.VIDEO ) ]
        self.executable = os.path.join('.','uv')
        # Set configuration parameters
        pass

    def __SetRTPDefaults(self, profile):
        if profile == None:
            self.log.exception("Invalid profile (None)")
            raise Exception, "Can't set RTP Defaults without a valid profile."
        if sys.platform == 'linux2':
            try:
                rtpDefaultsFile=os.path.join(os.environ["HOME"], ".RTPdefaults")
                rtpDefaultsFH=open( rtpDefaultsFile,"w")
		rtpDefaultsFH.write( "*rtpName:  %s\n" % ( profile.name ) )
		rtpDefaultsFH.write( "*rtpEmail: %s\n" % ( profile.email ) )
		rtpDefaultsFH.write( "*rtpPhone: %s\n" % ( profile.phoneNumber ) )
		rtpDefaultsFH.write( "*rtpLoc:   %s\n" % ( profile.location ) )
		rtpDefaultsFH.write( "*rtpNote:  %s\n" % ( profile.publicId ) )
                rtpDefaultsFH.close()
            except:
                self.log.exception("Error writing RTP defaults file: %s", rtpDefaultsFile)
        elif sys.platform == 'win32':
            try:
                # Set RTP defaults according to the profile
                k = _winreg.CreateKey(_winreg.HKEY_CURRENT_USER, r"Software\Mbone Applications\common")
                _winreg.SetValueEx(k, "*rtpName",  0, _winreg.REG_SZ, self.profile.name)
                _winreg.SetValueEx(k, "*rtpEmail", 0, _winreg.REG_SZ, self.profile.email)
                _winreg.SetValueEx(k, "*rtpPhone", 0, _winreg.REG_SZ, self.profile.phoneNumber)
                _winreg.SetValueEx(k, "*rtpLoc",   0, _winreg.REG_SZ, self.profile.location)
                _winreg.SetValueEx(k, "*rtpNote",  0, _winreg.REG_SZ, str(self.profile.publicId) )
                _winreg.CloseKey(k)
            except:
                self.log.exception("Error writing RTP defaults to registry")
        
    def Start( self ):
        __doc__ = """Start service"""
        try:
            # Start the service; in this case, store command line args in a list and let
            # the superclass _Start the service
            options = []
            if self.streamDescription.name and len(self.streamDescription.name.strip()) > 0:
		# Not yet supported -- csp
		pass
                # options.append( "-C" )
                # options.append( self.streamDescription.name )
            if self.streamDescription.encryptionFlag != 0:
		# Not yet supported -- csp
		pass
                # options.append( "-K" )
                # options.append( self.streamDescription.encryptionKey )
            # Check whether the network location has a "type" attribute
            # Note: this condition is only to maintain compatibility between
            # older venue servers creating network locations without this attribute
            # and newer services relying on the attribute; it should be removed
            # when the incompatibility is gone
            if self.streamDescription.location.__dict__.has_key("type"):
                if self.streamDescription.location.type == MulticastNetworkLocation.TYPE:
		    # Not yet supported -- csp
                    pass
                    # options.append( "-t" )
                    # options.append( '%d' % ( self.streamDescription.location.ttl ) )

            # Command line options to make UltraGrid run in sending mode:
            options.append( '-t hdtv' )

            # Add address/port options (these must occur last; don't add options beyond here)
            options.append( '%s' % ( self.streamDescription.location.host ) )
            # We don't yet support setting the port -- csp
            # options.append( '%s/%d' % ( self.streamDescription.location.host,
            #                             self.streamDescription.location.port ) )
            self.log.info("Starting uvSenderService")
            self.log.info(" executable = %s" % self.executable)
            self.log.info(" options    = %s" % options)
            self._Start( options )
        except:
            self.log.exception("Exception in uvSenderService.Start")
            raise Exception("Failed to start service")
    Start.soap_export_as = "Start"

    def Stop( self ):
        """Stop the service"""
        # uv doesn't die easily (on Linux at least), so force it to stop
        AGService.ForceStop(self)

    Stop.soap_export_as = "Stop"

    def ConfigureStream( self, streamDescription ):
        """Configure the Service according to the StreamDescription"""

        ret = AGService.ConfigureStream( self, streamDescription )
        if ret and self.started:
            # service is already running with this config; ignore
            return

        # if started, stop
        if self.started:
            self.Stop()

        # if enabled, start
        if self.enabled:
            self.Start()

    ConfigureStream.soap_export_as = "ConfigureStream"

    def SetIdentity(self, profile):
        """
        Set the identity of the user driving the node
        """
        self.__SetRTPDefaults(profile)
    SetIdentity.soap_export_as = "SetIdentity"



if __name__ == '__main__':
    from AccessGrid.AGService import AGServiceI, RunService

    service = uvSenderService()
    serviceI = AGServiceI(service)
    RunService(service,serviceI,int(sys.argv[1]))

