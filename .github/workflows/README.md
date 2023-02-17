UltraGrid GitHub workflows
==========================

Table of contents
-----------------
- [Dependencies](#dependencies)
  * [Linux](#linux)
  * [macOS](#macos)
  * [Windows](#windows)
- [Secrets](#secrets)
  * [Generating Apple keys to sign the image](#generating-apple-keys-to-sign-the-image)
- [Workflows](#workflows)
  * [ARM builds](#arm-builds)
  * [Coverity](#coverity)
  * [C/C++ CI](#cc-ci)

Dependencies
------------
Further are described external dependencies needed to build proprietary parts
of _UltraGrid_. The dependencies are not required - UltraGrid would build also
without.

These additional dependencies must be provided at URL specified by a secret **SDK\_URL**.

Further subsection briefly describe individual assets and how to obtain them.  All assets
are unmodified files downloaded from vendor website. However, rename may be required.

### macOS
- **VideoMaster\_SDK\_MacOSX.zip** - VideoMaster SDK for Mac from
  [DELTACAST](https://www.deltacast.tv/support/download-center)

### Windows
- **VideoMaster\_SDK\_Windows.zip** - VideoMaster SDK from DELTACAST for Windows

### Linux
**Note:** _VideoMaster SDK_ is not used because DELTACAST doesn't provide redistributable
libraries for Linux (as does for macOS and Windows).

Secrets
-------
- **NOTARYTOOL\_CREDENTIALS** - Apple developer credentials to be used with notarytool for macOS build (username:password:teamid)
  notarization in format "user:password" (app-specific password is strongly recommended)
- **APPLE\_KEY\_P12\_B64** - base64-encoded signing Apple key in P12 (see [below](#generating-apple-keys-to-sign-the-image))
- **APPIMAGE\_KEY** - GPG exported (armored) private key to sign AppImage
- **COVERITY\_TOKEN** - Coverity token to be used for build upload
- **NDI\_REMOTE\_SSH\_KEY** - SSH key to upload NDI builds over SSH (see [Workflows](#workflows) for additional details)
- **SDK\_URL** - URL where are located the [Dependencies](#dependencies) assets

**Note:** not all secrets are used by all workflows (see [Workflows](#workflows) for details)

### Generating Apple keys to sign the image

This section contains a procedure to get and process keys to be used as _APPLE\_KEY\_P12\_B64_ above.

- first generate signing request (replace _subject_ if needed):
   
   `openssl genrsa -out mykey.key 2048`
   `openssl req -new -key mykey.key -out CertificateSigningRequest.certSigningRequest -subj "/emailAddress=ultragrid-dev@cesnet.cz, CN=CESNET, C=CZ"`

- then login to Apple developer and generate a certificate from the above signing request for _"Developer ID Application"_
  and download **developerID\_application.cer**

- convert certificate to to PEM:
   
   `openssl x509 -inform DER -outform PEM -text -in developerID_application.cer -out developerID_application.pem`

- export private key with password "dummy":
  
  `openssl pkcs12 -export -out signing_key.p12 -in developerID_application.pem -inkey mykey.key -passout pass:dummy`

- add GitHub action secret **APPLE\_KEY\_P12\_B64** from output of:
   
   `base64 signing_key.p12`

Workflows
--------
Currently all workflows are triggered by push to the respective branch. There are 3 workflows:

### ARM builds 
Creates _ARM AppImages_. Trigerred by push to branch **arm-build**. In _CESNET/UltraGrid_ repo creates a release
asset, otherwise a build artifact. No secret are used.

### Coverity
Sends build for analysis to _Coverity_ service. Trigerred by push to **coverity\_scan** - requires
**COVERITY\_TOKEN**, useful is also **SDK\_URL** to increase code coverage.

### C/C++ CI
This is the basic workflow, has multiple modes depending on which branch is pushed to. Whether or not triggered
from _official_ repository influences where will the build be uploaded:

* push to _official_ repository (branches **master** or **release/\***) - triggers rebuild of release asset (_continuous_ for master) and uploads to
  release assets.
* push to _other_ repositories (branches **master** or **release/\***) - creates build artifacts
* push to branch **ndi-build** - builds with NDI support - requires NDI SDKs to be present in **SDK\_URL**, otherwise the _NDI_ support won't be enabled.
  - reads **NDI\_REMOTE\_SSH\_KEY**, if found, uploads the builds to predefined location (defined in [upload-ndi-build.sh](../scripts/upload-ndi-build.sh)).
    For non-official repositiry you would also need to set environment variables **NDI\_REMOTE\_SSH** and
    **NDI\_REMOTE\_SSH\_HOST\_KEY\_URL** to override defaults in the script - add following lines to [ccpp.yml](ccpp.yml):

        env:
          NDI_REMOTE_SSH: <user>@<host>:<path>
          NDI_REMOTE_SSH_HOST_KEY_URL: https://<path_to_host_key>
  - If the secret _NDI\_REMOTE\_SSH\_KEY_ is not defined, builds are uploaded as a build artifact.


This worflow utilizes **ALTOOL\_CREDENTIALS**, **APPLE\_KEY\_P12\_B64**, **APPIMAGE\_KEY**, **SDK\_URL** and **SSH\_KEY** (NDI only).

