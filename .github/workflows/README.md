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
  + [Semi-weekly tests](#semi-weekly-tests)

Dependencies
------------
Further are described external dependencies needed to build proprietary parts
of _UltraGrid_. The dependencies are not required - UltraGrid would build also
without.

These additional dependencies must be provided at URL specified by a secret **SDK\_URL**.

Further subsection briefly describe individual assets and how to obtain them.  All assets
are unmodified files downloaded from vendor website. However, rename may be required.

### macOS
- **videomaster-macos-dev.tar.gz** - VideoMaster SDK for Mac from
  [DELTACAST](https://www.deltacast.tv/support/download-center)

### Windows
- **videomaster-win.x64-dev.zip** - VideoMaster SDK from DELTACAST for Windows

### Linux
**Note:** _VideoMaster SDK_ is not used because DELTACAST doesn't provide redistributable
libraries for Linux (as does for macOS and Windows).

Secrets
-------
- **APPIMAGE\_KEY** - GPG exported (armored) private key to sign AppImage;
- **APPLE\_KEY\_P12\_B64** - base64-encoded Apple signing key in P12
(see [below](#generating-apple-keys-to-sign-the-image))
currently unused (no signing as for now - see the commit 2e321f65)
- **COVERITY\_TOKEN** - Coverity token to be used for build (Coverity CI only)
- **NOTARYTOOL_CREDENTIALS** - Apple developer credentials to be used with
notarytool for macOS build (username:password:teamid) notarization in
format "user:password" (app-specific password is strongly recommended)
- **SDK\_URL** - URL where are located the [Dependencies](#dependencies) assets
(currently Deltacast only)

**Note:** not all secrets are used by all workflows (see [Workflows](#workflows) for details)

**Note2:** secret names are _case-insensitive_

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
Currently workflows are triggered by push to the respective branch. Some of
are triggered by a cron (schedule). Workflow dispatches available as well.

There are 4 main workflows and 3 OS-specific reusable workflows called
by C/C++ CI:

### ARM builds

(file [.github/workflows/arm-build.yml](arm-build.yml))

Creates _ARM AppImages_. Trigerred by push to branch **arm-build**. In _CESNET/UltraGrid_ repo creates a release
asset, otherwise a build artifact. No secret are used.

Trigerred also once a week by a schedule.

### Coverity

(file [.github/workflows/coverity-scan.yml](coverity-scan.yml))

Sends build for analysis to _Coverity_ service. Trigerred by push to **coverity\_scan** - requires
**COVERITY\_TOKEN**, optionally also **SDK\_URL** to increase code coverage.

Coverity workflow currently uses Linux runner only.

Trigerred also 2 times per week by a schedule.

### C/C++ CI

(file [.github/workflows/ccpp.yml](ccpp.yml))

This is the basic workflow, has multiple modes depending on which branch is pushed to. Whether or not triggered
from _official_ repository influences where will the build be uploaded:

* push to _official_ repository (branch **master** or **v[0-9]+**) -
triggers rebuild of release asset (_continuous_ for master) and uploads
to release assets.
* push to _other_ repositories (branch **master** or **v[0-9]+**) -
creates build artifacts

This worflow utilizes **APPLE\_KEY\_P12\_B64**, **APPIMAGE\_KEY**
(currently not used, see above),  **NOTARYTOOL\_CREDENTIALS**,**SDK\_URL**.

The OS-specific runners are split to 3 YAML files, that can be triggered
separately by pushing to the respecitve branches - the upload rules are
the above, which means that build artifact is created if pushed to the
specific branch:

- [**linux.yml**](linux.yml) - push to _linux-build_
- [**macos.yml**](macos.yml) - push to _macos-build_
- [**windows.yml**](windows.yml) - push to _windows-build_

#### Default runner override

MacOS runners can be overridden by specifying MACOS_ARM64_RUNNER
or MACOS_X86_64_RUNNER repo variable to custom name like
_macos-15-arm64-self-hosted_. This may be used to route the build
to eg. a self-hosted runner. The self-hosted runner must, however,
behave like the GitHub-hosted image - namely have the same relevant SW
equipment. Self-hosted runners run in the environment where invoked, so
on bare metal if invoked so. Also GH doesn't provide the images directly
but there are [build recipes](https://github.com/actions/runner-images).

### Semi-weekly tests

(file [.github/workflows/semi-weekly_tests.yml](semi-weekly_tests.yml))

Some additional tests run twice a week in the main repo.
