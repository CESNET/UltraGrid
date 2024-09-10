Set-PSDebug -Trace 1

# Free some space - TODO: regular uninstall would be better
#Remove-Item -Recurse "C:\Program Files (x86)\Android"
#Remove-Item -Recurse "C:\Program Files (x86)\dotnet"

# Install CUDA
if (!${env:no_cuda}) {
  $url="https://developer.download.nvidia.com/compute/cuda/11.8.0/network_installers/cuda_11.8.0_windows_network.exe"
  $url -match 'cuda/(?<version>[0-9]+.[0-9]+)'
  $version=$Matches.version
  Invoke-WebRequest $url -OutFile cuda_inst.exe
  Start-Process -FilePath "cuda_inst.exe" -ArgumentList "-s cudart_$version nvcc_$version" -Wait -NoNewWindow
  Remove-Item cuda_inst.exe
  echo "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v$version\bin" >> ${env:GITHUB_PATH}
}

# Install XIMEA
$proc = Start-Process -FilePath C:\XIMEA_API_Installer.exe -ArgumentList "/S /SecXiApi=ON /SecDriversIeee1394=OFF /SecDriversUsb=OFF /SecDriversPCIe=OFF /SecXiApiNET=OFF /SecNETCore22=OFF /SecNETCore31=OFF /SecGenTL=OFF /SecPython=OFF /SecxiCamTool=OFF /SecxiCamToolExamples=OFF /SecxiCOP=OFF /SecXiLib=OFF /SecExamples=OFF /SecxiXiapiDNG=OFF" -PassThru # https://www.ximea.com/support/wiki/apis/Unattended_Installation_XIMEA_API_Installer, https://nsis.sourceforge.io/Can_I_decompile_an_existing_installer
$proc | Wait-Process -Timeout 300 -ErrorAction SilentlyContinue -ErrorVariable timeouted
if ($timeouted) {
  # terminate the process
  $proc | kill
  throw "XIMEA install timeout"
}

# Install NDI
# TODO: NDI installer opens a manual in a browser and doesn't end, thus StartProcess with -Wait
# waits infinitely. Therefore, there is a hack with Sleep (and not removint the installer)
#Start-Process -FilePath "C:\ndi.exe" -ArgumentList "/VERYSILENT" -Wait -NoNewWindow
Start-Process -FilePath "C:\ndi.exe" -ArgumentList "/VERYSILENT"
Sleep 10
$attempts = 0
while ($attempts -ne 10) {
  try {
    $sdk=(dir "C:\Program Files\NDI" -Filter *SDK -Name -ErrorAction Stop)
    break
  } catch [System.Exception] { # not (yet?) ready -> sleep some more time
    Sleep 30
    $attempts++
  }
}
while ($attempts -eq 10) {
  throw "Maximum amount of NDI install retries reached!"
}
echo "C:\Program Files\NDI\$sdk\Bin\x64" >> ${env:GITHUB_PATH}
#Remove-Item C:\ndi.exe

#Install vc redist 2010
$url="https://download.microsoft.com/download/1/6/5/165255E7-1014-4D0A-B094-B6A430A6BFFC/vcredist_x64.exe"
Invoke-WebRequest $url -OutFile vcredist_x64.exe
Start-Process -FilePath "vcredist_x64.exe" -ArgumentList "/install /quiet /norestart" -Wait -NoNewWindow
Remove-Item vcredist_x64.exe

# vim: set sw=2:
