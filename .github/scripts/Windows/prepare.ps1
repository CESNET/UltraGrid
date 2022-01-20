Set-PSDebug -Trace 1

# Free some space - TODO: regular uninstall would be better
#Remove-Item -Recurse "C:\Program Files (x86)\Android"
#Remove-Item -Recurse "C:\Program Files (x86)\dotnet"

# Install CUDA
if (!${env:no_cuda}) {
  $url="https://developer.download.nvidia.com/compute/cuda/11.5.1/local_installers/cuda_11.5.1_496.13_windows.exe"
  $url -match 'cuda/(?<version>[0-9]+.[0-9]+)'
  $version=$Matches.version
  Invoke-WebRequest $url -OutFile cuda_inst.exe
  Start-Process -FilePath "cuda_inst.exe" -ArgumentList "-s cudart_$version nvcc_$version" -Wait -NoNewWindow
  Remove-Item cuda_inst.exe
  echo "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v$version\bin" >> ${env:GITHUB_PATH}
}

# Install XIMEA
$proc = Start-Process -FilePath C:\XIMEA_API_Installer.exe -ArgumentList "/S /SecXiApi=ON" -PassThru
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
try {
  $sdk=(dir "C:\Program Files\NDI" -Filter *SDK -Name -ErrorAction Stop)
} catch [System.Exception] { # not (yet?) ready -> sleep some more time
  Sleep 30
  $sdk=(dir "C:\Program Files\NDI" -Filter *SDK -Name)
}
echo "C:\Program Files\NDI\$sdk\Bin\x64" >> ${env:GITHUB_PATH}
#Remove-Item C:\ndi.exe

# vim: set sw=2:
