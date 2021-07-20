#Set-PSDebug -Trace 1

# Free some space - TODO: regular uninstall would be better
Remove-Item -Recurse "C:\Program Files (x86)\Android"
Remove-Item -Recurse "C:\Program Files (x86)\dotnet"

# Install CUDA
if (!${env:no_cuda}) {
  Invoke-WebRequest https://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_441.22_win10.exe -OutFile cuda_inst.exe
  Start-Process -FilePath "cuda_inst.exe" -ArgumentList "-s nvcc_10.2" -Wait -NoNewWindow
  Remove-Item cuda_inst.exe
  echo "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\bin" >> ${env:GITHUB_PATH}
}

# Install XIMEA
Invoke-WebRequest https://www.ximea.com/support/attachments/download/37/XIMEA_API_Installer.exe -OutFile XIMEA_API_Installer.exe
Start-Process -FilePath .\XIMEA_API_Installer.exe -ArgumentList "/S /SecXiApi=ON" -Wait
Remove-Item XIMEA_API_Installer.exe

# Install NDI
if (${env:SDK_URL} -and ${env:GITHUB_REF} -eq "refs/heads/ndi-build") {
  Invoke-WebRequest ${env:SDK_URL}/NDI_SDK.exe -OutFile C:\ndi.exe
  # TODO: NDI installer opens a manual in a browser and doesn't end, thus StartProcess with -Wait
  # waits infinitely. Therefore, there is a hack with Sleep (and not removint the installer)
  #Start-Process -FilePath "C:\ndi.exe" -ArgumentList "/VERYSILENT" -Wait -NoNewWindow
  Start-Process -FilePath "C:\ndi.exe" -ArgumentList "/VERYSILENT"
  Sleep 10
  $sdk=(dir "C:\Program Files\NDI" -Filter *SDK -Name)
  echo "C:\Program Files\NDI\$sdk\Bin\x64" >> ${env:GITHUB_PATH}
  #Remove-Item C:\ndi.exe
}

