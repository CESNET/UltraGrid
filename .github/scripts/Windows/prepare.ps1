Set-PSDebug -Trace 1

# Install MSYS2
choco install --no-progress msys2 --params "/NoUpdate /InstallDir:C:\msys64"
echo "::set-env name=MSYS2_PATH_TYPE::inherit" # MSYS2 inherits PATH from Windows

# Install CUDA
Invoke-WebRequest https://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_441.22_win10.exe -OutFile cuda_inst.exe
Start-Process -FilePath "cuda_inst.exe" -ArgumentList "-s nvcc_10.2" -Wait -NoNewWindow
echo "::add-path::C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\bin"

# Build AJA
if (${env:sdk_pass}) {
  $pair = "sdk:${env:sdk_pass}"
  $encodedCreds = [System.Convert]::ToBase64String([System.Text.Encoding]::ASCII.GetBytes($pair))
  $basicAuthValue = "Basic $encodedCreds"
  $Headers = @{Authorization = $basicAuthValue}
  Invoke-WebRequest -Headers $Headers https://frakira.fi.muni.cz/~xpulec/sdks/ntv2sdkwin.zip -OutFile aja.zip
  Expand-Archive -LiteralPath 'aja.zip' -DestinationPath 'C:'
  mv c:\ntv2sdk* c:\AJA
  cd c:\AJA
  MSBuild.exe ntv2_vs12.sln -p:PlatformToolset=v142 -p:Configuration=Release -p:Platform=x64 -t:libajantv2
}

# Install XIMEA
if (${env:sdk_pass}) {
  $pair = "sdk:${env:sdk_pass}"
  $encodedCreds = [System.Convert]::ToBase64String([System.Text.Encoding]::ASCII.GetBytes($pair))
  $basicAuthValue = "Basic $encodedCreds"
  $Headers = @{Authorization = $basicAuthValue}
  Invoke-WebRequest -Headers $Headers https://frakira.fi.muni.cz/~xpulec/sdks/ximea.zip -OutFile ximea.zip
  Expand-Archive -LiteralPath 'ximea.zip' -DestinationPath 'C:'
}
