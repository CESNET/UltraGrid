#Set-PSDebug -Trace 1

# Build AJA
if (${env:SDK_URL}) {
  Invoke-WebRequest ${env:SDK_URL}/ntv2sdkwin.zip -OutFile aja.zip
  Expand-Archive -LiteralPath 'aja.zip' -DestinationPath 'C:\'
  Remove-Item aja.zip
  if (Test-Path c:\AJA) {
     Remove-Item -Recurse c:\AJA
  }
  mv c:\ntv2sdk* c:\AJA
  cd c:\AJA
  MSBuild.exe ntv2_vs12.sln -p:PlatformToolset=v142 -p:Configuration=Release -p:Platform=x64 -t:libajantv2:Rebuild
}

