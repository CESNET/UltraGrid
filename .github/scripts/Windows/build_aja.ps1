#Set-PSDebug -Trace 1

# Build AJA
if (${env:sdk_pass}) {
  $pair = "sdk:${env:sdk_pass}"
  $encodedCreds = [System.Convert]::ToBase64String([System.Text.Encoding]::ASCII.GetBytes($pair))
  $basicAuthValue = "Basic $encodedCreds"
  $Headers = @{Authorization = $basicAuthValue}
  Invoke-WebRequest -Headers $Headers https://frakira.fi.muni.cz/~xpulec/sdks/ntv2sdkwin.zip -OutFile aja.zip
  Expand-Archive -LiteralPath 'aja.zip' -DestinationPath 'C:\'
  Remove-Item aja.zip
  mv c:\ntv2sdk* c:\AJA
  cd c:\AJA
  MSBuild.exe ntv2_vs12.sln -p:PlatformToolset=v142 -p:Configuration=Release -p:Platform=x64 -t:libajantv2
}

