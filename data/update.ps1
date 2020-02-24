$ErrorActionPreference = "Stop"
$scriptPath = $MyInvocation.MyCommand.Path
$scriptDir = Split-Path $scriptPath
cd $scriptDir
rm UltraGrid-nightly*
Invoke-WebRequest https://github.com/CESNET/UltraGrid/releases/download/nightly/UltraGrid-nightly-win64.zip -OutFile UltraGrid-nightly-win64.zip
if ($LastExitCode -ne 0) {
        throw "Download failed"
}
$downloadExtractDir = "UltraGrid-nightly-latest-win64"
Expand-Archive -LiteralPath UltraGrid-nightly-win64.zip -DestinationPath $downloadExtractDir
$currentName = (Split-Path -Leaf Get-Location).Path
$downloadedName = (Get-ChildItem $downloadExtractDir).Name
if ($currentName -ne $downloadedName) {
        Move-Item $downloadExtractDir/* ..
        Write-Host "Downloaded ,$downloadedName removing $currentName."
        Set-Location ../$downloadedName
        Remove-Item -Recurse ../$currentName
} else {
        Remove-Item -Recurse $downloadExtractDir
        Remove-Item UltraGrid-nightly-win64.zip
}

