$ErrorActionPreference = "Stop"
$scriptPath = $MyInvocation.MyCommand.Path
$scriptDir = Split-Path $scriptPath
cd $scriptDir
rm UltraGrid-continuous*

Invoke-WebRequest https://github.com/CESNET/UltraGrid/releases/download/continuous/UltraGrid-continuous-win64.zip -OutFile UltraGrid-continuous-win64.zip
$downloadExtractDir = "UltraGrid-continuous-latest-win64"
Expand-Archive -LiteralPath UltraGrid-continuous-win64.zip -DestinationPath $downloadExtractDir
$currentName = (Get-Location | Split-Path -Leaf)
$downloadedName = (Get-ChildItem $downloadExtractDir).Name

Move-Item $downloadExtractDir/* ../$downloadedName-new
Set-Location ..
if ($currentName -eq $downloadedName) {
        if (Test-Path -Path $currentName-bkp) {
                Remove-Item -Recurse $currentName-bkp
        }
        Move-Item $currentName $currentName-bkp
}

Move-Item $downloadedName-new $downloadedName
Set-Location $downloadedName

Write-Host "Updated UltraGrid in $downloadedName. You can delete $currentName-bkp if everything is OK."

