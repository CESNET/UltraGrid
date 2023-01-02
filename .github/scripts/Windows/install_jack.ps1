#Set-PSDebug -Trace 1

if (Test-Path 'C:\Program Files\JACK2') {
  Remove-Item -Recurse 'C:\Program Files\JACK2'
}

# latest release grab inspired by https://gist.github.com/MarkTiedemann/c0adc1701f3f5c215fc2c2d5b1d5efd3
$repo = "jackaudio/jack2-releases"
$releases = "https://api.github.com/repos/$repo/releases"
$tag = (Invoke-WebRequest $releases -Headers @{Authorization = "token ${Env:GITHUB_TOKEN}"} | ConvertFrom-Json)[0].tag_name
$download = "https://github.com/$repo/releases/download/$tag/jack2-win64-$tag.exe"
Invoke-WebRequest $download -Headers @{Authorization = "token ${Env:GITHUB_TOKEN}"} -o jack2.exe
Start-Process -FilePath '.\jack2.exe' -ArgumentList '/SILENT' -Wait -NoNewWindow

# The lib is moved to the JACK library for 2 reasons:
# 1. it will be cached here
# 2. if it were in a Windows directory, it won't be bundled with UltraGrid
#    (also make sure to remove from the Windows directory)
New-Item -Type Directory 'C:\Program Files\JACK2\bin'
Move-Item 'C:\Windows\libjack64.dll' 'C:\Program Files\JACK2\bin'

