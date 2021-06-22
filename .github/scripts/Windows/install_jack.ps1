#Set-PSDebug -Trace 1

if (Test-Path 'C:\Program Files\JACK2') {
  Remove-Item -Recurse 'C:\Program Files\JACK2'
}

choco install -y --no-progress jack

# The lib is moved to the JACK library for 2 reasons:
# 1. it will be cached here
# 2. if it were in a Windows directory, it won't be bundled with UltraGrid
#    (also make sure to remove from the Windows directory)
New-Item -Type Directory 'C:\Program Files\JACK2\bin'
Move-Item 'C:\Windows\libjack64.dll' 'C:\Program Files\JACK2\bin'

