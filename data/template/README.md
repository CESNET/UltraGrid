files from this subdirectories will be deployed automatically:

1. bin/* - will be copied to $(bindir) and also to macOS bundle
Contents/MacOS subdir
2. macOS-bundle/* - will be copied to macOS bundle
3. macOS-legacy/README.html - will be copied to macOS dmg (at the same
level as application bundle). Contains hint how to allow unsigned UG.
