
& ./build.ps1

Write-Output "==== 128"
& ./nsys_easy.ps1 target/release/voronoi.exe 4096 128

Write-Output "==== 512"
& ./nsys_easy.ps1 target/release/voronoi.exe 4096 512

Write-Output "==== 1024"
& ./nsys_easy.ps1 target/release/voronoi.exe 4096 1024

Write-Output "==== 2048"
& ./nsys_easy.ps1 target/release/voronoi.exe 4096 2048