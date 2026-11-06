param([string]$OpenCvDir=$env:OpenCV_DIR)
$ErrorActionPreference='Stop'
if (!$OpenCvDir) { throw 'Set OpenCV_DIR to an OpenCV 3.4.0 build directory.' }
$vswhere=Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
$vs=& $vswhere -latest -products '*' -requires Microsoft.Component.MSBuild -property installationPath
if (!$vs) {throw 'Visual Studio C++ Build Tools are required.'}
$msbuild=Join-Path $vs 'MSBuild/Current/Bin/MSBuild.exe'
$env:OpenCV_DIR=$OpenCvDir
Push-Location $PSScriptRoot
try {
    # Build a separate entry point using the same library sources/toolchain.
    $project=(Get-Content golfBall.vcxproj -Raw).Replace('Include="golfBall.cpp"','Include="paper_main.cpp"')
    $project=$project.Replace('<PropertyGroup Label="UserMacros" />','<PropertyGroup Label="UserMacros" /><PropertyGroup><TargetName>hit_paper</TargetName><IntDir>$(Platform)\$(Configuration)\paper\</IntDir></PropertyGroup>')
    [IO.File]::WriteAllText((Join-Path $PSScriptRoot 'paper.generated.vcxproj'),$project)
    & $msbuild paper.generated.vcxproj /p:Configuration=Release /p:Platform=x64 /m /verbosity:quiet
    if($LASTEXITCODE) {throw "Build failed ($LASTEXITCODE)."}
} finally {Pop-Location}
