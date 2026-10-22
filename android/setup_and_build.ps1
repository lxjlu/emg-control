# One-click setup JDK + Android SDK and build APK
# Run: powershell -ExecutionPolicy Bypass -File setup_and_build.ps1

$ErrorActionPreference = "Continue"
$toolsDir = "$PSScriptRoot\.tools"
$jdkDir = "$toolsDir\jdk-17"
$sdkDir = "$toolsDir\android-sdk"

function Download-File($url, $dest, [long]$minSize = 0) {
    if (Test-Path $dest) {
        $sz = (Get-Item $dest).Length
        if ($minSize -gt 0 -and $sz -ge $minSize) {
            Write-Host "OK: $dest ($sz bytes)" -ForegroundColor Green
            return
        } else {
            Write-Host "Incomplete $dest ($sz bytes), re-downloading..." -ForegroundColor Yellow
            Remove-Item $dest -Force
        }
    }
    Write-Host "Downloading: $url" -ForegroundColor Cyan
    try {
        $curl = (Get-Command curl.exe -ErrorAction SilentlyContinue).Source
        if ($curl) {
            & $curl -L -o "$dest" "$url" --progress-bar
            if ($LASTEXITCODE -ne 0) { throw "curl failed" }
        } else {
            $ProgressPreference = 'SilentlyContinue'
            Invoke-WebRequest -Uri $url -OutFile $dest -UseBasicParsing
        }
        Write-Host "Done: $dest ($((Get-Item $dest).Length) bytes)" -ForegroundColor Green
    } catch {
        if (Test-Path $dest) { Remove-Item $dest -Force }
        throw "Download failed: $_"
    }
}

function Extract-Zip($zip, $dest) {
    if (Test-Path $dest) {
        Write-Host "Already extracted: $dest" -ForegroundColor Green
        return
    }
    Write-Host "Extracting: $zip -> $dest" -ForegroundColor Cyan
    Expand-Archive -Path $zip -DestinationPath $dest -Force
    Write-Host "Extracted" -ForegroundColor Green
}

New-Item -ItemType Directory -Force -Path $toolsDir | Out-Null

# JDK 17 (~190MB)
$jdkZip = "$toolsDir\jdk17.zip"
Download-File "https://cdn.azul.com/zulu/bin/zulu17.52.17-ca-jdk17.0.12-win_x64.zip" $jdkZip 100000000

$tmpJdk = "$toolsDir\jdk-tmp"
Extract-Zip $jdkZip $tmpJdk
$extractedJdk = Get-ChildItem $tmpJdk -Directory | Select-Object -First 1
if ($extractedJdk -and -not (Test-Path $jdkDir)) {
    Move-Item $extractedJdk.FullName $jdkDir
    Remove-Item $tmpJdk -Recurse -Force -ErrorAction SilentlyContinue
}

# Find java.exe
$javaExe = "$jdkDir\bin\java.exe"
if (-not (Test-Path $javaExe)) {
    $nestedJdk = Get-ChildItem $jdkDir -Directory | Select-Object -First 1
    if ($nestedJdk -and (Test-Path "$($nestedJdk.FullName)\bin\java.exe")) {
        $nestedJava = "$($nestedJdk.FullName)\bin\java.exe"
    } else {
        throw "java.exe not found. Delete .tools folder and retry."
    }
} else {
    $nestedJava = $javaExe
}

# Android SDK (~150MB)
$sdkZip = "$toolsDir\sdk-tools.zip"
Download-File "https://dl.google.com/android/repository/commandlinetools-win-11076708_latest.zip" $sdkZip 50000000

$tmpSdk = "$toolsDir\sdk-tmp"
Extract-Zip $sdkZip $tmpSdk
if (-not (Test-Path "$sdkDir\cmdline-tools\latest")) {
    New-Item -ItemType Directory -Force -Path "$sdkDir\cmdline-tools\latest" | Out-Null
    Copy-Item "$tmpSdk\cmdline-tools\*" "$sdkDir\cmdline-tools\latest" -Recurse -Force
    Remove-Item $tmpSdk -Recurse -Force -ErrorAction SilentlyContinue
}

# Env
$env:JAVA_HOME = $jdkDir
$env:ANDROID_HOME = $sdkDir
$env:PATH = "$jdkDir\bin;$sdkDir\cmdline-tools\latest\bin;$env:PATH"

Write-Host "`n=== ENV ===" -ForegroundColor Yellow
Write-Host "JAVA_HOME = $env:JAVA_HOME"
Write-Host "ANDROID_HOME = $env:ANDROID_HOME"

# Check Java
Write-Host "`nJava version:" -ForegroundColor Yellow
try { & $nestedJava -version 2>&1 | ForEach-Object { Write-Host $_ } } catch { }

# Install SDK components
Write-Host "`nInstalling Android SDK components..." -ForegroundColor Yellow
$licenseDir = "$sdkDir\licenses"
New-Item -ItemType Directory -Force -Path $licenseDir | Out-Null
"24333f8a63b6825ea9c5514f83c2829b004d1fee" | Set-Content "$licenseDir\android-sdk-license" -Encoding ASCII
"84831b9409646a918e30573bab4c9c91346d8abd" | Add-Content "$licenseDir\android-sdk-license" -Encoding ASCII
"d975f751698a77b662f1254ddbeed3901e976f5a" | Add-Content "$licenseDir\android-sdk-license" -Encoding ASCII

$sdkManager = "$sdkDir\cmdline-tools\latest\bin\sdkmanager.bat"
& $sdkManager "platforms;android-34" "build-tools;34.0.0" 2>&1 | ForEach-Object { Write-Host $_ }

# Build APK
Write-Host "`n=== Building APK ===" -ForegroundColor Yellow
Set-Location $PSScriptRoot
& .\gradlew.bat :app:assembleDebug

if ($LASTEXITCODE -eq 0) {
    $apkPath = "$PSScriptRoot\app\build\outputs\apk\debug\app-debug.apk"
    Write-Host "`n=== SUCCESS ===" -ForegroundColor Green
    Write-Host "APK: $apkPath"
    Write-Host "`nInstall to phone (USB debug required):"
    Write-Host "  .tools\android-sdk\platform-tools\adb.exe install -r `"$apkPath`""
} else {
    Write-Host "`n=== BUILD FAILED ===" -ForegroundColor Red
}
