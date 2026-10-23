@echo off
setlocal EnableExtensions
title Build sEMG App

rem ============================================================
rem  One-click build for the Android App (Windows)
rem    build_app.bat            build only
rem    build_app.bat /install   build and install to a connected phone
rem
rem  The script is ASCII-only on purpose: Chinese text in .bat files
rem  breaks under the default GBK console code page.
rem ============================================================

set "ROOT=%~dp0"
set "APP_DIR=%ROOT%android"
set "DO_INSTALL="
if /i "%~1"=="/install" set "DO_INSTALL=1"
if /i "%~1"=="install" set "DO_INSTALL=1"

echo ==========================================
echo   Build sEMG Gesture App
echo ==========================================
echo.

if not exist "%APP_DIR%\gradlew.bat" (
    echo [ERROR] android\gradlew.bat not found.
    echo         Please run this script from the repository root.
    goto :fail
)

rem ---------- 1. locate JDK ----------
set "JDK_HOME="
call :try_jdk "%JAVA_HOME%"
call :try_jdk "%ProgramFiles%\Android\Android Studio\jbr"
call :try_jdk "%LOCALAPPDATA%\Programs\Android Studio\jbr"
call :try_jdk "%APP_DIR%\.tools\jdk-17"
call :try_jdk "%ProgramFiles%\Java\jdk-17"
call :try_jdk "%ProgramFiles%\Eclipse Adoptium\jdk-17"
for /d %%D in ("%ProgramFiles%\Java\jdk-17*") do call :try_jdk "%%D"
for /d %%D in ("%ProgramFiles%\Eclipse Adoptium\jdk-17*") do call :try_jdk "%%D"

if defined JDK_HOME goto :jdk_found

rem fallback: java on PATH (must also be 11-17)
set "JVER="
for /f "tokens=3" %%V in ('java -version 2^>^&1 ^| findstr /i "version"') do if not defined JVER set "JVER=%%~V"
if not defined JVER goto :no_jdk
set JRAW=%JVER:"=%
if "%JRAW:~0,2%"=="1." set "JRAW=%JRAW:~2%"
set "JMAJOR="
for /f "tokens=1 delims=._-+" %%M in ("%JRAW%") do set "JMAJOR=%%M"
set "JVER="
if not defined JMAJOR goto :no_jdk
if %JMAJOR% LSS 11 goto :no_jdk
if %JMAJOR% GTR 17 goto :no_jdk
echo [1/3] JDK : java from PATH (%JRAW%)
goto :jdk_done

:no_jdk
echo [ERROR] No usable JDK found.
echo         Current JAVA_HOME: %JAVA_HOME%
echo         Android Gradle Plugin 7.2 / Gradle 7.3 need Java 11-17.
echo         JDK 8 and JDK 21+ will NOT work. Please install JDK 17
echo         (Temurin / Corretto / Oracle) and set JAVA_HOME to it.
goto :fail

:jdk_found
set "JAVA_HOME=%JDK_HOME%"
echo [1/3] JDK : %JDK_HOME% (%JDK_VER%)

:jdk_done

rem ---------- 2. locate Android SDK ----------
set "SDK_DIR="
if exist "%ANDROID_HOME%\platforms" set "SDK_DIR=%ANDROID_HOME%"
if not defined SDK_DIR if exist "%ANDROID_SDK_ROOT%\platforms" set "SDK_DIR=%ANDROID_SDK_ROOT%"
if not defined SDK_DIR if exist "%LOCALAPPDATA%\Android\Sdk\platforms" set "SDK_DIR=%LOCALAPPDATA%\Android\Sdk"
if not defined SDK_DIR if exist "%APP_DIR%\.tools\android-sdk\platforms" set "SDK_DIR=%APP_DIR%\.tools\android-sdk"

if defined SDK_DIR goto :sdk_found

if exist "%APP_DIR%\local.properties" (
    echo [2/3] SDK : using sdk.dir from android\local.properties
    goto :sdk_done
)

echo [ERROR] Android SDK not found.
echo         Install Android SDK Platform 34 and set ANDROID_HOME, or create
echo         android\local.properties containing:
echo             sdk.dir=C:/Users/YOUR_NAME/AppData/Local/Android/Sdk
goto :fail

:sdk_found
set "SDK_UNIX=%SDK_DIR:\=/%"
> "%APP_DIR%\local.properties" echo sdk.dir=%SDK_UNIX%
set "ANDROID_HOME=%SDK_DIR%"
echo [2/3] SDK : %SDK_DIR%

:sdk_done

rem ---------- 3. build ----------
echo [3/3] Building. The first run downloads Gradle and dependencies...
echo.
pushd "%APP_DIR%"
call gradlew.bat :app:assembleDebug
set "CODE=%ERRORLEVEL%"
popd

if not "%CODE%"=="0" goto :fail

set "APK=%APP_DIR%\app\build\outputs\apk\debug\app-debug.apk"
if not exist "%APK%" (
    echo [ERROR] Build finished but the APK was not found.
    goto :fail
)

for %%F in ("%APK%") do set "APK_SIZE=%%~zF"
set /a APK_MB=%APK_SIZE%/1048576

echo.
echo ==========================================
echo   BUILD SUCCESSFUL
echo   APK  : %APK%
echo   Size : about %APK_MB% MB
echo ==========================================

rem ---------- optional install ----------
if not defined DO_INSTALL goto :skip_install

set "ADB="
if exist "%SDK_DIR%\platform-tools\adb.exe" set "ADB=%SDK_DIR%\platform-tools\adb.exe"
if not defined ADB for /f "delims=" %%A in ('where adb 2^>nul') do if not defined ADB set "ADB=%%A"
if not defined ADB (
    echo.
    echo [INFO] adb not found. Copy the APK to the phone and install it manually.
    goto :after_install
)

echo.
echo Installing to the connected phone...
"%ADB%" install -r "%APK%"
goto :after_install

:skip_install
echo.
echo Install to a phone:
echo   1. copy the APK to the phone and tap it, or
echo   2. run: build_app.bat /install

:after_install
echo.
pause
exit /b 0

:fail
echo.
echo Build failed. Common causes:
echo   1. JDK 17 is missing or JAVA_HOME points to another version
echo   2. Android SDK Platform 34 / Build-Tools 34.0.0 not installed
echo   3. First build needs network access to download Gradle and dependencies
echo.
pause
exit /b 1

:try_jdk
if defined JDK_HOME goto :eof
if "%~1"=="" goto :eof
if not exist "%~1\bin\java.exe" goto :eof
set "TV="
if exist "%~1\release" for /f "tokens=2 delims==" %%V in ('findstr /b /i "JAVA_VERSION" "%~1\release"') do if not defined TV set "TV=%%~V"
if not defined TV goto :eof
set TRAW=%TV:"=%
if "%TRAW:~0,2%"=="1." set "TRAW=%TRAW:~2%"
set "TMAJOR="
for /f "tokens=1 delims=._-+" %%M in ("%TRAW%") do set "TMAJOR=%%M"
if not defined TMAJOR goto :eof
if %TMAJOR% LSS 11 goto :eof
if %TMAJOR% GTR 17 goto :eof
set "JDK_HOME=%~1"
set "JDK_VER=%TRAW%"
goto :eof
