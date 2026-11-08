@echo off
setlocal
if not defined ANDROID_SDK_ROOT (
    echo Set ANDROID_SDK_ROOT to your Android SDK directory. 1>&2
    exit /b 1
)
if not defined BOOST_ROOT (
    echo Set BOOST_ROOT to an external Boost 1.83 or newer source directory. 1>&2
    exit /b 1
)
if not defined NDK_VERSION set "NDK_VERSION=27.2.12479018"
if not defined CMAKE_VERSION set "CMAKE_VERSION=3.22.1"
set "CMAKE_EXE=%ANDROID_SDK_ROOT%\cmake\%CMAKE_VERSION%\bin\cmake.exe"
set "TOOLCHAIN=%ANDROID_SDK_ROOT%\ndk\%NDK_VERSION%\build\cmake\android.toolchain.cmake"
"%CMAKE_EXE%" -S "%~dp0." -B "%~dp0build_arm64" ^
    -G Ninja ^
    "-DCMAKE_MAKE_PROGRAM=%ANDROID_SDK_ROOT%\cmake\%CMAKE_VERSION%\bin\ninja.exe" ^
    "-DCMAKE_TOOLCHAIN_FILE=%TOOLCHAIN%" ^
    "-DBOOST_ROOT=%BOOST_ROOT%" ^
    -DANDROID_ABI=arm64-v8a ^
    -DANDROID_PLATFORM=android-26 ^
    -DCMAKE_BUILD_TYPE=Release
if errorlevel 1 exit /b 1
"%CMAKE_EXE%" --build "%~dp0build_arm64" --config Release
exit /b %errorlevel%
