cmake_minimum_required(VERSION 3.22)
project(dxtf_arm64_bridge LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# Boost is supplied externally; pass -DBOOST_ROOT=<boost-root> when configuring.
set(Boost_NO_SYSTEM_PATHS ON)
set(Boost_NO_BOOST_CMAKE ON)
# Boost headers are architecture-independent and live outside the NDK sysroot.
set(_bridge_include_mode "${CMAKE_FIND_ROOT_PATH_MODE_INCLUDE}")
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE NEVER)
find_package(Boost 1.83 REQUIRED)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE "${_bridge_include_mode}")
unset(_bridge_include_mode)
include_directories(SYSTEM ${Boost_INCLUDE_DIRS})
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Configure Dynarmic.
# NOTE: the real option is DYNARMIC_FRONTENDS (a list), not DYNARMIC_ENABLE_A32/A64 --
# the old names were silently ignored, so A64 was being built too.
set(DYNARMIC_FRONTENDS "A32" CACHE STRING "" FORCE)
set(DYNARMIC_TESTS OFF CACHE BOOL "" FORCE)
set(DYNARMIC_WARNINGS_AS_ERRORS OFF CACHE BOOL "" FORCE)
set(DYNARMIC_USE_BUNDLED_EXTERNALS ON CACHE BOOL "" FORCE)  # cross-compiling: never pick up host libs

message(STATUS "Fetching Dynarmic from GitHub...")
FetchContent_Declare(
    dynarmic
    GIT_REPOSITORY https://github.com/azahar-emu/dynarmic.git
    GIT_TAG        e77b1ba0b7da7cbe93021b01a663acfe7c4dd516
)
FetchContent_MakeAvailable(dynarmic)

set(BRIDGE_SOURCES
    src/elf_loader.cpp
    src/dynarmic_host.cpp
    src/thunk_manager.cpp
    src/thunks_gl.cpp
    src/thunks_libc.cpp
    src/thunks_libc_extra.cpp
    src/thunks_zlib.cpp
    src/guest_signals.cpp
    src/guest_cp15.cpp
    src/thunks_egl.cpp
    src/thunks_opensles.cpp
    src/apk_reader.cpp
    src/thunks_android.cpp
    src/bridge_config.cpp
    src/guest_jnienv.cpp
    src/unity_natives.cpp
    src/unity_jni.cpp
    src/native_activity_bridge.cpp
    src/dxtf_bridge.cpp
)

add_library(dxtf_arm64_bridge SHARED ${BRIDGE_SOURCES})

# UnityPlayerNativeActivity resolves android.app.lib_name = "main".
set_target_properties(dxtf_arm64_bridge PROPERTIES OUTPUT_NAME "main")

target_include_directories(dxtf_arm64_bridge PRIVATE
    src
    ${Boost_INCLUDE_DIRS}
)

target_link_libraries(dxtf_arm64_bridge PRIVATE
    dynarmic
    GLESv2
    EGL
    OpenSLES
    log
    android
    m
    z
)

target_compile_definitions(dxtf_arm64_bridge PRIVATE
    ANDROID
    __ANDROID__
)

# Android 15+ refuses to map shared objects that assume 4 KiB pages.
target_link_options(dxtf_arm64_bridge PRIVATE "-Wl,-z,max-page-size=16384")
