# ARM64 Bridge for Deus Ex: The Fall

An experimental ARM32-to-ARM64 compatibility bridge for the Android version of **Deus Ex: The Fall**. It builds a native ARM64 `libmain.so` that executes the original ARMv7 game libraries through Dynarmic and forwards their Android APIs to the host process.

This repository contains bridge source and build/inspection scripts only. Game binaries, assets, APK/OBB files, managed assemblies, signing keys, SDKs and third-party source trees are not distributed.

## How it works

- Loads and relocates the original ELF32 guest libraries into a guest address space.
- Uses Dynarmic's A32 JIT backend to execute ARM/Thumb code on ARM64.
- Bridges Unity NativeActivity and JNI calls, graphics (EGL/OpenGL ES), OpenSL ES audio, Android services, libc, threading and signals.
- Reads guest libraries directly from uncompressed APK assets and provides access to packaged Unity data.
- Links the bridge for 16 KiB page compatibility. This alone does not establish compatibility with every Android version or device.

The code is specific to this game's old Unity runtime and library interfaces; it is not a general Android application translator. Some APIs are partial implementations or stubs. Gameplay, performance and stability require device testing.

## Requirements

- A legally obtained copy of Deus Ex: The Fall for Android. The local project targets game version `0.0.36`.
- Android NDK r27c (`27.2.12479018`), CMake 3.22 or newer, Ninja and Git.
- An external Boost source/header installation, version 1.83 or newer (1.83 matches the original build).
- Internet access on the first CMake configure to fetch Dynarmic and its recursive submodules.
- For packaging: Bash (Git Bash on Windows), Python 3, Java, apktool, Android Build Tools with 16 KiB-aware zipalign, apksigner, and your own APK signing key/certificate.

## Dependencies

Boost is not vendored: set `BOOST_ROOT` to the external directory containing `boost/version.hpp`. No compiled Boost libraries are needed. Get it from [Boost](https://www.boost.org/releases/).

CMake fetches [Dynarmic](https://github.com/azahar-emu/dynarmic) at commit `e77b1ba0b7da7cbe93021b01a663acfe7c4dd516`. Its dependency submodules (including fmt, mcl, oaknut and robin-map) remain upstream dependencies and are downloaded into the ignored build tree. The cross build uses those upstream copies rather than host libraries. CMake's `FETCHCONTENT_SOURCE_DIR_DYNARMIC` can point to an external recursive checkout for offline builds.

Dependencies retain their own licenses: Boost uses BSL-1.0, and Dynarmic uses 0BSD; see their upstream license files and those of their submodules.

## Build the bridge

On Windows, set `ANDROID_SDK_ROOT` and `BOOST_ROOT` in your local environment, then run:

```bat
arm64_bridge\build_arm64.bat
```

The script defaults to SDK CMake `3.22.1` and NDK `27.2.12479018`; set `CMAKE_VERSION` or `NDK_VERSION` to override them. It works from any current directory and stops if configuration fails.

On other hosts, set `ANDROID_NDK_HOME` and `BOOST_ROOT`, install CMake/Ninja on PATH, then run:

```sh
cmake -S arm64_bridge -B arm64_bridge/build_arm64 -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake" \
  -DBOOST_ROOT="$BOOST_ROOT" \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-26 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build arm64_bridge/build_arm64
```

Output: `arm64_bridge/build_arm64/libmain.so`. This project targets Android API 26 or newer and ARM64 hosts.

## Prepare game files and package

`build.sh` repackages an **already prepared apktool project**; it does not convert a stock APK into a fully patched game. The managed-assembly patcher, injected managed plugin sources and OBB preparation tools referenced by the earlier local pipeline are absent from this checkout. Reproducing those preparation steps from a stock game is outside the supplied source set.

Supply your prepared tree as `app/` or set `APP_DIR`. It must contain:

```text
app/
  AndroidManifest.xml
  apktool.yml
  assets/
    guestlib/
      libmain.so
      libmono.so
      libunity.so
      libAkSoundEnginePlugin.so
    bin/Data/...
  lib/arm64-v8a/    (bridge staged here by build.sh)
  res/...
  smali/...
```

Use the original **32-bit** libraries under `assets/guestlib/`. Keep only `arm64-v8a` in APK `lib/`; additional ABI directories cause packaging to fail validation. The manifest's NativeActivity must select `main` as `android.app.lib_name`. Include the game's prepared Unity data and managed assemblies, and merge any required OBB content into assets beforehand. In `apktool.yml`, store `so` uncompressed (`doNotCompress`); streaming audio banks may also require `bnk` and `wem` entries.

Set these environment variables to your local tools and signing files:

| Variable | Input |
| --- | --- |
| `APKTOOL_JAR` | apktool JAR (local project used 3.0.3) |
| `APKSIGNER_JAR` | Android Build Tools apksigner JAR |
| `ZIPALIGN` | Android Build Tools zipalign executable |
| `LLVM_STRIP` | NDK llvm-strip executable |
| `SIGNING_KEY` | Your APK signing key in PKCS#8 `.pk8` format |
| `SIGNING_CERT` | Its X.509 `.x509.pem` certificate |
| `APP_DIR` | Optional prepared tree; defaults to `app/` |
| `BRIDGE_LIB` | Optional built bridge; defaults to the output above |

```sh
bash ./build.sh
```

The script validates inputs, stages the stripped bridge, rebuilds, validates ABI and guest compression, aligns native libraries for 16 KiB pages, signs and verifies the APK. Output: `out/DeusEx_TheFall_arm64.apk`. It modifies the prepared tree's ARM64 bridge file. Keep keys outside the repository.

## Inspect guest imports

```sh
python arm64_bridge/check_symbols.py game-library.so
```

The script compares ELF32 dynamic imports with literal `RegisterThunk` registrations in the bridge. Its missing-import list is a diagnostic aid: guest-to-guest exports and indirect registrations can make it overreport missing host functions.

## License

The bridge and scripts are licensed under **GNU GPL version 3 or later** (`GPL-3.0-or-later`); see [LICENSE](LICENSE). Distributed modified/combined covered code must follow that license, including its corresponding-source requirements. Third-party dependencies retain their licenses.

No rights to the game, its assets, Unity, or other proprietary components are granted. This is an independent project and is not affiliated with or endorsed by the game's publishers or developers.
