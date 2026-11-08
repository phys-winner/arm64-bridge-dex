// SPDX-License-Identifier: GPL-3.0-or-later
// Deus Ex: The Fall - arm64 bridge entry point.
//
// The reference bridge (Galaxy on Fire 2) was driven from Java: the game's Java
// layer owned the render loop and called down through JNI. Deus Ex is the other
// shape - com.unity3d.player.UnityPlayerNativeActivity extends
// android.app.NativeActivity, so Android loads libmain.so and calls
// ANativeActivity_onCreate, and the *guest* owns the loop.
//
// This file is therefore built as a 64-bit libmain.so that stands in for the
// guest's. It exports what the platform and Unity's Java layer look for, boots
// the Dynarmic host, and forwards into the 32-bit images.
//
// Phase 1 scope: load and relocate the guest libraries and hand control to the
// guest entry point. Everything below that (guest ANativeActivity marshalling,
// input, GL, audio) is phases 2-5.

#include <android/log.h>
#include <android/native_activity.h>
#include <jni.h>

#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <exception>
#include <stdexcept>
#include <unistd.h>
#include <unwind.h>
#include <dlfcn.h>

#include "dynarmic_host.h"
#include "elf_loader.h"
#include "native_activity_bridge.h"
#include "thunk_manager.h"
#include "apk_reader.h"
#include "unity_jni.h"

#define LOG_TAG "DXTF_Bridge"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {

// diag: what the guest's garbage collector is told about itself. Stats are cheap
// - a couple of lines per collection. Stress caps the Boehm heap so collections
// come constantly, which is how a race inside stop-the-world gets caught in a
// minute instead of once in six runs. Both off for a normal build.
constexpr bool kGuestGcStats = false;
constexpr bool kGuestGcStress = false;

JavaVM* g_jvm = nullptr;
std::unique_ptr<DynarmicHost> g_host;
std::mutex g_boot_mutex;
std::string g_guest_lib_dir;

// The guest images, in dependency order. libunity.so lists libmono and libmain in
// its DT_NEEDED, so both must be relocated before it. The Wwise plugin is opened
// by the engine through dlopen() but is loaded up front - the dlopen thunk hands
// back an already-resolved handle rather than parsing an ELF mid-run.
const char* const kGuestLibraries[] = {
    "libmain.so",
    "libmono.so",
    "libunity.so",
    "libAkSoundEnginePlugin.so",
};

// The 32-bit originals ship as assets, not as libraries: lib/ holds the arm64
// bridge and nothing else, because the platform picks the process ABI from which
// lib/<abi>/ directories exist and one leftover armeabi-v7a entry would start the
// game 32-bit with the bridge never running.
//
// Nothing unpacks them. They are stored uncompressed in the package and the ELF
// loader preads each one straight out of it - a guest image is copied into the
// arena and relocated either way, so a file on disk would buy nothing but 17 MiB
// of private storage and a slower first launch.
std::string GuestPath(const char* soname) {
    return g_guest_lib_dir + "/" + soname;
}

// NativeActivity.onCreate() loads libmain.so and calls ANativeActivity_onCreate
// from inside the super() call at the top of UnityPlayerNativeActivity.onCreate,
// which is long before UnityPlayer's constructor reaches NativeLoader.load. So
// the directory has to be read off the activity rather than waited for.
// Reads one String field off the activity's ApplicationInfo.
std::string QueryApplicationInfoField(ANativeActivity* activity, const char* field_name) {
    if (!activity || !activity->env || !activity->clazz) return {};
    JNIEnv* env = activity->env;

    jclass ctx_cls = env->GetObjectClass(activity->clazz);
    jmethodID get_info = ctx_cls ? env->GetMethodID(ctx_cls, "getApplicationInfo",
                                                    "()Landroid/content/pm/ApplicationInfo;")
                                 : nullptr;
    jobject info = get_info ? env->CallObjectMethod(activity->clazz, get_info) : nullptr;
    jclass info_cls = info ? env->GetObjectClass(info) : nullptr;
    jfieldID field = info_cls ? env->GetFieldID(info_cls, field_name, "Ljava/lang/String;") : nullptr;
    auto path = field ? static_cast<jstring>(env->GetObjectField(info, field)) : nullptr;

    std::string result;
    if (path) {
        if (const char* utf = env->GetStringUTFChars(path, nullptr)) {
            result = utf;
            env->ReleaseStringUTFChars(path, utf);
        }
    }
    if (env->ExceptionCheck()) {
        env->ExceptionDescribe();
        env->ExceptionClear();
    }
    return result;
}

bool BootGuest() {
    std::lock_guard<std::mutex> lock(g_boot_mutex);
    if (g_host) return true;

    if (g_guest_lib_dir.empty()) {
        LOGE("guest images were never unpacked - cannot boot");
        return false;
    }

    // Mono compiles IL to ARM32 at runtime. By default it relies on hardware traps
    // for null checks and unwinds them out of a ucontext, which would mean
    // synthesising a 32-bit signal frame inside the JIT. libmono.so understands
    // MONO_DEBUG=explicit-null-checks, so the traps are turned off instead.
    setenv("MONO_DEBUG", "explicit-null-checks", 1);

    // Our guest signals are cooperative: a suspend request can race a Java-to-
    // guest callback between ClearHalt() and Jit::Run(). Boehm already has a
    // retry protocol for exactly a lost suspend acknowledgement, but this old
    // Android build leaves it opt-in. Enable it so a rare miss costs roughly the
    // collector's 100 ms retry interval instead of stalling the splash screen.
    setenv("GC_RETRY_SIGNALS", "1", 1);

    // diag: Boehm reads both of these with getenv at start-up, and everything it
    // prints goes to fd 2, which the write thunk now puts in the log. Stats say
    // whether a collection was running when the guest gave up; the heap cap turns
    // "a collection now and then" into "a collection every few frames", which is
    // the only way to hunt a race that lives in stop-the-world. A cap this small
    // will eventually make Boehm shout "Out of Memory!" - that message is itself
    // the answer to "was it that, or was it the race".
    if (kGuestGcStats) setenv("GC_PRINT_STATS", "1", 1);
    if (kGuestGcStress) setenv("GC_MAXIMUM_HEAP_SIZE", "67108864", 1);

    auto host = std::make_unique<DynarmicHost>();
    // The handle table allocates its ids from the guest heap, so it has to know the
    // host before Initialize() lets any guest code run its static initialisers.
    BindGuestBridge(*host);
    if (!host->Initialize()) {
        LOGE("DynarmicHost::Initialize failed");
        return false;
    }

    std::vector<DynarmicHost::GuestLibrary> libs;
    for (const char* soname : kGuestLibraries) {
        libs.push_back({soname, GuestPath(soname)});
    }

    if (!host->LoadGuestLibraries(libs)) {
        LOGE("failed to load guest libraries");
        return false;
    }

    // Interface calls used to dispatch to the wrong method here, and the note that
    // used to sit in this spot blamed Mono's IMT thunk. It was the bridge: `qsort`
    // was registered as an empty lambda, and Mono sorts the entries of an IMT slot
    // with it before building the binary-search thunk that dispatches them. On an
    // unsorted list an entry lands in the wrong half, no comparison ever matches
    // it, and the call takes the thunk's default entry - which is why a slot with
    // one entry always worked and a slot with two did not. See thunks_libc_extra.cpp.
    //
    // Worth keeping from that hunt: `mono_use_imt` is an exported global and
    // clearing it is the documented way to turn IMT off, but this build's non-IMT
    // path does not work - with it cleared the process stops three frames in, every
    // time. So the thunk had to be right, not avoided.

    g_host = std::move(host);
    LOGI("guest booted: %zu libraries", libs.size());
    return true;
}

}  // namespace

JavaVM* GetHostJavaVM() { return g_jvm; }

DynarmicHost* GetDynarmicHost() { return g_host.get(); }

// Same object, separate name so guest_jnienv.cpp does not have to pull in this
// translation unit's other externs.
DynarmicHost* GetDynarmicHostForJni() { return g_host.get(); }

extern "C" {

JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* /*reserved*/) {
    g_jvm = vm;
    // Without this an abort inside the bridge reaches logcat as the single word
    // "terminating", with a backtrace the unwinder cannot walk past libc++abi.
    std::set_terminate([] {
        if (auto e = std::current_exception()) {
            try {
                std::rethrow_exception(e);
            } catch (const std::exception& ex) {
                LOGE("terminate: %s", ex.what());
            } catch (...) {
                LOGE("terminate: unknown exception type");
            }
        } else {
            LOGE("terminate: called with no exception in flight");
        }
        // libc++abi's own abort path is what the platform unwinder trips over, so
        // the backtrace is taken here, while our frames are still on the stack.
        struct Frames {
            void* pc[32];
            int n = 0;
        } frames;
        _Unwind_Backtrace(
            [](_Unwind_Context* ctx, void* arg) -> _Unwind_Reason_Code {
                auto* f = static_cast<Frames*>(arg);
                uintptr_t ip = _Unwind_GetIP(ctx);
                if (ip && f->n < 32) f->pc[f->n++] = reinterpret_cast<void*>(ip);
                return f->n >= 32 ? _URC_END_OF_STACK : _URC_NO_REASON;
            },
            &frames);
        for (int i = 0; i < frames.n; ++i) {
            Dl_info info{};
            if (dladdr(frames.pc[i], &info) && info.dli_fname) {
                LOGE("terminate:  #%02d %p %s+0x%lx", i, frames.pc[i], info.dli_fname,
                     static_cast<unsigned long>(reinterpret_cast<uintptr_t>(frames.pc[i]) -
                                                reinterpret_cast<uintptr_t>(info.dli_fbase)));
            } else {
                LOGE("terminate:  #%02d %p", i, frames.pc[i]);
            }
        }
        _exit(1);
    });
    LOGI("JNI_OnLoad");
    return JNI_VERSION_1_6;
}

// UnityPlayer.smali:906 -> NativeLoader.load, with the app's nativeLibraryDir.
// That directory holds only the arm64 bridge now, so the argument is ignored -
// the guest images come out of the package (see ANativeActivity_onCreate, which
// always runs first because NativeActivity loads "main" from inside super()).
//
// The forward into the guest's own load() matters: that is where libunity.so gets
// opened and its second RegisterNatives pass runs, which is what supplies
// nativeFile, initJni and the rest of UnityPlayer's table.
JNIEXPORT jboolean JNICALL
Java_com_unity3d_player_NativeLoader_load(JNIEnv* env, jclass clazz, jstring lib_dir) {
    LOGI("NativeLoader.load");
    if (!BootGuest()) return JNI_FALSE;

    uint32_t guest_load = GuestNative("load");
    if (!guest_load) {
        LOGE("guest libmain never registered NativeLoader.load");
        return JNI_FALSE;
    }
    uint32_t ok = g_host->RunFunction(
        guest_load, {g_host->GetThunkManager().GetGuestJNIEnv(),
                     GuestRefForHostObject(env, clazz), GuestRefForHostObject(env, lib_dir)});
    LOGI("guest NativeLoader.load -> %u", ok);
    return ok ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jboolean JNICALL
Java_com_unity3d_player_NativeLoader_unload(JNIEnv* env, jclass clazz) {
    LOGI("NativeLoader.unload");
    uint32_t guest_unload = g_host ? GuestNative("unload") : 0;
    if (!guest_unload) return JNI_TRUE;
    return g_host->RunFunction(guest_unload, {g_host->GetThunkManager().GetGuestJNIEnv(),
                                              GuestRefForHostObject(env, clazz)})
               ? JNI_TRUE
               : JNI_FALSE;
}

// Platform entry point for a NativeActivity. Phase 2 turns this into a call into
// the guest's own ANativeActivity_onCreate with a guest-side activity struct;
// for now it only proves the bridge is reachable and that the guest images are
// mapped by the time the platform gets here.
JNIEXPORT void ANativeActivity_onCreate(ANativeActivity* activity, void* saved_state,
                                        size_t saved_state_size) {
    LOGI("ANativeActivity_onCreate(activity=%p, saved=%p/%zu)", activity, saved_state,
         saved_state_size);

    // The whole guest boots from here, and everything it does through JNI needs a
    // host JNIEnv - which is reached through this VM. Waiting for our own JNI_OnLoad
    // to hand it over is too late by a wide margin: System.loadLibrary runs after
    // onCreate, so libunity's JNI_OnLoad had already asked for
    // com/unity3d/player/ReflectionHelper, been told there is no VM, and given up on
    // it. Unity's whole AndroidJavaObject layer is built on that class, so
    // `new AndroidJavaClass("android.os.Build")` threw, and the throw came out of
    // Globals.Awake() before it reached LoadMissions() - leaving the mission table
    // empty and every screen that reads it raising NullReferenceException per frame.
    if (activity && activity->vm) g_jvm = activity->vm;

    // Mono's assembly root is a package-relative path, so the reader needs to know
    // which package before the first assembly is opened. The guest images are read
    // out of the same package, by path.
    std::string source_dir = QueryApplicationInfoField(activity, "sourceDir");
    ApkSetPrimaryPackage(source_dir.c_str());
    if (!source_dir.empty()) {
        g_guest_lib_dir = source_dir + "/assets/guestlib";
        LOGI("guest images in %s", g_guest_lib_dir.c_str());
    }

    // Unity points Mono at "assets/bin/Data/Managed" inside the package and relies
    // on its own file layer to resolve that; Mono's own search ends up with an empty
    // path and it exits before loading mscorlib. Unpacking the assemblies once and
    // handing Mono a real directory through MONO_PATH sidesteps the whole question -
    // that path is the first one mono_assembly_load_corlib looks in.
    if (activity && activity->internalDataPath) {
        std::string managed = std::string(activity->internalDataPath) + "/managed";
        if (ApkExtractPrefix("assets/bin/Data/Managed/", managed.c_str()) > 0) {
            setenv("MONO_PATH", managed.c_str(), 1);
            LOGI("MONO_PATH=%s", managed.c_str());
        }
    }

    if (!BootGuest()) {
        LOGE("guest not booted - refusing to continue");
        return;
    }

    if (!StartGuestActivity(*g_host, activity, saved_state, saved_state_size)) {
        LOGE("guest activity did not start");
    }
}

}  // extern "C"
