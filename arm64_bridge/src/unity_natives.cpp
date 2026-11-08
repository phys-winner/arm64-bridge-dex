// SPDX-License-Identifier: GPL-3.0-or-later
// Host-side JNI entry points for com.unity3d.player.
//
// libunity.so keeps its Java_com_unity3d_player_* implementations private and
// registers them from JNI_OnLoad, so nothing in the guest is reachable by name.
// The bridge therefore exports the names Java looks for, and each one forwards to
// whatever address RegisterNatives handed over (see unity_jni.cpp).
//
// Marshalling is the mirror of guest_jnienv.cpp: objects become handles over
// global refs, floats travel as raw bit patterns because the guest is softfp, and
// doubles occupy an 8-byte-aligned register or stack pair.

#include <android/log.h>
#include <jni.h>

#include <cstring>
#include <vector>
#include <map>
#include <mutex>
#include <string>
#include <unistd.h>

#include "dynarmic_host.h"
#include "native_activity_bridge.h"
#include "thunk_manager.h"
#include "unity_jni.h"

#define LOG_TAG "DXTF_Natives"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

DynarmicHost* GetDynarmicHost();

namespace {

uint32_t GuestEnv() {
    DynarmicHost* host = GetDynarmicHost();
    return host ? host->GetThunkManager().GetGuestJNIEnv() : 0;
}

uint32_t Ref(JNIEnv* env, jobject obj) {
    return GuestRefForHostObject(env, obj);
}

uint32_t Bits(jfloat f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    return bits;
}

// A double occupies two words and has to start on an even slot, so callers push a
// padding word themselves when the preceding argument leaves an odd offset.
void PushDouble(std::vector<uint32_t>& words, jdouble d) {
    uint64_t bits;
    memcpy(&bits, &d, sizeof(bits));
    words.push_back(static_cast<uint32_t>(bits));
    words.push_back(static_cast<uint32_t>(bits >> 32));
}

uint32_t Call(const char* name, std::vector<uint32_t> args) {
    DynarmicHost* host = GetDynarmicHost();
    uint32_t fn = host ? GuestNative(name) : 0;
    if (!fn) {
        LOGE("Java called '%s' but the guest never registered it", name);
        return 0;
    }
    // diag: which of these Java actually drives, and how often
    {
        static std::mutex m;
        static std::map<std::string, uint64_t> seen;
        std::lock_guard<std::mutex> lock(m);
        uint64_t& n = seen[name];
        if (++n == 1 || n % 300 == 0)
            LOGI("[diag] Java -> %s (#%llu) on tid %d", name, (unsigned long long)n, gettid());
    }
    return host->RunFunction(fn, args);
}

// Same forwarding, for guest natives that are plain exported symbols rather than
// entries RegisterNatives handed over.
uint32_t CallGuestExport(const char* symbol, std::vector<uint32_t> args) {
    DynarmicHost* host = GetDynarmicHost();
    uint32_t fn = host ? host->FindSymbol(symbol) : 0;
    if (!fn) {
        LOGE("Java called '%s' but no guest image exports it", symbol);
        return 0;
    }
    return host->RunFunction(fn, args);
}

}  // namespace

// Every entry point has the same two leading guest arguments: the guest JNIEnv and
// a handle for the receiver (an object for instance methods, a class for statics).
#define GUEST_CALL(name, ...) Call(name, {GuestEnv(), Ref(env, self), ##__VA_ARGS__})

extern "C" {

#define UNITY_PLAYER(ret, name) \
    JNIEXPORT ret JNICALL Java_com_unity3d_player_UnityPlayer_##name

UNITY_PLAYER(void, initJni)(JNIEnv* env, jobject self, jobject context) {
    GUEST_CALL("initJni", Ref(env, context));
}
UNITY_PLAYER(void, nativeFile)(JNIEnv* env, jobject self, jstring path) {
    GUEST_CALL("nativeFile", Ref(env, path));
}
UNITY_PLAYER(void, nativeDone)(JNIEnv* env, jobject self) {
    GUEST_CALL("nativeDone");
}
UNITY_PLAYER(jboolean, nativeRender)(JNIEnv* env, jobject self) {
    return GUEST_CALL("nativeRender") ? JNI_TRUE : JNI_FALSE;
}
UNITY_PLAYER(jboolean, nativePause)(JNIEnv* env, jobject self) {
    return GUEST_CALL("nativePause") ? JNI_TRUE : JNI_FALSE;
}
UNITY_PLAYER(void, nativeResume)(JNIEnv* env, jobject self) {
    GUEST_CALL("nativeResume");
}
UNITY_PLAYER(void, nativeFocusChanged)(JNIEnv* env, jobject self, jboolean focused) {
    GUEST_CALL("nativeFocusChanged", focused ? 1u : 0u);
}
UNITY_PLAYER(void, nativeResize)(JNIEnv* env, jobject self, jint w, jint h, jint fw, jint fh) {
    // diag: the size Java hands the engine, against what actually gets drawn.
    LOGI("[resize] Java -> nativeResize %dx%d fixed %dx%d", w, h, fw, fh);
    GUEST_CALL("nativeResize", static_cast<uint32_t>(w), static_cast<uint32_t>(h),
               static_cast<uint32_t>(fw), static_cast<uint32_t>(fh));
}
UNITY_PLAYER(void, nativeRecreateGfxState)(JNIEnv* env, jobject self, jobject surface) {
    LOGI("[surface] Java -> nativeRecreateGfxState(surface=%p)", (void*)surface);
    GUEST_CALL("nativeRecreateGfxState", Ref(env, surface));
}
UNITY_PLAYER(jboolean, nativeInjectEvent)(JNIEnv* env, jobject self, jobject event) {
    return GUEST_CALL("nativeInjectEvent", Ref(env, event)) ? JNI_TRUE : JNI_FALSE;
}
UNITY_PLAYER(void, nativeInitWWW)(JNIEnv* env, jobject self, jclass www) {
    GUEST_CALL("nativeInitWWW", Ref(env, www));
}
UNITY_PLAYER(jint, nativeActivityIndicatorStyle)(JNIEnv* env, jobject self) {
    return static_cast<jint>(GUEST_CALL("nativeActivityIndicatorStyle"));
}
UNITY_PLAYER(jint, nativeRequestedAA)(JNIEnv* env, jobject self) {
    return static_cast<jint>(GUEST_CALL("nativeRequestedAA"));
}
UNITY_PLAYER(jboolean, nativeRequested32bitDisplayBuffer)(JNIEnv* env, jobject self) {
    return GUEST_CALL("nativeRequested32bitDisplayBuffer") ? JNI_TRUE : JNI_FALSE;
}
UNITY_PLAYER(void, nativeSetDefaultDisplay)(JNIEnv* env, jobject self, jint display) {
    GUEST_CALL("nativeSetDefaultDisplay", static_cast<uint32_t>(display));
}
UNITY_PLAYER(void, nativeSetExtras)(JNIEnv* env, jobject self, jobject extras) {
    GUEST_CALL("nativeSetExtras", Ref(env, extras));
}
UNITY_PLAYER(void, nativeSetInputCanceled)(JNIEnv* env, jobject self, jboolean canceled) {
    GUEST_CALL("nativeSetInputCanceled", canceled ? 1u : 0u);
}
UNITY_PLAYER(void, nativeSetInputString)(JNIEnv* env, jobject self, jstring text) {
    GUEST_CALL("nativeSetInputString", Ref(env, text));
}
UNITY_PLAYER(void, nativeSetTouchDeltaY)(JNIEnv* env, jobject self, jfloat dy) {
    GUEST_CALL("nativeSetTouchDeltaY", Bits(dy));
}
UNITY_PLAYER(void, nativeSoftInputClosed)(JNIEnv* env, jobject self) {
    GUEST_CALL("nativeSoftInputClosed");
}
UNITY_PLAYER(void, nativeDeviceOrientation)(JNIEnv* env, jobject self, jint orientation) {
    GUEST_CALL("nativeDeviceOrientation", static_cast<uint32_t>(orientation));
}
UNITY_PLAYER(void, nativeForwardEventsToDalvik)(JNIEnv* env, jobject self, jboolean forward) {
    GUEST_CALL("nativeForwardEventsToDalvik", forward ? 1u : 0u);
}
UNITY_PLAYER(void, nativeVideoFrameCallback)(JNIEnv* env, jobject self, jint id, jbyteArray data,
                                             jint offset, jint size) {
    GUEST_CALL("nativeVideoFrameCallback", static_cast<uint32_t>(id), Ref(env, data),
               static_cast<uint32_t>(offset), static_cast<uint32_t>(size));
}
UNITY_PLAYER(void, nativeSetLocationStatus)(JNIEnv* env, jobject self, jint status) {
    GUEST_CALL("nativeSetLocationStatus", static_cast<uint32_t>(status));
}
UNITY_PLAYER(void, nativeSetLocation)(JNIEnv* env, jobject self, jfloat a, jfloat b, jfloat c,
                                      jfloat d, jdouble t, jfloat e) {
    // r2=a r3=b, then the stack: c, d, then t on an 8-byte boundary, then e.
    std::vector<uint32_t> words{GuestEnv(), Ref(env, self), Bits(a), Bits(b), Bits(c), Bits(d)};
    PushDouble(words, t);
    words.push_back(Bits(e));
    Call("nativeSetLocation", words);
}

JNIEXPORT void JNICALL Java_com_unity3d_player_UnityPlayer_UnitySendMessage(
    JNIEnv* env, jclass self, jstring object, jstring method, jstring message) {
    GUEST_CALL("UnitySendMessage", Ref(env, object), Ref(env, method), Ref(env, message));
}

#define UNITY_WWW(ret, name) \
    JNIEXPORT ret JNICALL Java_com_unity3d_player_WWW_##name

UNITY_WWW(void, doneCallback)(JNIEnv* env, jclass self, jint id) {
    GUEST_CALL("doneCallback", static_cast<uint32_t>(id));
}
UNITY_WWW(void, errorCallback)(JNIEnv* env, jclass self, jint id, jstring message) {
    GUEST_CALL("errorCallback", static_cast<uint32_t>(id), Ref(env, message));
}
UNITY_WWW(jboolean, headerCallback)(JNIEnv* env, jclass self, jint id, jstring header) {
    return GUEST_CALL("headerCallback", static_cast<uint32_t>(id), Ref(env, header)) ? JNI_TRUE
                                                                                    : JNI_FALSE;
}
UNITY_WWW(jboolean, readCallback)(JNIEnv* env, jclass self, jint id, jbyteArray data, jint size) {
    return GUEST_CALL("readCallback", static_cast<uint32_t>(id), Ref(env, data),
                      static_cast<uint32_t>(size)) ? JNI_TRUE : JNI_FALSE;
}
UNITY_WWW(void, progressCallback)(JNIEnv* env, jclass self, jint id, jfloat a, jfloat b, jdouble c,
                                  jint d) {
    // r2=id r3=a, then the stack: b, a padding word, c on its 8-byte boundary, d.
    std::vector<uint32_t> words{GuestEnv(), Ref(env, self), static_cast<uint32_t>(id), Bits(a),
                                Bits(b), 0};
    PushDouble(words, c);
    words.push_back(static_cast<uint32_t>(d));
    Call("progressCallback", words);
}

#define UNITY_REFLECT(ret, name) \
    JNIEXPORT ret JNICALL Java_com_unity3d_player_ReflectionHelper_##name

UNITY_REFLECT(void, nativeProxyFinalize)(JNIEnv* env, jclass self, jint id) {
    GUEST_CALL("nativeProxyFinalize", static_cast<uint32_t>(id));
}
UNITY_REFLECT(jobject, nativeProxyInvoke)(JNIEnv* env, jclass self, jint id, jstring method,
                                          jobjectArray args) {
    uint32_t result = GUEST_CALL("nativeProxyInvoke", static_cast<uint32_t>(id), Ref(env, method),
                                 Ref(env, args));
    return static_cast<jobject>(HostPointerFor(result));
}

}  // extern "C"

// FMOD's Java audio thread. Unity 4 keeps its built-in FMOD mixer even in a Wwise
// title, and FMODAudioDevice.start() calls straight into it, so these have to
// resolve or the first window-focus change kills the process.
extern "C" {

#define FMOD_DEVICE(ret, name) \
    JNIEXPORT ret JNICALL Java_org_fmod_FMODAudioDevice_##name

FMOD_DEVICE(jint, fmodInitJni)(JNIEnv* env, jobject self) {
    return static_cast<jint>(GUEST_CALL("fmodInitJni"));
}
FMOD_DEVICE(jint, fmodGetInfo)(JNIEnv* env, jobject self, jint which) {
    return static_cast<jint>(GUEST_CALL("fmodGetInfo", static_cast<uint32_t>(which)));
}
FMOD_DEVICE(jint, fmodBlockStreaming)(JNIEnv* env, jobject self) {
    return static_cast<jint>(GUEST_CALL("fmodBlockStreaming"));
}
FMOD_DEVICE(jint, fmodUnblockStreaming)(JNIEnv* env, jobject self) {
    return static_cast<jint>(GUEST_CALL("fmodUnblockStreaming"));
}
// The guest mixes into a direct ByteBuffer it cannot address, so it writes into a
// shadow that is copied back to the real buffer before Java reads it.
FMOD_DEVICE(jint, fmodProcess)(JNIEnv* env, jobject self, jobject buffer) {
    jint written = static_cast<jint>(GUEST_CALL("fmodProcess", Ref(env, buffer)));
    FlushGuestDirectBuffer(env, buffer);
    return written;
}
FMOD_DEVICE(jint, fmodProcessMicData)(JNIEnv* env, jobject self, jobject buffer, jint size) {
    jint read = static_cast<jint>(
        GUEST_CALL("fmodProcessMicData", Ref(env, buffer), static_cast<uint32_t>(size)));
    FlushGuestDirectBuffer(env, buffer);
    return read;
}

// Wwise's asset-manager bridge. Unlike Unity's, the plugin exports these under
// their JNI names instead of registering them, so they are resolved by symbol
// rather than through the RegisterNatives registry.
//
// The Java side of it, SoundBankIOInitalizerJavaClass, loads its own library from
// a static initialiser - and in this package there is no arm64 library by that
// name to load, only the 32-bit guest under assets/guestlib/. The
// initialiser therefore threw, the class stayed permanently un-initialisable, and
// every later GetMethodID on it failed. Wwise reports that as "Failed to find Java
// class method" and gives up on the whole sound engine. The smali now swallows
// that one exception; ART then resolves the two natives below out of libmain.so,
// which is already loaded.
JNIEXPORT jint JNICALL
Java_com_audiokinetic_aksoundengine_SoundBankIOInitalizerJavaClass_NativeSetAssetManager(
    JNIEnv* env, jobject self, jobject asset_manager) {
    return static_cast<jint>(CallGuestExport(
        "Java_com_audiokinetic_aksoundengine_SoundBankIOInitalizerJavaClass_NativeSetAssetManager",
        {GuestEnv(), Ref(env, self), Ref(env, asset_manager)}));
}

JNIEXPORT jint JNICALL
Java_com_audiokinetic_aksoundengine_SoundBankIOInitalizerJavaClass_NativeSetExpansionData(
    JNIEnv* env, jobject self, jstring main_obb, jstring patch_obb) {
    return static_cast<jint>(CallGuestExport(
        "Java_com_audiokinetic_aksoundengine_SoundBankIOInitalizerJavaClass_NativeSetExpansionData",
        {GuestEnv(), Ref(env, self), Ref(env, main_obb), Ref(env, patch_obb)}));
}

}  // extern "C"
