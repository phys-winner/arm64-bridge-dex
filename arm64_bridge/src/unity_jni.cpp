// SPDX-License-Identifier: GPL-3.0-or-later
// Registry of the guest's JNI natives, learned from RegisterNatives.

#include "unity_jni.h"

#include <android/log.h>

#include <map>
#include <mutex>

#include "dynarmic_host.h"
#include "elf_loader.h"

#define LOG_TAG "DXTF_UnityJNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {

std::mutex g_mutex;
std::map<std::string, uint32_t> g_natives;

}  // namespace

void RecordGuestNative(const std::string& class_hint, const std::string& name,
                       const std::string& signature, uint32_t guest_fn) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_natives[name] = guest_fn;
    LOGI("registered %s%s%s -> 0x%08x", class_hint.c_str(), class_hint.empty() ? "" : ".",
         (name + signature).c_str(), guest_fn);
}

uint32_t GuestNative(const std::string& name) {
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_natives.find(name);
    return it == g_natives.end() ? 0 : it->second;
}

void RunGuestJniOnLoad(DynarmicHost& host, uint32_t guest_vm) {
    // Order mirrors the real load order: the platform dlopen()s libmain.so, whose
    // ANativeActivity_onCreate then dlopen()s libunity.so. JNI_OnLoad is not run by
    // dlopen at all - System.loadLibrary does it - so both are driven from here.
    for (const char* soname : {"libmain.so", "libunity.so", "libAkSoundEnginePlugin.so"}) {
        ElfLoader* lib = host.GetLibrary(soname);
        if (!lib) continue;
        uint32_t fn = lib->GetSymbolAddress("JNI_OnLoad");
        if (!fn) continue;
        LOGI("calling %s JNI_OnLoad at 0x%08x", soname, fn);
        uint32_t version = host.RunFunction(fn, {guest_vm, 0});
        LOGI("%s JNI_OnLoad returned 0x%08x", soname, version);
    }
}
