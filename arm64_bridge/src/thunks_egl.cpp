// SPDX-License-Identifier: GPL-3.0-or-later
// EGL for the guest.
//
// Without this Unity gets EGL_NO_DISPLAY from the very first call, prints
// "[EGL] Unable to retrieve the default EGL display!" and calls exit(1) - which
// in a bridge does not end the process, it just unwinds and lets the whole
// start-up run again, so the log fills with a second mono_init asserting on
// icalls it already registered. Every symptom downstream of the first frame
// starts here.
//
// EGLDisplay, EGLConfig, EGLSurface and EGLContext are all opaque host pointers
// and do not fit in a guest word, so they travel as handles - the same table
// that already carries ANativeWindow and jobject. Attribute lists are EGL_NONE
// terminated arrays in guest memory and are copied out before use.

#include "thunk_manager.h"
#include "dynarmic_host.h"
#include "thunk_types.h"
#include "native_activity_bridge.h"

#include <EGL/egl.h>
#include <android/native_window.h>
#include <android/log.h>

#include <vector>

#define LOG_TAG "DXTF_Egl"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

using Jit = Dynarmic::A32::Jit;

namespace {

template <typename T>
T HandleTo(uint32_t handle) {
    return static_cast<T>(HostPointerFor(handle));
}

// EGL_DEFAULT_DISPLAY is 0 and so is "no handle", which happens to be the same
// value on both sides - but only by luck, so it is spelled out rather than
// relied on.
EGLNativeDisplayType NativeDisplay(uint32_t guest_value) {
    return guest_value == 0 ? EGL_DEFAULT_DISPLAY
                            : static_cast<EGLNativeDisplayType>(HostPointerFor(guest_value));
}

// Attribute lists end at EGL_NONE. The cap is a guard against a guest pointer
// that is not a list at all; a real EGL config request is a dozen entries.
std::vector<EGLint> ReadAttribList(DynarmicHost& host, uint32_t addr) {
    std::vector<EGLint> attribs;
    if (!addr) {
        attribs.push_back(EGL_NONE);
        return attribs;
    }
    for (int i = 0; i < 128; ++i) {
        auto value = static_cast<EGLint>(host.MemoryRead32(addr + i * 4));
        attribs.push_back(value);
        if (value == EGL_NONE) return attribs;
    }
    LOGE("attribute list at 0x%08X is not EGL_NONE terminated", addr);
    attribs.push_back(EGL_NONE);
    return attribs;
}

}  // namespace

void ThunkManager::RegisterEglThunks() {
    RegisterThunk("eglGetDisplay", [](Jit& jit, DynarmicHost&) {
        EGLDisplay dpy = eglGetDisplay(NativeDisplay(jit.Regs()[0]));
        jit.Regs()[0] = GuestHandleFor(dpy);
    });

    RegisterThunk("eglInitialize", [](Jit& jit, DynarmicHost& host) {
        EGLint major = 0, minor = 0;
        EGLBoolean ok = eglInitialize(HandleTo<EGLDisplay>(jit.Regs()[0]), &major, &minor);
        if (jit.Regs()[1]) host.MemoryWrite32(jit.Regs()[1], static_cast<uint32_t>(major));
        if (jit.Regs()[2]) host.MemoryWrite32(jit.Regs()[2], static_cast<uint32_t>(minor));
        LOGI("eglInitialize -> %d (EGL %d.%d)", ok, major, minor);
        jit.Regs()[0] = ok;
    });

    RegisterThunk("eglTerminate", [](Jit& jit, DynarmicHost&) {
        jit.Regs()[0] = eglTerminate(HandleTo<EGLDisplay>(jit.Regs()[0]));
    });

    // eglChooseConfig(dpy, attrib_list, configs, config_size, num_config): the
    // last argument is on the stack, and `configs` is an array of guest words
    // that has to receive handles rather than the host pointers themselves.
    RegisterThunk("eglChooseConfig", [](Jit& jit, DynarmicHost& host) {
        EGLDisplay dpy = HandleTo<EGLDisplay>(jit.Regs()[0]);
        std::vector<EGLint> attribs = ReadAttribList(host, jit.Regs()[1]);
        uint32_t configs_addr = jit.Regs()[2];
        EGLint config_size = static_cast<EGLint>(jit.Regs()[3]);
        uint32_t num_addr = host.MemoryRead32(jit.Regs()[13]);

        std::vector<EGLConfig> configs(config_size > 0 && configs_addr ? config_size : 0);
        EGLint num = 0;
        EGLBoolean ok = eglChooseConfig(dpy, attribs.data(), configs.empty() ? nullptr : configs.data(),
                                        configs.empty() ? 0 : config_size, &num);
        if (ok) {
            for (EGLint i = 0; i < num && i < static_cast<EGLint>(configs.size()); ++i) {
                host.MemoryWrite32(configs_addr + i * 4, GuestHandleFor(configs[i]));
            }
        }
        if (num_addr) host.MemoryWrite32(num_addr, static_cast<uint32_t>(num));
        jit.Regs()[0] = ok;
    });

    RegisterThunk("eglGetConfigAttrib", [](Jit& jit, DynarmicHost& host) {
        EGLint value = 0;
        EGLBoolean ok = eglGetConfigAttrib(HandleTo<EGLDisplay>(jit.Regs()[0]),
                                           HandleTo<EGLConfig>(jit.Regs()[1]),
                                           static_cast<EGLint>(jit.Regs()[2]), &value);
        if (jit.Regs()[3]) host.MemoryWrite32(jit.Regs()[3], static_cast<uint32_t>(value));
        jit.Regs()[0] = ok;
    });

    RegisterThunk("eglCreateWindowSurface", [](Jit& jit, DynarmicHost& host) {
        auto* window = HandleTo<ANativeWindow*>(jit.Regs()[2]);
        std::vector<EGLint> attribs = ReadAttribList(host, jit.Regs()[3]);
        EGLSurface surface = eglCreateWindowSurface(HandleTo<EGLDisplay>(jit.Regs()[0]),
                                                    HandleTo<EGLConfig>(jit.Regs()[1]),
                                                    window, attribs.data());
        if (surface == EGL_NO_SURFACE) LOGE("eglCreateWindowSurface failed: 0x%04X", eglGetError());
        // diag: which native window the engine is drawing into. A surface built from
        // a window that has since been replaced still swaps happily and shows
        // nothing - the layer on screen never receives a buffer.
        LOGI("[surface] eglCreateWindowSurface(window %p) -> %p", (void*)window, (void*)surface);
        jit.Regs()[0] = GuestHandleFor(surface);
    });

    RegisterThunk("eglDestroySurface", [](Jit& jit, DynarmicHost&) {
        LOGI("[surface] eglDestroySurface(%p)", (void*)HandleTo<EGLSurface>(jit.Regs()[1]));
        jit.Regs()[0] = eglDestroySurface(HandleTo<EGLDisplay>(jit.Regs()[0]),
                                          HandleTo<EGLSurface>(jit.Regs()[1]));
    });

    RegisterThunk("eglCreateContext", [](Jit& jit, DynarmicHost& host) {
        std::vector<EGLint> attribs = ReadAttribList(host, jit.Regs()[3]);
        EGLContext ctx = eglCreateContext(HandleTo<EGLDisplay>(jit.Regs()[0]),
                                          HandleTo<EGLConfig>(jit.Regs()[1]),
                                          HandleTo<EGLContext>(jit.Regs()[2]), attribs.data());
        if (ctx == EGL_NO_CONTEXT) LOGE("eglCreateContext failed: 0x%04X", eglGetError());
        jit.Regs()[0] = GuestHandleFor(ctx);
    });

    RegisterThunk("eglDestroyContext", [](Jit& jit, DynarmicHost&) {
        jit.Regs()[0] = eglDestroyContext(HandleTo<EGLDisplay>(jit.Regs()[0]),
                                          HandleTo<EGLContext>(jit.Regs()[1]));
    });

    RegisterThunk("eglMakeCurrent", [](Jit& jit, DynarmicHost&) {
        EGLBoolean ok = eglMakeCurrent(HandleTo<EGLDisplay>(jit.Regs()[0]),
                                       HandleTo<EGLSurface>(jit.Regs()[1]),
                                       HandleTo<EGLSurface>(jit.Regs()[2]),
                                       HandleTo<EGLContext>(jit.Regs()[3]));
        if (!ok) LOGE("eglMakeCurrent failed: 0x%04X", eglGetError());
        jit.Regs()[0] = ok;
    });

    // The swap count is the one number that says whether the guest is drawing at
    // all, as opposed to running but never reaching the end of a frame.
    RegisterThunk("eglSwapBuffers", [](Jit& jit, DynarmicHost&) {
        static uint64_t frames = 0;
        EGLBoolean ok = eglSwapBuffers(HandleTo<EGLDisplay>(jit.Regs()[0]),
                                       HandleTo<EGLSurface>(jit.Regs()[1]));
        if (++frames <= 40 || frames % 60 == 0)
            LOGI("eglSwapBuffers #%llu -> %d", (unsigned long long)frames, ok);
        jit.Regs()[0] = ok;
    });

    RegisterThunk("eglGetError", [](Jit& jit, DynarmicHost&) {
        jit.Regs()[0] = static_cast<uint32_t>(eglGetError());
    });

    // The address handed back is one the guest will branch to, so it has to be a
    // guest address - a host function pointer would be executed as ARM code.
    // Only names that already have a thunk get one: minting a trampoline for
    // anything else would turn "extension missing" into "extension present and
    // silently does nothing", which is far harder to diagnose.
    RegisterThunk("eglGetProcAddress", [this](Jit& jit, DynarmicHost& host) {
        const char* name = reinterpret_cast<const char*>(
            GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[0]));
        if (!name) { jit.Regs()[0] = 0; return; }
        if (!IsRegistered(name)) {
            LOGI("eglGetProcAddress('%s') - not bridged, reporting unsupported", name);
            jit.Regs()[0] = 0;
            return;
        }
        jit.Regs()[0] = GetOrCreateThunkAddress(name);
    });
}
