// SPDX-License-Identifier: GPL-3.0-or-later
#include "thunk_manager.h"
#include "dynarmic_host.h"
#include "elf_loader.h"
#include "bridge_config.h"
#include "guest_signals.h"
#include <atomic>
#include <chrono>
#include <ctime>
#include <thread>
#include <unistd.h>
#include "thunk_types.h"
#include "unity_jni.h"
#include <android/log.h>
#include <algorithm>
#include <vector>
#include <string>
#include <cstdio>
#include <dlfcn.h>
#include <signal.h>
#include <sys/syscall.h>
#include <sys/uio.h>
#include <ucontext.h>

#define LOG_TAG "GOF2_ThunkManager"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

uint32_t ThunkManager::s_screenWidth = 1213;
uint32_t ThunkManager::s_screenHeight = 540;
uint32_t ThunkManager::s_forced_dpi = 0;
uint32_t ThunkManager::s_ui_inset = 0;

ThunkManager::ThunkManager(DynarmicHost& host) : m_host(host) {}

ThunkManager::~ThunkManager() = default;


namespace {
// Defined with the rest of the diagnostics further down. Started here rather than
// from every dispatch: even std::call_once is an atomic load, and that path runs
// two million times a second.
//
// The investigation hooks are intentionally compiled but dormant in release
// builds.  Keeping their two atomic increments on every thunk cost millions of
// contended operations during startup, and the emergency PC dump interrupts all
// JITs at once - useful while hunting a hang, inappropriate in the shipped game.
constexpr bool kRuntimeDiagnostics = false;
void StartWatchdog();
}  // namespace

void ThunkManager::RegisterAllThunks() {
    if (kRuntimeDiagnostics) StartWatchdog();
    RegisterGLThunks();
    RegisterLibcThunks();
    RegisterExtraLibcThunks();
    RegisterZlibThunks();
    RegisterEglThunks();
    RegisterAndroidThunks();
    RegisterOpenSLThunks();
    RegisterDataSymbols();
    InstallFastMemcpy();
    SetupGuestJNIEnv();
}

#include <jni.h>
#include <sys/system_properties.h>
#include <cmath>
#include <cctype>
#include <cstring>

JavaVM* GetHostJavaVM();

static std::string QueryDeviceLanguage() {
    std::string lang = "en";
    JavaVM* jvm = GetHostJavaVM();
    if (jvm) {
        JNIEnv* env = nullptr;
        bool attached = false;
        if (jvm->GetEnv((void**)&env, JNI_VERSION_1_6) != JNI_OK) {
            if (jvm->AttachCurrentThread(&env, nullptr) == JNI_OK) {
                attached = true;
            }
        }
        if (env) {
            jclass locCls = env->FindClass("java/util/Locale");
            if (locCls) {
                jmethodID getDefault = env->GetStaticMethodID(locCls, "getDefault", "()Ljava/util/Locale;");
                if (getDefault) {
                    jobject locObj = env->CallStaticObjectMethod(locCls, getDefault);
                    if (locObj) {
                        jmethodID getLang = env->GetMethodID(locCls, "getLanguage", "()Ljava/lang/String;");
                        if (getLang) {
                            jstring jStr = (jstring)env->CallObjectMethod(locObj, getLang);
                            if (jStr) {
                                const char* utf = env->GetStringUTFChars(jStr, nullptr);
                                if (utf && strlen(utf) > 0) {
                                    lang = utf;
                                }
                                if (utf) env->ReleaseStringUTFChars(jStr, utf);
                            }
                        }
                    }
                }
            }
            if (env->ExceptionCheck()) env->ExceptionClear();
            if (attached) jvm->DetachCurrentThread();
        }
    }
    if (lang.empty() || lang == "en") {
        char prop[PROP_VALUE_MAX] = {0};
        if (__system_property_get("persist.sys.locale", prop) > 0 ||
            __system_property_get("ro.product.locale", prop) > 0 ||
            __system_property_get("persist.sys.language", prop) > 0) {
            char* dash = strchr(prop, '-');
            if (dash) *dash = '\0';
            char* under = strchr(prop, '_');
            if (under) *under = '\0';
            if (strlen(prop) > 0) lang = prop;
        }
    }
    for (char& c : lang) c = static_cast<char>(tolower(static_cast<unsigned char>(c)));
    return lang;
}

static std::string QueryDeviceCountryCode() {
    std::string country = "US";
    JavaVM* jvm = GetHostJavaVM();
    if (jvm) {
        JNIEnv* env = nullptr;
        bool attached = false;
        if (jvm->GetEnv((void**)&env, JNI_VERSION_1_6) != JNI_OK) {
            if (jvm->AttachCurrentThread(&env, nullptr) == JNI_OK) {
                attached = true;
            }
        }
        if (env) {
            jclass locCls = env->FindClass("java/util/Locale");
            if (locCls) {
                jmethodID getDefault = env->GetStaticMethodID(locCls, "getDefault", "()Ljava/util/Locale;");
                if (getDefault) {
                    jobject locObj = env->CallStaticObjectMethod(locCls, getDefault);
                    if (locObj) {
                        jmethodID getCountry = env->GetMethodID(locCls, "getCountry", "()Ljava/lang/String;");
                        if (getCountry) {
                            jstring jStr = (jstring)env->CallObjectMethod(locObj, getCountry);
                            if (jStr) {
                                const char* utf = env->GetStringUTFChars(jStr, nullptr);
                                if (utf && strlen(utf) > 0) {
                                    country = utf;
                                }
                                if (utf) env->ReleaseStringUTFChars(jStr, utf);
                            }
                        }
                    }
                }
            }
            if (env->ExceptionCheck()) env->ExceptionClear();
            if (attached) jvm->DetachCurrentThread();
        }
    }
    if (country.empty() || country == "US") {
        char prop[PROP_VALUE_MAX] = {0};
        if (__system_property_get("persist.sys.country", prop) > 0 ||
            __system_property_get("ro.product.locale.region", prop) > 0) {
            if (strlen(prop) > 0) country = prop;
        } else if (__system_property_get("persist.sys.locale", prop) > 0 ||
                   __system_property_get("ro.product.locale", prop) > 0) {
            char* dash = strchr(prop, '-');
            if (dash && strlen(dash + 1) >= 2) {
                country = std::string(dash + 1, 2);
            } else {
                char* under = strchr(prop, '_');
                if (under && strlen(under + 1) >= 2) {
                    country = std::string(under + 1, 2);
                }
            }
        }
    }
    for (char& c : country) c = static_cast<char>(toupper(static_cast<unsigned char>(c)));
    return country;
}

static uint32_t QueryDeviceDensity() {
    if (ThunkManager::s_forced_dpi > 0) {
        return ThunkManager::s_forced_dpi;
    }
    BridgeConfig& ext = GetBridgeConfig();
    if (ext.forced_dpi > 0) {
        return ext.forced_dpi;
    }
    uint32_t density = 0;
    JavaVM* jvm = GetHostJavaVM();
    if (jvm) {
        JNIEnv* env = nullptr;
        bool attached = false;
        if (jvm->GetEnv((void**)&env, JNI_VERSION_1_6) != JNI_OK) {
            if (jvm->AttachCurrentThread(&env, nullptr) == JNI_OK) {
                attached = true;
            }
        }
        if (env) {
            jclass resCls = env->FindClass("android/content/res/Resources");
            if (resCls) {
                jmethodID getSys = env->GetStaticMethodID(resCls, "getSystem", "()Landroid/content/res/Resources;");
                if (getSys) {
                    jobject resObj = env->CallStaticObjectMethod(resCls, getSys);
                    if (resObj) {
                        jmethodID getDM = env->GetMethodID(resCls, "getDisplayMetrics", "()Landroid/util/DisplayMetrics;");
                        if (getDM) {
                            jobject dmObj = env->CallObjectMethod(resObj, getDM);
                            if (dmObj) {
                                jclass dmCls = env->FindClass("android/util/DisplayMetrics");
                                jfieldID fid = env->GetFieldID(dmCls, "densityDpi", "I");
                                if (fid) {
                                    density = static_cast<uint32_t>(env->GetIntField(dmObj, fid));
                                }
                            }
                        }
                    }
                }
            }
            if (env->ExceptionCheck()) env->ExceptionClear();
            if (attached) jvm->DetachCurrentThread();
        }
    }
    if (density == 0) {
        char prop[PROP_VALUE_MAX] = {0};
        if (__system_property_get("ro.sf.lcd_density", prop) > 0 ||
            __system_property_get("qemu.sf.lcd_density", prop) > 0) {
            density = static_cast<uint32_t>(atoi(prop));
        }
    }
    if (density == 0) {
        density = 320;
    }
    return density;
}

static uint32_t QueryDeviceIsPad() {
    JavaVM* jvm = GetHostJavaVM();
    if (jvm) {
        JNIEnv* env = nullptr;
        bool attached = false;
        if (jvm->GetEnv((void**)&env, JNI_VERSION_1_6) != JNI_OK) {
            if (jvm->AttachCurrentThread(&env, nullptr) == JNI_OK) {
                attached = true;
            }
        }
        if (env) {
            jclass devInfoCls = env->FindClass("net/fishlabs/ow_utils/DeviceInfo");
            if (devInfoCls) {
                jmethodID isPadMethod = env->GetStaticMethodID(devInfoCls, "isPad", "()Z");
                if (isPadMethod) {
                    jboolean res = env->CallStaticBooleanMethod(devInfoCls, isPadMethod);
                    if (env->ExceptionCheck()) env->ExceptionClear();
                    if (attached) jvm->DetachCurrentThread();
                    return res ? 1 : 0;
                }
            }
            if (env->ExceptionCheck()) env->ExceptionClear();

            jclass resCls = env->FindClass("android/content/res/Resources");
            if (resCls) {
                jmethodID getSys = env->GetStaticMethodID(resCls, "getSystem", "()Landroid/content/res/Resources;");
                if (getSys) {
                    jobject resObj = env->CallStaticObjectMethod(resCls, getSys);
                    if (resObj) {
                        jmethodID getCfg = env->GetMethodID(resCls, "getConfiguration", "()Landroid/content/res/Configuration;");
                        if (getCfg) {
                            jobject cfgObj = env->CallObjectMethod(resObj, getCfg);
                            if (cfgObj) {
                                jclass cfgCls = env->FindClass("android/content/res/Configuration");
                                jfieldID slFid = env->GetFieldID(cfgCls, "screenLayout", "I");
                                if (slFid) {
                                    jint screenLayout = env->GetIntField(cfgObj, slFid);
                                    if ((screenLayout & 0x0F) >= 3) { // SCREENLAYOUT_SIZE_LARGE or XLARGE
                                        if (env->ExceptionCheck()) env->ExceptionClear();
                                        if (attached) jvm->DetachCurrentThread();
                                        return 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            if (env->ExceptionCheck()) env->ExceptionClear();
            if (attached) jvm->DetachCurrentThread();
        }
    }

    uint32_t w = ThunkManager::s_screenWidth;
    uint32_t h = ThunkManager::s_screenHeight;
    uint32_t dpi = QueryDeviceDensity();
    if (w > 0 && h > 0 && dpi > 0) {
        double w_in = static_cast<double>(w) / static_cast<double>(dpi);
        double h_in = static_cast<double>(h) / static_cast<double>(dpi);
        double diag = sqrt(w_in * w_in + h_in * h_in);
        if (diag >= 6.5) return 1;
    }
    return 0;
}




namespace {
// Both live with the rest of the diagnostic state further down; declared here so
// registration can publish into them.
void PublishFastThunk(uint32_t id, const ThunkHandler* handler, const char* name);
void SetDiagContext(const std::unordered_map<uint32_t, std::string>* names, DynarmicHost* host);
}  // namespace

void ThunkManager::RegisterThunk(const std::string& name, ThunkHandler handler) {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    // Registering a name twice mints a second trampoline and repoints the symbol at
    // it, so the last registration silently wins. That is how abort() went back to
    // returning to its caller months after it was fixed to jump to the exit trap:
    // the fix was still in the file, just shadowed thirty screens further down.
    if (m_symbol_to_address.count(name)) {
        LOGE("thunk '%s' registered twice - the later one wins, the earlier is dead code",
             name.c_str());
    }
    uint32_t thunk_id = static_cast<uint32_t>(m_id_to_handler.size() + 1);
    uint32_t thunk_addr = m_next_thunk_addr;
    m_next_thunk_addr += 8; // 2 instructions per trampoline

    // Write ARM32 trampoline instructions into guest memory:
    // 0: SVC #thunk_id (0xEF000000 | thunk_id)
    // 4: BX LR (0xE12FFF1E)
    uint32_t svc_instr = 0xEF000000 | (thunk_id & 0x00FFFFFF);
    uint32_t bx_lr_instr = 0xE12FFF1E;

    m_host.MemoryWrite32(thunk_addr, svc_instr);
    m_host.MemoryWrite32(thunk_addr + 4, bx_lr_instr);
    // Trampolines are also created at runtime (dlsym, OpenSL ES vtables), so the JIT may
    // already hold a translation of this address range.
    m_host.InvalidateGuestCode(thunk_addr, 8);

    m_symbol_to_address[name] = thunk_addr;
    m_id_to_handler[thunk_id] = std::move(handler);
    m_id_to_name[thunk_id] = name;
    // unordered_map nodes never move and nothing is erased, so both the handler
    // and the name string keep these addresses for the life of the process.
    PublishFastThunk(thunk_id, &m_id_to_handler[thunk_id], m_id_to_name[thunk_id].c_str());
    SetDiagContext(&m_id_to_name, &m_host);
}

void ThunkManager::RegisterSymbol(const std::string& name, uint32_t address) {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    m_symbol_to_address[name] = address;
}

bool ThunkManager::IsRegistered(const std::string& name) {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    return m_symbol_to_address.find(name) != m_symbol_to_address.end();
}

uint32_t ThunkManager::GetOrCreateThunkAddress(const std::string& name) {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    auto it = m_symbol_to_address.find(name);
    if (it != m_symbol_to_address.end()) {
        return it->second;
    }
    // Default dummy stub if not explicitly registered
    RegisterThunk(name, [name](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        LOGI("[THUNK] Default stub executed for: %s", name.c_str());
        jit.Regs()[0] = 0;
    });
    return m_symbol_to_address[name];
}

// A guest that stops making progress stops logging too, so there is nothing in
// the log to read - every thread is simply asleep. This records which thunk each
// guest thread is currently inside, and a watchdog names the ones that have not
// come back. Blocking in pthread_cond_wait is normal; blocking in it forever is
// the bug, and the difference is only visible from outside.
namespace {

struct ThunkFrame {
    std::atomic<const char*> name{nullptr};
    std::atomic<int64_t> entered_ms{0};
    std::atomic<int32_t> tid{0};
    // Guest return address, so a stuck wait can be traced to the guest code that
    // decided to wait rather than just to the libc function it is sitting in.
    std::atomic<uint32_t> guest_lr{0};
    // diag: a stack scan taken every so often, from inside the thunk - which is
    // the only place another thread's guest registers can be read safely.
    std::atomic<uint32_t> calls{0};
    std::atomic<uint32_t> stack[12];
    // diag: the thunk that was running when this thread last re-entered the guest,
    // and how deep it is now.
    std::atomic<const char*> outer_name{nullptr};
    std::atomic<int> depth{0};
    // Where in the bridge itself this thread is, which a thunk name cannot say:
    // inside Jit::Run, or queued on one of the bridge's own locks.
    std::atomic<const char*> phase{nullptr};
    // diag: guest stacks are fixed slots packed against each other, and the whole
    // 32-bit space is mapped, so a thread that runs past the bottom of its own
    // silently overwrites the top of the next one instead of faulting. The deepest
    // stack pointer seen says how much room was actually left.
    std::atomic<uint32_t> min_sp{0xFFFFFFFFu};
    std::atomic<uint32_t> stack_low{0};
    std::atomic<bool> overflowed{false};
    // The watchdog used to read the clock on the way into every thunk to know how
    // long a thread had been inside one. In game that path runs about two million
    // times a second - memcpy alone - and the clock read costs more there than all
    // the rest of the bookkeeping together. It samples the call counter instead: a
    // thread still inside a thunk whose counter has not moved across a whole
    // watchdog period has not come back from it. Same answer, same 10-second
    // resolution, one relaxed increment.
    std::atomic<uint32_t> calls_seen{0};
    std::atomic<int> stuck_periods{0};
};

// A raw stack dump mixes return addresses with every object pointer the frames
// happen to hold, and the object pointers are the majority. A word only counts as
// a return address if the instruction in front of it is a call - that turns the
// dump into a backtrace, which is the difference between "libunity, somewhere" and
// the name of the loop that is not finishing.
bool LooksLikeReturnAddress(DynarmicHost& host, uint32_t addr) {
    if (addr < 0x40000000u || addr >= 0x47000000u) return false;
    uint32_t arm = host.MemoryRead32(addr - 4);
    if ((arm & 0x0F000000u) == 0x0B000000u && (arm >> 28) != 0xF) return true;  // BL<c>
    if ((arm & 0xFE000000u) == 0xFA000000u) return true;                        // BLX imm
    if ((arm & 0x0FFFFFF0u) == 0x012FFF30u) return true;                        // BLX reg
    uint16_t first = static_cast<uint16_t>(arm & 0xFFFF);
    uint16_t second = static_cast<uint16_t>(arm >> 16);
    if (first >= 0xF000 && first <= 0xF7FF && second >= 0xE800) return true;    // Thumb BL/BLX
    if ((second & 0xFF80) == 0x4780) return true;                               // Thumb BLX reg
    return false;
}

constexpr size_t kMaxTrackedThreads = 128;
ThunkFrame g_frames[kMaxTrackedThreads];
std::atomic<size_t> g_frame_count{0};

// diag: call rate per thunk. A guest that spins through a thunk never looks stuck
// - each call returns immediately - so the watchdog's "who is blocked" view cannot
// see it. The rate can.
constexpr size_t kMaxThunkIds = 4096;
std::atomic<uint64_t> g_thunk_calls[kMaxThunkIds];
uint64_t g_thunk_calls_prev[kMaxThunkIds];

// The lock-free dispatch table. Ids come out of RegisterThunk densely from 1, so
// an array indexed by id needs no hashing; dlsym mints trampolines at runtime, so
// the slots are published with a release store and read with an acquire load.
// Anything past the end falls back to the locked map, which is correct, just slow.
constexpr size_t kFastThunkSlots = 8192;
std::atomic<const ThunkHandler*> g_fast_handler[kFastThunkSlots];
std::atomic<const char*> g_fast_name[kFastThunkSlots];

// diag: the call-site histogram and the periodic guest backtrace. Both are real
// work in a path that runs half a million times a second - the histogram hashes
// and probes, the backtrace reads a thousand guest words - so they stay off unless
// something is being chased. Turn on together with the watchdog's other output.
constexpr bool kThunkDiagnostics = false;

// diag: call-site histogram. Knowing which thunk is busiest does not say who is
// calling it - Unity funnels every lock through one wrapper, so the return address
// alone is the wrapper every time, and the guest object in r0 is what tells two
// spinning loops apart.
// The table is emptied after every report. Startup makes millions of one-shot sites
// - every memcpy with a different pointer is its own - and a table that only ever
// fills is full long before the interesting part of the run, which is exactly when
// it stops recording anything.
constexpr size_t kMaxSites = 16384;
struct CallSite {
    std::atomic<uint32_t> id{0};  // thunk id + 1; 0 means the slot is free
    std::atomic<uint32_t> lr{0};
    std::atomic<uint32_t> arg0{0};
    std::atomic<int32_t> tid{0};
    std::atomic<uint32_t> count{0};
};
CallSite g_sites[kMaxSites];

void NoteCallSite(uint32_t id, uint32_t lr, uint32_t arg0) {
    uint32_t h = id * 2654435761u ^ lr * 2246822519u ^ arg0 * 3266489917u;
    for (uint32_t probe = 0; probe < 8; ++probe) {
        CallSite& site = g_sites[(h + probe) & (kMaxSites - 1)];
        uint32_t cur = site.id.load(std::memory_order_relaxed);
        if (cur == 0) {
            uint32_t expected = 0;
            // Losing the race just costs this one sample, which a histogram does
            // not care about.
            if (!site.id.compare_exchange_strong(expected, id + 1)) continue;
            site.lr.store(lr, std::memory_order_relaxed);
            site.arg0.store(arg0, std::memory_order_relaxed);
        } else if (cur != id + 1 || site.lr.load(std::memory_order_relaxed) != lr ||
                   site.arg0.load(std::memory_order_relaxed) != arg0) {
            continue;
        }
        site.tid.store(gettid(), std::memory_order_relaxed);
        site.count.fetch_add(1, std::memory_order_relaxed);
        return;
    }
}

int64_t NowMs() {
    struct timespec ts {};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<int64_t>(ts.tv_sec) * 1000 + ts.tv_nsec / 1000000;
}

ThunkFrame& MyFrame() {
    thread_local ThunkFrame* mine = [] {
        size_t index = g_frame_count.fetch_add(1);
        if (index >= kMaxTrackedThreads) index = kMaxTrackedThreads - 1;
        g_frames[index].tid.store(gettid());
        return &g_frames[index];
    }();
    return *mine;
}

// Names come from the manager, which owns the id -> name table.
const std::unordered_map<uint32_t, std::string>* g_thunk_names = nullptr;
DynarmicHost* g_diag_host = nullptr;

void PublishFastThunk(uint32_t id, const ThunkHandler* handler, const char* name) {
    if (id >= kFastThunkSlots) return;
    g_fast_name[id].store(name, std::memory_order_relaxed);
    g_fast_handler[id].store(handler, std::memory_order_release);
}

void SetDiagContext(const std::unordered_map<uint32_t, std::string>* names, DynarmicHost* host) {
    g_thunk_names = names;
    g_diag_host = host;
}

void ReportBusiestThunks() {
    std::vector<std::pair<uint64_t, uint32_t>> ranked;
    for (uint32_t id = 0; id < kMaxThunkIds; ++id) {
        uint64_t now = g_thunk_calls[id].load();
        uint64_t delta = now - g_thunk_calls_prev[id];
        g_thunk_calls_prev[id] = now;
        if (delta > 100) ranked.emplace_back(delta, id);
    }
    if (ranked.empty()) return;
    std::sort(ranked.rbegin(), ranked.rend());
    for (size_t i = 0; i < ranked.size() && i < 8; ++i) {
        const char* name = "?";
        if (g_thunk_names) {
            auto it = g_thunk_names->find(ranked[i].second);
            if (it != g_thunk_names->end()) name = it->second.c_str();
        }
        LOGE("[watchdog] %s called %llu times in the last 10s", name,
             (unsigned long long)ranked[i].first);
    }
    // Whoever is calling that often is spinning, and the stack says where from.
    size_t count = g_frame_count.load();
    for (size_t i = 0; i < count && i < kMaxTrackedThreads; ++i) {
        if (g_frames[i].stack[0].load() == 0) continue;
        LOGE("[watchdog] tid %d stack: %08X %08X %08X %08X %08X %08X %08X %08X %08X %08X %08X %08X",
             g_frames[i].tid.load(), g_frames[i].stack[0].load(), g_frames[i].stack[1].load(),
             g_frames[i].stack[2].load(), g_frames[i].stack[3].load(), g_frames[i].stack[4].load(),
             g_frames[i].stack[5].load(), g_frames[i].stack[6].load(), g_frames[i].stack[7].load(),
             g_frames[i].stack[8].load(), g_frames[i].stack[9].load(), g_frames[i].stack[10].load(),
             g_frames[i].stack[11].load());
    }
}

void ReportBusiestCallSites() {
    std::vector<std::pair<uint32_t, size_t>> ranked;
    for (size_t i = 0; i < kMaxSites; ++i) {
        uint32_t count = g_sites[i].count.load(std::memory_order_relaxed);
        if (count > 20) ranked.emplace_back(count, i);
    }
    std::sort(ranked.rbegin(), ranked.rend());
    for (size_t i = 0; i < ranked.size() && i < 14; ++i) {
        CallSite& site = g_sites[ranked[i].second];
        const char* name = "?";
        if (g_thunk_names) {
            auto it = g_thunk_names->find(site.id.load() - 1);
            if (it != g_thunk_names->end()) name = it->second.c_str();
        }
        LOGE("[site] tid %d %s from 0x%08X arg0=0x%08X x%u", site.tid.load(), name,
             site.lr.load(), site.arg0.load(), ranked[i].first);
    }
    for (size_t i = 0; i < kMaxSites; ++i) {
        g_sites[i].count.store(0, std::memory_order_relaxed);
        g_sites[i].id.store(0, std::memory_order_relaxed);
    }
}

// diag: every thunk the holder of Boehm's allocator lock calls while it holds it.
// The lock ends the run held by a thread that is waiting for it, so the answer is in
// what that thread did between taking it and asking for it again.
struct OwnerCall {
    std::atomic<const char*> name{nullptr};
    std::atomic<int> depth{0};
    std::atomic<uint32_t> lr{0};
    std::atomic<uint32_t> sp{0};
};
constexpr size_t kOwnerCalls = 128;
OwnerCall g_owner_calls[kOwnerCalls];
std::atomic<uint32_t> g_owner_call_count{0};

std::atomic<uint32_t> g_owner_epoch{0};

// Keeps the FIRST calls after each acquisition, not the last: the tail is the spin
// this ends in, and the interesting part is what the owner did before it got there.
void RecordOwnerCall(uint32_t epoch, const char* name, int depth, uint32_t lr, uint32_t sp) {
    if (g_owner_epoch.load(std::memory_order_relaxed) != epoch) {
        g_owner_epoch.store(epoch);
        g_owner_call_count.store(0);
    }
    uint32_t slot = g_owner_call_count.fetch_add(1);
    if (slot >= kOwnerCalls) return;
    g_owner_calls[slot].name.store(name);
    g_owner_calls[slot].depth.store(depth);
    g_owner_calls[slot].lr.store(lr);
    g_owner_calls[slot].sp.store(sp);
}

void DumpOwnerCalls(int owner_tid, uint32_t acquire_sp, uint32_t acquire_lr) {
    uint32_t total = g_owner_call_count.load();
    if (!total) return;
    uint32_t shown = total < 24 ? total : 24;
    LOGE("[owner] tid %d holds the allocator lock, taken at sp 0x%08X lr 0x%08X; "
         "first %u of %u thunks since:", owner_tid, acquire_sp, acquire_lr, shown, total);
    std::string line;
    for (uint32_t i = 0; i < shown; ++i) {
        const OwnerCall& c = g_owner_calls[i];
        const char* n = c.name.load();
        uint32_t sp = c.sp.load();
        // '^' marks a thunk called from above the acquiring frame: the guest left
        // the function that took the lock without running its unlock.
        char buf[80];
        snprintf(buf, sizeof(buf), "%s@%08X/%d/sp%+d%s ", n ? n : "?", c.lr.load(), c.depth.load(),
                 static_cast<int>(sp - acquire_sp), sp > acquire_sp ? "^" : "");
        line += buf;
    }
    LOGE("[owner] %s", line.c_str());
}

// diag: a thread that is neither in a thunk nor answering the guest-side PC dump is
// somewhere no guest-side probe can reach - spinning in translated code that never
// gets back to the dispatcher, or inside the bridge itself. A host signal reaches it
// wherever it is. dladdr answers the first question by itself: a PC inside a mapped
// object is bridge code, a PC in no object at all is Dynarmic's own output.
//
// Dynarmic's arm64 backend keeps the guest register file in X28 while translated code
// runs, so when the PC turns out to be JIT output the guest PC is right there - read
// through process_vm_readv, which reports a bad address instead of taking the process
// down with it.
constexpr size_t kJitStateGuestPc = 24 + 15 * 4;  // A32JitState::regs[15]

void HostPcHandler(int, siginfo_t*, void* raw) {
    auto* uc = static_cast<ucontext_t*>(raw);
    auto pc = static_cast<uintptr_t>(uc->uc_mcontext.pc);
    Dl_info info{};
    if (dladdr(reinterpret_cast<void*>(pc), &info) && info.dli_fname) {
        LOGE("[hostpc] tid %d pc %p in %s+0x%zx (%s)", gettid(), reinterpret_cast<void*>(pc),
             info.dli_fname, static_cast<size_t>(pc - reinterpret_cast<uintptr_t>(info.dli_fbase)),
             info.dli_sname ? info.dli_sname : "no exported symbol");
        return;
    }
    uint32_t guest_pc = 0;
    auto state = static_cast<uintptr_t>(uc->uc_mcontext.regs[28]);
    iovec local{&guest_pc, sizeof(guest_pc)};
    iovec remote{reinterpret_cast<void*>(state + kJitStateGuestPc), sizeof(guest_pc)};
    bool got = syscall(SYS_process_vm_readv, getpid(), &local, 1, &remote, 1, 0) ==
               static_cast<long>(sizeof(guest_pc));
    LOGE("[hostpc] tid %d pc %p is JIT output; guest pc 0x%08X%s", gettid(),
         reinterpret_cast<void*>(pc), guest_pc, got ? "" : " (unreadable)");
}

void RequestHostPc(int tid) {
    static std::once_flag once;
    std::call_once(once, [] {
        struct sigaction sa {};
        sa.sa_sigaction = HostPcHandler;
        sa.sa_flags = SA_SIGINFO | SA_RESTART;
        sigemptyset(&sa.sa_mask);
        sigaction(SIGPROF, &sa, nullptr);
    });
    syscall(SYS_tgkill, getpid(), tid, SIGPROF);
}

void StartWatchdog() {
    static std::once_flag once;
    std::call_once(once, [] {
        std::thread([] {
            for (;;) {
                std::this_thread::sleep_for(std::chrono::seconds(10));
                size_t count = g_frame_count.load();
                if (count > kMaxTrackedThreads) count = kMaxTrackedThreads;
                bool any = false;
                bool moved[kMaxTrackedThreads] = {};
                for (size_t i = 0; i < count; ++i) {
                    uint32_t calls = g_frames[i].calls.load();
                    moved[i] = calls != g_frames[i].calls_seen.exchange(calls);
                }
                for (size_t i = 0; i < count; ++i) {
                    const char* name = g_frames[i].name.load();
                    if (!name || moved[i]) {
                        g_frames[i].stuck_periods.store(0);
                        continue;
                    }
                    int periods = g_frames[i].stuck_periods.fetch_add(1) + 1;
                    LOGE("[watchdog] tid %d inside %s for at least %d s, called from guest 0x%08X",
                         g_frames[i].tid.load(), name, periods * 10,
                         g_frames[i].guest_lr.load());
                    // The full report halts every Jit to ask each thread for its PC,
                    // which interrupts the render thread. Worker threads parked in
                    // sem_wait for the life of the process are normal here, so
                    // escalating every period meant doing that forever, ten seconds
                    // apart. Once per episode, three periods in, is enough to catch a
                    // real hang and costs nothing the rest of the time.
                    if (periods == 3) any = true;
                }
                if (any) {
                    // A thread that is not in a thunk shows up nowhere else, and
                    // "nowhere else" covers both "running guest code" and "this thread
                    // is gone" - which is the difference between a slow lock owner and
                    // a lock owner that no longer exists.
                    for (size_t i = 0; i < count; ++i) {
                        const char* name = g_frames[i].name.load();
                        const char* outer = g_frames[i].outer_name.load();
                        const char* phase = g_frames[i].phase.load();
                        uint32_t low = g_frames[i].stack_low.load();
                        uint32_t deepest = g_frames[i].min_sp.load();
                        LOGE("[census] tid %d in %s (%s) depth %d under %s phase %s "
                             "stack 0x%08X..0x%08X deepest 0x%08X (%d KiB spare)",
                             g_frames[i].tid.load(), name ? name : "guest code",
                             moved[i] ? "moving" : "no progress this period",
                             g_frames[i].depth.load(), outer ? outer : "-",
                             phase ? phase : "-", low,
                             low + DynarmicHost::GUEST_STACK_SIZE, deepest,
                             static_cast<int>((deepest - low) / 1024));
                    }
                    LOGE("[watchdog] ---- end of stuck list ----");
                    GuestRequestPcDump();
                    // Threads that have been out of every thunk for a while: either
                    // busy in guest code, which is normal, or wedged, which is not.
                    // Either way the guest-side dump only answers for the ones that
                    // reach the end of a Run, so the rest are asked from the host.
                    for (size_t i = 0; i < count; ++i) {
                        if (g_frames[i].name.load()) continue;
                        if (g_frames[i].calls.load() == 0) continue;
                        if (moved[i]) continue;
                        RequestHostPc(g_frames[i].tid.load());
                    }
                }
                if (any && g_diag_host) {
                    GuestReportStarvedSemaphores(*g_diag_host);
                    g_diag_host->DumpExclusiveWitnesses();
                    g_diag_host->DumpWordWatch();
                    g_diag_host->DumpStrexLog();
                    int owner = g_diag_host->WatchedWordOwner();
                    if (owner) {
                        DumpOwnerCalls(owner, g_diag_host->WatchedWordOwnerSp(),
                                       g_diag_host->WatchedWordOwnerLr());
                        g_diag_host->DumpWatchedWordAcquireStack();
                    }
                }
                if (GuestSignalReportUndelivered()) {
                    // Print every thread, not just the ones over the threshold:
                    // the one that swallowed the signal is by definition not
                    // sitting anywhere long enough to be called stuck.
                    for (size_t i = 0; i < count; ++i) {
                        const char* name = g_frames[i].name.load();
                        LOGE("[signals]   tid %d is in %s (%s), guest lr 0x%08X",
                             g_frames[i].tid.load(), name ? name : "guest code",
                             moved[i] ? "moving" : "no progress this period",
                             g_frames[i].guest_lr.load());
                    }
                }
                ReportBusiestThunks();
                ReportBusiestCallSites();
            }
        }).detach();
    });
}

}  // namespace


// diag: called from the execution core so the census can tell "running guest code"
// apart from "queued on a bridge lock".
void BridgeSetPhase(const char* phase) {
    if (kRuntimeDiagnostics) MyFrame().phase.store(phase);
}

int GuestBacktrace(DynarmicHost& host, uint32_t sp, uint32_t lr, uint32_t* out, int max_out) {
    int found = 0;
    if (max_out <= 0) return 0;
    out[found++] = lr;
    for (int i = 0; i < 256 && found < max_out; ++i) {
        uint32_t word = host.MemoryRead32(sp + i * 4);
        if (!LooksLikeReturnAddress(host, word)) continue;
        out[found++] = word;
    }
    for (int i = found; i < max_out; ++i) out[i] = 0;
    return found;
}

void ThunkManager::DispatchThunk(uint32_t thunk_id, Dynarmic::A32::Jit& jit) {
    // This runs about half a million times a second - memcpy alone accounts for
    // twenty million calls in a two-minute run - so everything on the way to the
    // handler has to be free.
    //
    // It used to take the table mutex and hash the id twice, and then *copy* the
    // std::function out of the map: the default stub captures its own name by
    // value, so that copy was a heap allocation per call. Thunk ids are handed out
    // densely from 1, so the fast path is a flat table published at registration
    // time and read here without a lock, and the handler is called through a
    // pointer instead of a copy.
    const ThunkHandler* handler = nullptr;
    const char* name = nullptr;
    if (thunk_id < kFastThunkSlots) {
        handler = g_fast_handler[thunk_id].load(std::memory_order_acquire);
        name = g_fast_name[thunk_id].load(std::memory_order_relaxed);
    }
    if (!handler) {
        // diag: a thread waiting here is not "running guest code", which is what the
        // census would otherwise call it - and a bridge-side lock held by a thread
        // that is itself blocked looks identical to a guest deadlock from outside.
        MyFrame().name.store("bridge-table-lock");
        std::lock_guard<std::recursive_mutex> lock(m_mutex);
        MyFrame().name.store(nullptr);
        // Nodes of an unordered_map keep their addresses across inserts and nothing
        // is ever erased, so a pointer taken here stays valid after the lock goes.
        auto found = m_id_to_handler.find(thunk_id);
        if (found != m_id_to_handler.end()) handler = &found->second;
        auto named = m_id_to_name.find(thunk_id);
        if (named != m_id_to_name.end()) name = named->second.c_str();
    }
    if (handler) {
        if (!kRuntimeDiagnostics) {
            (*handler)(jit, m_host);
            if (GuestSignalPending()) GuestSignalDeliverPending(m_host);
            return;
        }
        if (thunk_id < kMaxThunkIds) g_thunk_calls[thunk_id].fetch_add(1, std::memory_order_relaxed);
        if (kThunkDiagnostics) NoteCallSite(thunk_id, jit.Regs()[14], jit.Regs()[0]);
        ThunkFrame& frame = MyFrame();
        frame.guest_lr.store(jit.Regs()[14], std::memory_order_relaxed);
        if (kThunkDiagnostics &&
            (frame.calls.fetch_add(1, std::memory_order_relaxed) & 0x3FF) == 0) {
            uint32_t sp = jit.Regs()[13];
            int found = 0;
            frame.stack[found++].store(jit.Regs()[14]);
            for (int i = 0; i < 512 && found < 12; ++i) {
                uint32_t word = m_host.MemoryRead32(sp + i * 4);
                if (!LooksLikeReturnAddress(m_host, word)) continue;
                frame.stack[found++].store(word);
            }
            for (int i = found; i < 12; ++i) frame.stack[i].store(0);
        } else if (!kThunkDiagnostics) {
            frame.calls.fetch_add(1, std::memory_order_relaxed);
        }
        int depth = m_host.NestingDepth();
        frame.depth.store(depth, std::memory_order_relaxed);
        // diag: how much of this thread's fixed guest stack slot is left. Nothing
        // faults when it runs out - fastmem maps the whole 32-bit space and the next
        // thread's slot starts exactly there - so the only sign is this watermark.
        const uint32_t sp = jit.Regs()[13];
        // A guest stack slot is fixed the moment the thread gets a Jit, so this is
        // asked once per thread rather than on every call.
        thread_local uint32_t stack_low = 0;
        if (stack_low == 0) {
            stack_low = m_host.GuestStackLimit();
            frame.stack_low.store(stack_low);
        }
        if (sp < frame.min_sp.load(std::memory_order_relaxed))
            frame.min_sp.store(sp, std::memory_order_relaxed);
        if (sp < stack_low && !frame.overflowed.exchange(true)) {
            LOGE("[stack] tid %d ran past the bottom of its guest stack: sp 0x%08X, slot "
                 "0x%08X..0x%08X, in %s called from 0x%08X at depth %d",
                 gettid(), sp, stack_low, stack_low + DynarmicHost::GUEST_STACK_SIZE,
                 name ? name : "?", jit.Regs()[14], depth);
        }
        {
            thread_local int my_tid = gettid();
            if (m_host.WatchedWordOwner() == my_tid) {
                RecordOwnerCall(m_host.WatchedWordEpoch(), name, depth, jit.Regs()[14], sp);
            }
        }
        if (depth == 0) frame.outer_name.store(name ? name : "<unnamed>", std::memory_order_relaxed);
        frame.name.store(name ? name : "<unnamed>", std::memory_order_relaxed);
        (*handler)(jit, m_host);
        frame.name.store(nullptr, std::memory_order_relaxed);
        // A real signal is delivered on the way out of a syscall. The Jit halt alone
        // leaves it undelivered on a thread that spends its life going in and out of
        // short waits, and one such thread hangs Boehm's stop-the-world for good.
        if (GuestSignalPending()) GuestSignalDeliverPending(m_host);
        return;
    }

    LOGE("[THUNK] Unknown thunk ID: %u at PC: 0x%08X", thunk_id, jit.Regs()[15]);
    jit.Regs()[0] = 0;
}

void ThunkManager::SetupGuestJNIEnv() {
    // 232 entries in JNINativeInterface; 256 leaves room and keeps the maths simple.
    uint32_t table_addr = m_host.GuestAlloc(256 * 4, 8);
    m_guest_jni_env = m_host.GuestAlloc(4, 4);
    m_host.MemoryWrite32(m_guest_jni_env, table_addr);

    // Anything not wired below lands on a stub that names its own slot index, so a
    // gap shows up as "JNI slot 123 is not bridged" instead of a silent zero.
    for (int i = 0; i < 256; ++i) {
        std::string name = "JNI_slot_" + std::to_string(i);
        RegisterThunk(name, [i](Dynarmic::A32::Jit& jit, DynarmicHost&) {
            LOGE("JNI slot %d is not bridged", i);
            jit.Regs()[0] = 0;
        });
        m_host.MemoryWrite32(table_addr + i * 4, GetOrCreateThunkAddress(name));
    }

    InstallGuestJNIEnv(*this, m_host, table_addr);
    LOGI("Guest JNIEnv configured at 0x%08X (table at 0x%08X)", m_guest_jni_env, table_addr);
}
