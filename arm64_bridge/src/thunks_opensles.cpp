// SPDX-License-Identifier: GPL-3.0-or-later
// OpenSL ES for the guest.
//
// Only libAkSoundEnginePlugin.so imports it, and it imports exactly four names:
// slCreateEngine plus the data symbols SL_IID_ENGINE, SL_IID_PLAY and
// SL_IID_BUFFERQUEUE. CAkSinkOpenSL::Init/Start then drive the standard Android
// playback path - create engine, realize, get SL_IID_ENGINE, CreateOutputMix,
// CreateAudioPlayer over an Android simple buffer queue, get SL_IID_PLAY and
// SL_IID_BUFFERQUEUE, register a buffer-queue callback, SetPlayState(PLAYING).
// Every method index used below was read off that disassembly, so the tables
// here are not a guess about what Wwise might call.
//
// Nothing is emulated: each call is forwarded to the device's own libOpenSLES,
// which is the same library the working 32-bit build uses. What the bridge has
// to supply is the shape, not the behaviour:
//
//  - OpenSL ES is called as (*obj)->Method(obj, ...), so an object cannot be a
//    handle the way EGLDisplay is. The guest dereferences it. Each object is a
//    guest word holding the address of a guest method table full of trampolines,
//    exactly as guest_jnienv.cpp builds JNIEnv.
//  - SL_IID_* are OBJECT symbols. Resolving a data import to a trampoline makes
//    the guest read an `svc` opcode as the value, so they get real storage with a
//    copy of the host UUID behind it - the same rule as __page_size.
//  - The buffer-queue callback runs on the host's audio thread, which has a small
//    stack and must not block. It only counts a completed buffer; a bridge-owned
//    pump thread (8 MiB, its own Jit) is what actually re-enters the guest.

#include "thunk_manager.h"
#include "dynarmic_host.h"
#include "thunk_types.h"

#include <SLES/OpenSLES.h>
#include <SLES/OpenSLES_Android.h>
#include <android/log.h>

#include <sys/resource.h>
#include <cerrno>

#include <chrono>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#define LOG_TAG "DXTF_OpenSL"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

using Jit = Dynarmic::A32::Jit;

namespace {

enum class Kind { Object, Engine, Play, BufferQueue };

struct GuestItf {
    const void* host = nullptr;  // SLObjectItf / SLEngineItf / SLPlayItf / SLBufferQueueItf
    Kind kind = Kind::Object;
    uint32_t owner = 0;      // guest address of the object this interface came from
    uint32_t callback = 0;   // guest function, buffer queue only
    uint32_t context = 0;    // its pContext
};

std::mutex g_mutex;
std::unordered_map<uint32_t, GuestItf> g_itfs;      // guest object word -> host side
std::unordered_map<const void*, uint32_t> g_by_host;  // host interface -> guest object word
std::unordered_map<uint32_t, SLInterfaceID> g_iids;   // guest UUID address -> host IID

uint32_t g_vtable_object = 0;
uint32_t g_vtable_engine = 0;
uint32_t g_vtable_play = 0;
uint32_t g_vtable_bufferqueue = 0;

// The one buffer queue the sink creates. The pump thread reads it to find the
// guest callback; Destroy clears it so a late wakeup cannot use a freed queue.
uint32_t g_pump_queue = 0;
bool g_pump_started = false;
std::mutex g_pump_mutex;
std::condition_variable g_pump_cv;
int g_pump_pending = 0;

uint32_t VTableFor(Kind kind) {
    switch (kind) {
        case Kind::Engine: return g_vtable_engine;
        case Kind::Play: return g_vtable_play;
        case Kind::BufferQueue: return g_vtable_bufferqueue;
        case Kind::Object: break;
    }
    return g_vtable_object;
}

// Wraps a host interface in a guest object. The guest word is never freed: the
// tables here are keyed by guest address, and the heap hands a freed address to
// the next allocation, which would let an unrelated object inherit this entry.
// A sound engine makes about half a dozen of these for the life of the process.
uint32_t MakeGuestItf(DynarmicHost& host, const void* host_itf, Kind kind, uint32_t owner) {
    if (!host_itf) return 0;
    auto seen = g_by_host.find(host_itf);
    if (seen != g_by_host.end()) return seen->second;

    uint32_t addr = host.GuestAlloc(4, 4);
    if (!addr) {
        LOGE("out of guest heap while wrapping an OpenSL interface");
        return 0;
    }
    host.MemoryWrite32(addr, VTableFor(kind));
    g_itfs[addr] = GuestItf{host_itf, kind, owner, 0, 0};
    g_by_host[host_itf] = addr;
    return addr;
}

bool Lookup(uint32_t guest_addr, GuestItf& out) {
    std::lock_guard<std::mutex> lock(g_mutex);
    auto found = g_itfs.find(guest_addr);
    if (found == g_itfs.end()) return false;
    out = found->second;
    return true;
}

// Every method that takes `self` first needs the same three lines; a miss means
// the guest handed back something that is not one of our objects, which is a
// bridge bug worth naming rather than a silent SL_RESULT_SUCCESS.
#define SL_SELF(itf_var, jit_ref, what)                                              \
    GuestItf itf_var;                                                                \
    if (!Lookup((jit_ref).Regs()[0], itf_var)) {                                     \
        LOGE("%s called on 0x%08X, which is not an OpenSL object", what,             \
             (jit_ref).Regs()[0]);                                                   \
        (jit_ref).Regs()[0] = SL_RESULT_PARAMETER_INVALID;                           \
        return;                                                                      \
    }

// Reads a guest array of SLInterfaceID (guest pointers to our UUID copies).
std::vector<SLInterfaceID> ReadInterfaceIds(DynarmicHost& host, uint32_t addr, uint32_t count) {
    std::vector<SLInterfaceID> ids;
    for (uint32_t i = 0; i < count; ++i) {
        uint32_t guest_iid = host.MemoryRead32(addr + i * 4);
        std::lock_guard<std::mutex> lock(g_mutex);
        auto found = g_iids.find(guest_iid);
        if (found == g_iids.end()) {
            LOGE("interface id at 0x%08X is not one the bridge published", guest_iid);
            ids.push_back(nullptr);
        } else {
            ids.push_back(found->second);
        }
    }
    return ids;
}

std::vector<SLboolean> ReadRequiredFlags(DynarmicHost& host, uint32_t addr, uint32_t count) {
    std::vector<SLboolean> req;
    for (uint32_t i = 0; i < count; ++i) {
        req.push_back(static_cast<SLboolean>(addr ? host.MemoryRead32(addr + i * 4) : SL_BOOLEAN_FALSE));
    }
    return req;
}

// Runs on the device's audio callback thread. Doing the refill here would mean
// running guest code - and Dynarmic's IR passes recurse as deep as the guest's
// own code decides - on a stack the platform sized for a memcpy.
void HostBufferQueueCallback(SLBufferQueueItf, void*) {
    std::lock_guard<std::mutex> lock(g_pump_mutex);
    ++g_pump_pending;
    g_pump_cv.notify_one();
}

void StartPump(DynarmicHost& host) {
    if (g_pump_started) return;
    g_pump_started = true;
    DynarmicHost* h = &host;
    SpawnHostThread([h] {
        // The queue holds four 1024-frame buffers, about 85 ms at 48 kHz, and the
        // refill has to run guest code to get there. At the default priority this
        // thread competes with twenty guest threads and a JIT, and the track
        // reports "disabled due to previous underrun" - which is what the
        // stuttering was. -16 is what Android gives its own audio threads
        // (THREAD_PRIORITY_AUDIO); an app may set it for its own threads.
        if (setpriority(PRIO_PROCESS, 0, -16) != 0) {
            LOGE("[opensl] could not raise the pump thread's priority (errno %d)", errno);
        }
        LOGI("[opensl] buffer queue pump started");
        for (;;) {
            {
                std::unique_lock<std::mutex> lock(g_pump_mutex);
                // Timed, so the thread stays answerable if the queue ever stops
                // calling back - a silent pump and a stopped one look the same
                // from outside otherwise.
                g_pump_cv.wait_for(lock, std::chrono::milliseconds(100),
                                   [] { return g_pump_pending > 0; });
                if (g_pump_pending == 0) continue;
                --g_pump_pending;
            }
            uint32_t queue = 0, callback = 0, context = 0;
            {
                std::lock_guard<std::mutex> lock(g_mutex);
                queue = g_pump_queue;
                auto found = g_itfs.find(queue);
                if (found != g_itfs.end()) {
                    callback = found->second.callback;
                    context = found->second.context;
                }
            }
            if (!callback) continue;
            h->RunFunction(callback, {queue, context});
        }
    });
}

}  // namespace

void ThunkManager::RegisterOpenSLThunks() {
    DynarmicHost& host = m_host;

    // --- SL_IID_* -----------------------------------------------------------
    // The symbol itself is a pointer variable; behind it goes a copy of the
    // host's own 16-byte UUID so that a guest that compares the bytes still
    // sees the right ones.
    auto publish_iid = [&](const char* name, SLInterfaceID iid) {
        uint32_t uuid = host.GuestAlloc(sizeof(SLInterfaceID_), 4);
        memcpy(host.GetGuestMemoryBase() + uuid, iid, sizeof(SLInterfaceID_));
        uint32_t slot = host.GuestAlloc(4, 4);
        host.MemoryWrite32(slot, uuid);
        RegisterSymbol(name, slot);
        g_iids[uuid] = iid;
    };
    publish_iid("SL_IID_ENGINE", SL_IID_ENGINE);
    publish_iid("SL_IID_PLAY", SL_IID_PLAY);
    publish_iid("SL_IID_BUFFERQUEUE", SL_IID_BUFFERQUEUE);
    // Not imported by this build of Wwise, but CreateAudioPlayer over an Android
    // buffer queue is the path it takes, and other Wwise builds ask for this id
    // instead. Publishing it costs 20 bytes and removes a whole failure mode.
    publish_iid("SL_IID_ANDROIDSIMPLEBUFFERQUEUE", SL_IID_ANDROIDSIMPLEBUFFERQUEUE);

    // --- SLObjectItf --------------------------------------------------------
    RegisterThunk("SL$Object$Realize", [](Jit& jit, DynarmicHost&) {
        SL_SELF(itf, jit, "Realize");
        auto obj = static_cast<SLObjectItf>(itf.host);
        SLresult r = (*obj)->Realize(obj, static_cast<SLboolean>(jit.Regs()[1]));
        if (r != SL_RESULT_SUCCESS) LOGE("Realize -> %u", r);
        jit.Regs()[0] = r;
    });

    RegisterThunk("SL$Object$Resume", [](Jit& jit, DynarmicHost&) {
        SL_SELF(itf, jit, "Resume");
        auto obj = static_cast<SLObjectItf>(itf.host);
        jit.Regs()[0] = (*obj)->Resume(obj, static_cast<SLboolean>(jit.Regs()[1]));
    });

    RegisterThunk("SL$Object$GetState", [](Jit& jit, DynarmicHost& host) {
        SL_SELF(itf, jit, "GetState");
        auto obj = static_cast<SLObjectItf>(itf.host);
        SLuint32 state = 0;
        SLresult r = (*obj)->GetState(obj, &state);
        if (jit.Regs()[1]) host.MemoryWrite32(jit.Regs()[1], state);
        jit.Regs()[0] = r;
    });

    RegisterThunk("SL$Object$GetInterface", [](Jit& jit, DynarmicHost& host) {
        SL_SELF(itf, jit, "GetInterface");
        uint32_t self = jit.Regs()[0];
        uint32_t guest_iid = jit.Regs()[1];
        uint32_t out = jit.Regs()[2];

        SLInterfaceID iid = nullptr;
        Kind kind = Kind::Object;
        {
            std::lock_guard<std::mutex> lock(g_mutex);
            auto found = g_iids.find(guest_iid);
            if (found != g_iids.end()) iid = found->second;
        }
        if (!iid) {
            LOGE("GetInterface for id 0x%08X, which the bridge never published", guest_iid);
            jit.Regs()[0] = SL_RESULT_FEATURE_UNSUPPORTED;
            return;
        }
        if (iid == SL_IID_ENGINE) kind = Kind::Engine;
        else if (iid == SL_IID_PLAY) kind = Kind::Play;
        else if (iid == SL_IID_BUFFERQUEUE || iid == SL_IID_ANDROIDSIMPLEBUFFERQUEUE)
            kind = Kind::BufferQueue;

        auto obj = static_cast<SLObjectItf>(itf.host);
        void* host_itf = nullptr;
        SLresult r = (*obj)->GetInterface(obj, iid, &host_itf);
        if (r != SL_RESULT_SUCCESS) {
            LOGE("GetInterface -> %u", r);
            jit.Regs()[0] = r;
            return;
        }
        std::lock_guard<std::mutex> lock(g_mutex);
        uint32_t wrapped = MakeGuestItf(host, host_itf, kind, self);
        if (out) host.MemoryWrite32(out, wrapped);
        LOGI("[opensl] GetInterface -> guest 0x%08X", wrapped);
        jit.Regs()[0] = SL_RESULT_SUCCESS;
    });

    // The object callback only reports asynchronous events. Everything here is
    // realized synchronously (async = SL_BOOLEAN_FALSE), so no event can ever
    // fire; the callback is recorded rather than forwarded, which keeps the host
    // audio thread out of guest code entirely.
    RegisterThunk("SL$Object$RegisterCallback", [](Jit& jit, DynarmicHost&) {
        SL_SELF(itf, jit, "Object RegisterCallback");
        std::lock_guard<std::mutex> lock(g_mutex);
        auto& entry = g_itfs[jit.Regs()[0]];
        entry.callback = jit.Regs()[1];
        entry.context = jit.Regs()[2];
        jit.Regs()[0] = SL_RESULT_SUCCESS;
    });

    RegisterThunk("SL$Object$AbortAsyncOperation", [](Jit& jit, DynarmicHost&) {
        SL_SELF(itf, jit, "AbortAsyncOperation");
        auto obj = static_cast<SLObjectItf>(itf.host);
        (*obj)->AbortAsyncOperation(obj);
    });

    RegisterThunk("SL$Object$Destroy", [](Jit& jit, DynarmicHost&) {
        SL_SELF(itf, jit, "Destroy");
        uint32_t self = jit.Regs()[0];
        auto obj = static_cast<SLObjectItf>(itf.host);
        {
            // Destroy takes the object's interfaces with it, so their entries go
            // first - a stale one would hand a freed host pointer to the next call.
            std::lock_guard<std::mutex> lock(g_mutex);
            for (auto it = g_itfs.begin(); it != g_itfs.end();) {
                if (it->second.owner == self) {
                    if (it->first == g_pump_queue) g_pump_queue = 0;
                    g_by_host.erase(it->second.host);
                    it = g_itfs.erase(it);
                } else {
                    ++it;
                }
            }
            g_by_host.erase(itf.host);
            g_itfs.erase(self);
        }
        (*obj)->Destroy(obj);
        LOGI("[opensl] object 0x%08X destroyed", self);
    });

    RegisterThunk("SL$Object$SetPriority", [](Jit& jit, DynarmicHost&) {
        SL_SELF(itf, jit, "SetPriority");
        auto obj = static_cast<SLObjectItf>(itf.host);
        jit.Regs()[0] = (*obj)->SetPriority(obj, static_cast<SLint32>(jit.Regs()[1]),
                                            static_cast<SLboolean>(jit.Regs()[2]));
    });

    RegisterThunk("SL$Object$GetPriority", [](Jit& jit, DynarmicHost& host) {
        SL_SELF(itf, jit, "GetPriority");
        auto obj = static_cast<SLObjectItf>(itf.host);
        SLint32 priority = 0;
        SLboolean preemptable = SL_BOOLEAN_FALSE;
        SLresult r = (*obj)->GetPriority(obj, &priority, &preemptable);
        if (jit.Regs()[1]) host.MemoryWrite32(jit.Regs()[1], static_cast<uint32_t>(priority));
        if (jit.Regs()[2]) host.MemoryWrite32(jit.Regs()[2], preemptable);
        jit.Regs()[0] = r;
    });

    // --- SLEngineItf --------------------------------------------------------
    RegisterThunk("SL$Engine$CreateAudioPlayer", [](Jit& jit, DynarmicHost& host) {
        SL_SELF(itf, jit, "CreateAudioPlayer");
        uint32_t out_player = jit.Regs()[1];
        uint32_t src_addr = jit.Regs()[2];
        uint32_t snk_addr = jit.Regs()[3];
        uint32_t sp = jit.Regs()[13];
        uint32_t num_itf = host.MemoryRead32(sp);
        uint32_t ids_addr = host.MemoryRead32(sp + 4);
        uint32_t req_addr = host.MemoryRead32(sp + 8);

        // SLDataSource/SLDataSink are two pointers on both sides; the locators
        // behind them are what differ, because one of them holds an SLObjectItf.
        uint32_t src_locator = host.MemoryRead32(src_addr);
        uint32_t src_format = host.MemoryRead32(src_addr + 4);
        uint32_t snk_locator = host.MemoryRead32(snk_addr);
        uint32_t snk_format = host.MemoryRead32(snk_addr + 4);

        SLDataLocator_AndroidSimpleBufferQueue bq_locator{};
        SLDataFormat_PCM pcm{};
        SLDataLocator_OutputMix mix_locator{};
        SLDataSource source{};
        SLDataSink sink{};

        uint32_t src_type = src_locator ? host.MemoryRead32(src_locator) : 0;
        if (src_type != SL_DATALOCATOR_ANDROIDSIMPLEBUFFERQUEUE &&
            src_type != SL_DATALOCATOR_BUFFERQUEUE) {
            LOGE("CreateAudioPlayer source locator 0x%08X is not a buffer queue", src_type);
            jit.Regs()[0] = SL_RESULT_FEATURE_UNSUPPORTED;
            return;
        }
        bq_locator.locatorType = src_type;
        bq_locator.numBuffers = host.MemoryRead32(src_locator + 4);
        source.pLocator = &bq_locator;

        uint32_t fmt_type = src_format ? host.MemoryRead32(src_format) : 0;
        if (fmt_type != SL_DATAFORMAT_PCM) {
            LOGE("CreateAudioPlayer source format 0x%08X is not PCM", fmt_type);
            jit.Regs()[0] = SL_RESULT_CONTENT_UNSUPPORTED;
            return;
        }
        // Seven SLuint32 in the same order on both sides.
        pcm.formatType = fmt_type;
        pcm.numChannels = host.MemoryRead32(src_format + 4);
        pcm.samplesPerSec = host.MemoryRead32(src_format + 8);
        pcm.bitsPerSample = host.MemoryRead32(src_format + 12);
        pcm.containerSize = host.MemoryRead32(src_format + 16);
        pcm.channelMask = host.MemoryRead32(src_format + 20);
        pcm.endianness = host.MemoryRead32(src_format + 24);
        source.pFormat = &pcm;

        uint32_t snk_type = snk_locator ? host.MemoryRead32(snk_locator) : 0;
        if (snk_type != SL_DATALOCATOR_OUTPUTMIX) {
            LOGE("CreateAudioPlayer sink locator 0x%08X is not an output mix", snk_type);
            jit.Regs()[0] = SL_RESULT_FEATURE_UNSUPPORTED;
            return;
        }
        GuestItf mix;
        if (!Lookup(host.MemoryRead32(snk_locator + 4), mix)) {
            LOGE("CreateAudioPlayer sink names 0x%08X, which is not an OpenSL object",
                 host.MemoryRead32(snk_locator + 4));
            jit.Regs()[0] = SL_RESULT_PARAMETER_INVALID;
            return;
        }
        mix_locator.locatorType = SL_DATALOCATOR_OUTPUTMIX;
        mix_locator.outputMix = static_cast<SLObjectItf>(mix.host);
        sink.pLocator = &mix_locator;
        // A sink on an output mix carries no format of its own; the guest passes
        // NULL here and anything else would need a second format marshaller for
        // no caller.
        if (snk_format) LOGE("CreateAudioPlayer sink format 0x%08X ignored", snk_format);
        sink.pFormat = nullptr;

        std::vector<SLInterfaceID> ids = ReadInterfaceIds(host, ids_addr, num_itf);
        std::vector<SLboolean> req = ReadRequiredFlags(host, req_addr, num_itf);

        auto engine = static_cast<SLEngineItf>(itf.host);
        SLObjectItf player = nullptr;
        SLresult r = (*engine)->CreateAudioPlayer(engine, &player, &source, &sink, num_itf,
                                                  ids.empty() ? nullptr : ids.data(),
                                                  req.empty() ? nullptr : req.data());
        LOGI("[opensl] CreateAudioPlayer %u ch @ %u mHz, %u buffers -> %u",
             pcm.numChannels, pcm.samplesPerSec, bq_locator.numBuffers, r);
        if (r != SL_RESULT_SUCCESS) {
            jit.Regs()[0] = r;
            return;
        }
        std::lock_guard<std::mutex> lock(g_mutex);
        uint32_t wrapped = MakeGuestItf(host, player, Kind::Object, 0);
        if (out_player) host.MemoryWrite32(out_player, wrapped);
        jit.Regs()[0] = SL_RESULT_SUCCESS;
    });

    RegisterThunk("SL$Engine$CreateOutputMix", [](Jit& jit, DynarmicHost& host) {
        SL_SELF(itf, jit, "CreateOutputMix");
        uint32_t out_mix = jit.Regs()[1];
        uint32_t num_itf = jit.Regs()[2];
        uint32_t ids_addr = jit.Regs()[3];
        uint32_t req_addr = host.MemoryRead32(jit.Regs()[13]);

        std::vector<SLInterfaceID> ids = ReadInterfaceIds(host, ids_addr, num_itf);
        std::vector<SLboolean> req = ReadRequiredFlags(host, req_addr, num_itf);

        auto engine = static_cast<SLEngineItf>(itf.host);
        SLObjectItf mix = nullptr;
        SLresult r = (*engine)->CreateOutputMix(engine, &mix, num_itf,
                                                ids.empty() ? nullptr : ids.data(),
                                                req.empty() ? nullptr : req.data());
        LOGI("[opensl] CreateOutputMix -> %u", r);
        if (r != SL_RESULT_SUCCESS) {
            jit.Regs()[0] = r;
            return;
        }
        std::lock_guard<std::mutex> lock(g_mutex);
        uint32_t wrapped = MakeGuestItf(host, mix, Kind::Object, 0);
        if (out_mix) host.MemoryWrite32(out_mix, wrapped);
        jit.Regs()[0] = SL_RESULT_SUCCESS;
    });

    // --- SLPlayItf ----------------------------------------------------------
    RegisterThunk("SL$Play$SetPlayState", [](Jit& jit, DynarmicHost&) {
        SL_SELF(itf, jit, "SetPlayState");
        auto play = static_cast<SLPlayItf>(itf.host);
        SLresult r = (*play)->SetPlayState(play, jit.Regs()[1]);
        LOGI("[opensl] SetPlayState(%u) -> %u", jit.Regs()[1], r);
        jit.Regs()[0] = r;
    });

    RegisterThunk("SL$Play$GetPlayState", [](Jit& jit, DynarmicHost& host) {
        SL_SELF(itf, jit, "GetPlayState");
        auto play = static_cast<SLPlayItf>(itf.host);
        SLuint32 state = 0;
        SLresult r = (*play)->GetPlayState(play, &state);
        if (jit.Regs()[1]) host.MemoryWrite32(jit.Regs()[1], state);
        jit.Regs()[0] = r;
    });

    // --- SLBufferQueueItf ---------------------------------------------------
    RegisterThunk("SL$BufferQueue$Enqueue", [](Jit& jit, DynarmicHost& host) {
        SL_SELF(itf, jit, "Enqueue");
        // Whoever calls Enqueue is on the audio path - either the pump or Wwise's
        // own mixing thread. The guest asks for a priority through
        // pthread_attr_setschedparam, which the bridge does not carry across, and
        // this is the one place the audio thread names itself.
        thread_local bool boosted = false;
        if (!boosted) {
            boosted = true;
            if (setpriority(PRIO_PROCESS, 0, -16) != 0)
                LOGE("[opensl] could not raise an audio thread's priority (errno %d)", errno);
        }
        auto queue = static_cast<SLBufferQueueItf>(itf.host);
        // The guest arena is ordinary host memory, so the mixer can read the
        // guest's own buffer; Wwise does not touch it again until the callback
        // says that buffer is done, which is the contract OpenSL already has.
        void* buffer = GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[1]);
        SLresult r = (*queue)->Enqueue(queue, buffer, jit.Regs()[2]);
        static uint64_t enqueued = 0;
        if (++enqueued <= 4 || enqueued % 500 == 0) {
            // diag: the peak of the buffer itself. "Audio is running" and "audio is
            // audible" are different claims, and only this one separates a mix that
            // never reached the queue from a queue that is playing silence.
            const int16_t* pcm = static_cast<const int16_t*>(buffer);
            int32_t peak = 0;
            for (uint32_t i = 0; pcm && i < jit.Regs()[2] / 2; ++i) {
                int32_t v = pcm[i] < 0 ? -pcm[i] : pcm[i];
                if (v > peak) peak = v;
            }
            LOGI("[opensl] Enqueue #%llu %u bytes peak %d -> %u", (unsigned long long)enqueued,
                 jit.Regs()[2], peak, r);
        }
        jit.Regs()[0] = r;
    });

    RegisterThunk("SL$BufferQueue$Clear", [](Jit& jit, DynarmicHost&) {
        SL_SELF(itf, jit, "Clear");
        auto queue = static_cast<SLBufferQueueItf>(itf.host);
        jit.Regs()[0] = (*queue)->Clear(queue);
    });

    RegisterThunk("SL$BufferQueue$GetState", [](Jit& jit, DynarmicHost& host) {
        SL_SELF(itf, jit, "BufferQueue GetState");
        auto queue = static_cast<SLBufferQueueItf>(itf.host);
        SLBufferQueueState state{};
        SLresult r = (*queue)->GetState(queue, &state);
        if (jit.Regs()[1]) {
            host.MemoryWrite32(jit.Regs()[1], state.count);
            host.MemoryWrite32(jit.Regs()[1] + 4, state.playIndex);
        }
        jit.Regs()[0] = r;
    });

    RegisterThunk("SL$BufferQueue$RegisterCallback", [](Jit& jit, DynarmicHost& host) {
        SL_SELF(itf, jit, "BufferQueue RegisterCallback");
        uint32_t self = jit.Regs()[0];
        {
            std::lock_guard<std::mutex> lock(g_mutex);
            auto& entry = g_itfs[self];
            entry.callback = jit.Regs()[1];
            entry.context = jit.Regs()[2];
            g_pump_queue = self;
        }
        auto queue = static_cast<SLBufferQueueItf>(itf.host);
        SLresult r = (*queue)->RegisterCallback(queue, HostBufferQueueCallback, nullptr);
        LOGI("[opensl] buffer queue callback 0x%08X ctx 0x%08X -> %u", jit.Regs()[1],
             jit.Regs()[2], r);
        if (r == SL_RESULT_SUCCESS) StartPump(host);
        jit.Regs()[0] = r;
    });

    // --- the method tables --------------------------------------------------
    // A slot with no name gets a stub that says which interface and index it was,
    // so a gap reads as "OpenSL SLEngineItf slot 4 is not bridged" rather than as
    // a success the caller then acts on.
    auto build = [&](const char* iface, const std::vector<const char*>& methods) {
        uint32_t table = host.GuestAlloc(methods.size() * 4, 4);
        for (size_t i = 0; i < methods.size(); ++i) {
            std::string name;
            if (methods[i]) {
                name = methods[i];
            } else {
                name = std::string("SL$") + iface + "$slot" + std::to_string(i);
                std::string label = iface;
                int index = static_cast<int>(i);
                RegisterThunk(name, [label, index](Jit& jit, DynarmicHost&) {
                    LOGE("OpenSL %s slot %d is not bridged", label.c_str(), index);
                    jit.Regs()[0] = SL_RESULT_FEATURE_UNSUPPORTED;
                });
            }
            host.MemoryWrite32(table + i * 4, GetOrCreateThunkAddress(name));
        }
        return table;
    };

    g_vtable_object = build("SLObjectItf", {
        "SL$Object$Realize", "SL$Object$Resume", "SL$Object$GetState",
        "SL$Object$GetInterface", "SL$Object$RegisterCallback",
        "SL$Object$AbortAsyncOperation", "SL$Object$Destroy",
        "SL$Object$SetPriority", "SL$Object$GetPriority", nullptr,
    });
    g_vtable_engine = build("SLEngineItf", {
        nullptr, nullptr, "SL$Engine$CreateAudioPlayer", nullptr, nullptr, nullptr,
        nullptr, "SL$Engine$CreateOutputMix", nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr,
    });
    g_vtable_play = build("SLPlayItf", {
        "SL$Play$SetPlayState", "SL$Play$GetPlayState", nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
    });
    g_vtable_bufferqueue = build("SLBufferQueueItf", {
        "SL$BufferQueue$Enqueue", "SL$BufferQueue$Clear", "SL$BufferQueue$GetState",
        "SL$BufferQueue$RegisterCallback",
    });

    // --- slCreateEngine -----------------------------------------------------
    RegisterThunk("slCreateEngine", [](Jit& jit, DynarmicHost& host) {
        uint32_t out_engine = jit.Regs()[0];
        uint32_t num_options = jit.Regs()[1];
        uint32_t options_addr = jit.Regs()[2];
        uint32_t num_itf = jit.Regs()[3];
        uint32_t sp = jit.Regs()[13];
        uint32_t ids_addr = host.MemoryRead32(sp);
        uint32_t req_addr = host.MemoryRead32(sp + 4);

        // SLEngineOption is two SLuint32 on both sides.
        std::vector<SLEngineOption> options;
        for (uint32_t i = 0; i < num_options; ++i) {
            SLEngineOption option{};
            option.feature = host.MemoryRead32(options_addr + i * 8);
            option.data = host.MemoryRead32(options_addr + i * 8 + 4);
            options.push_back(option);
        }
        std::vector<SLInterfaceID> ids = ReadInterfaceIds(host, ids_addr, num_itf);
        std::vector<SLboolean> req = ReadRequiredFlags(host, req_addr, num_itf);

        SLObjectItf engine = nullptr;
        SLresult r = slCreateEngine(&engine, num_options,
                                    options.empty() ? nullptr : options.data(), num_itf,
                                    ids.empty() ? nullptr : ids.data(),
                                    req.empty() ? nullptr : req.data());
        LOGI("[opensl] slCreateEngine -> %u", r);
        if (r != SL_RESULT_SUCCESS) {
            jit.Regs()[0] = r;
            return;
        }
        std::lock_guard<std::mutex> lock(g_mutex);
        uint32_t wrapped = MakeGuestItf(host, engine, Kind::Object, 0);
        if (out_engine) host.MemoryWrite32(out_engine, wrapped);
        jit.Regs()[0] = SL_RESULT_SUCCESS;
    });

    LOGI("OpenSL ES thunks registered");
}
