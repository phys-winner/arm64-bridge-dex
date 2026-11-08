// SPDX-License-Identifier: GPL-3.0-or-later
#include "dynarmic_host.h"
#include "elf_loader.h"
#include "thunk_manager.h"
#include "bridge_config.h"
#include "guest_signals.h"
#include "guest_cp15.h"
#include "thunk_types.h"
#include <unistd.h>
#include <cstring>
#include <set>
#include <map>
#include <vector>
#include <algorithm>
#include <thread>
#include <android/log.h>

#if defined(_WIN32)
#include <windows.h>
#else
#include <sys/mman.h>
#endif

#define LOG_TAG "GOF2_ARM64_Loader"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

#define EXIT_TRAP_ARM   0x00000004
#define EXIT_TRAP_THUMB 0x00000008

// An SVC number no thunk will ever be given.
static constexpr uint32_t kNullCallSvc = 0x00FFFFFF;
// Nothing but the two exit traps lives below this; the trampolines start at 0x10000.
static constexpr uint32_t kTrapPageEnd = 0x40;

namespace {

// Guest thread stacks live between the thunk trampolines and the library load base
// at 0x40000000, which nothing else claims. Guest thread 0 keeps the original top.
constexpr uint32_t kThreadStackTop  = 0x40000000;
constexpr uint32_t kThreadStackSize = DynarmicHost::GUEST_STACK_SIZE;

// Per-host-thread guest execution state. There is only ever one DynarmicHost, so
// these can be plain thread_locals rather than a map keyed by host thread id.
struct GuestThreadState {
    Dynarmic::A32::Jit* jit = nullptr;     // this thread's outermost Jit
    Dynarmic::A32::Jit* active = nullptr;  // innermost one currently executing
    uint32_t stack_top = 0;
    bool running = false;
    // Jits kept for nested guest calls made from inside a thunk, one per depth.
    std::vector<Dynarmic::A32::Jit*> nested;
    int depth = 0;
};
thread_local GuestThreadState t_thread;

}  // namespace

DynarmicHost::DynarmicHost() = default;

DynarmicHost::~DynarmicHost() {
    if (m_memory) {
#if defined(_WIN32)
        VirtualFree(m_memory, 0, MEM_RELEASE);
#else
        munmap(m_memory, m_memory_size);
#endif
        m_memory = nullptr;
    }
}

// Reserve guest memory. MAP_NORESERVE keeps the 4 GiB reservation virtual: untouched pages
// cost nothing and only written pages are ever committed.
static uint8_t* MapGuestMemory(size_t size) {
#if defined(_WIN32)
    return reinterpret_cast<uint8_t*>(VirtualAlloc(nullptr, size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
#else
    void* p = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    return p == MAP_FAILED ? nullptr : reinterpret_cast<uint8_t*>(p);
#endif
}

bool DynarmicHost::Initialize() {
    // Mapping the whole 32-bit space lets the JIT emit a single `ldr Wt, [Xfastmem, Waddr, uxtw]`
    // per guest access instead of a call into MemoryRead*/MemoryWrite*.
    m_memory_size = GUEST_ADDRESS_SPACE;
    m_memory = MapGuestMemory(m_memory_size);

    bool fastmem = m_memory != nullptr && GetBridgeConfig().fastmem != 0;
    if (m_memory == nullptr) {
        LOGE("Could not reserve %zu bytes for the guest address space; "
             "falling back to %zu bytes without fastmem.", GUEST_ADDRESS_SPACE, GUEST_FALLBACK_SIZE);
        m_memory_size = GUEST_FALLBACK_SIZE;
        m_memory = MapGuestMemory(m_memory_size);
        if (!m_memory) {
            LOGE("Failed to allocate guest memory (%zu bytes)", m_memory_size);
            m_memory_size = 0;
            return false;
        }
    }

    // A call through a null function pointer lands on address zero and falls
    // through into the exit trap, which looks exactly like the function returning
    // normally - the thread just quietly ends. Claiming address zero turns that
    // into a log line naming the caller.
    *reinterpret_cast<uint32_t*>(m_memory + 0) = 0xEF000000 | kNullCallSvc;

    // ARM exit trap at 0x00000004: SVC 0 (0xEF000000)
    *reinterpret_cast<uint32_t*>(m_memory + EXIT_TRAP_ARM) = 0xEF000000;

    // Thumb exit trap at 0x00000008: SVC 0 (0xDF00, 0xDF00)
    *reinterpret_cast<uint32_t*>(m_memory + EXIT_TRAP_THUMB) = 0xDF00DF00;

    m_thunk_manager = std::make_unique<ThunkManager>(*this);
    m_thunk_manager->RegisterAllThunks();

    // One monitor shared by every guest thread; processor_id indexes into it, so it
    // has to be at least as large as the number of Jits that will ever exist.
    m_monitor = std::make_unique<Dynarmic::ExclusiveMonitor>(kMaxJits);

    m_jit_config = {};
    m_jit_config.callbacks = this;
    m_jit_config.global_monitor = m_monitor.get();
    m_jit_config.arch_version = Dynarmic::A32::ArchVersion::v7;  // guest libs are ARMv7-A + NEON
    m_jit_config.enable_cycle_counting = false;                  // no timing model; skips AddTicks emission
    m_jit_config.always_little_endian = true;                    // guest never uses SETEND; drops CPSR.E handling
    m_jit_config.define_unpredictable_behaviour = true;          // define rather than halt on odd encodings
    // Both bionic and Mono's generated code read the thread pointer out of CP15.
    // With no coprocessor attached Dynarmic turns that into an assertion inside
    // its emitter, which aborts the process rather than raising anything the
    // guest could see.
    // Every slot gets one: an MCR to a coprocessor with no handler is an
    // assertion inside the emitter, and which coprocessor the guest reaches for
    // is not knowable up front. Only 15 does anything; the rest report
    // themselves once and behave as a register that reads zero.
    // 10 and 11 are VFP/NEON and the frontend decodes those itself - but only the
    // encodings it knows. Anything it does not recognise still arrives here, so
    // those slots cannot be left empty either.
    for (unsigned n = 0; n < 16; ++n) {
        m_coprocessors[n] = std::make_shared<GuestCp15>(*this, n);
        m_jit_config.coprocessors[n] = m_coprocessors[n];
    }
    if (fastmem) {
        m_jit_config.fastmem_pointer = reinterpret_cast<uintptr_t>(m_memory);
    }

    CurrentJit();  // the calling thread becomes guest thread 0
    StartWordWatch();
    LOGI("Dynarmic Host initialized: %zu bytes guest space, fastmem %s",
         m_memory_size, fastmem ? "ON" : "OFF");
    return true;
}

bool DynarmicHost::LoadGame(const std::string& elf_path) {
    LOGI("Loading main game ELF from file: %s", elf_path.c_str());
    m_elf_loader = std::make_unique<ElfLoader>(m_memory, 0x40000000);
    if (!m_elf_loader->Load(elf_path)) {
        LOGE("Failed to load ELF: %s", elf_path.c_str());
        return false;
    }

    auto resolver = [this](const std::string& name) -> uint32_t {
        for (auto it = m_libs.rbegin(); it != m_libs.rend(); ++it) {
            uint32_t addr = it->second->GetSymbolAddress(name);
            if (addr != 0) return addr;
        }
        if (m_elf_loader) {
            uint32_t addr = m_elf_loader->GetSymbolAddress(name);
            if (addr != 0) return addr;
        }
        return m_thunk_manager->GetOrCreateThunkAddress(name);
    };

    if (!m_elf_loader->Relocate(resolver)) {
        LOGE("Failed to relocate ELF.");
        return false;
    }

    LOGI("Running %zu static initializers in .init_array...", m_elf_loader->GetInitArray().size());
    for (uint32_t init_func : m_elf_loader->GetInitArray()) {
        RunFunction(init_func);
    }

    LOGI("Game ELF loaded and initialized successfully!");
    return true;
}

bool DynarmicHost::LoadGameFromMemory(const uint8_t* data, size_t size) {
    LOGI("Loading main game ELF from memory (%zu bytes)...", size);
    m_elf_loader = std::make_unique<ElfLoader>(m_memory, 0x40000000);
    if (!m_elf_loader->LoadFromMemory(data, size)) {
        LOGE("Failed to load ELF from memory buffer");
        return false;
    }

    auto resolver = [this](const std::string& name) -> uint32_t {
        for (auto it = m_libs.rbegin(); it != m_libs.rend(); ++it) {
            uint32_t addr = it->second->GetSymbolAddress(name);
            if (addr != 0) return addr;
        }
        if (m_elf_loader) {
            uint32_t addr = m_elf_loader->GetSymbolAddress(name);
            if (addr != 0) return addr;
        }
        return m_thunk_manager->GetOrCreateThunkAddress(name);
    };

    if (!m_elf_loader->Relocate(resolver)) {
        LOGE("Failed to relocate ELF.");
        return false;
    }

    LOGI("Running %zu static initializers in .init_array...", m_elf_loader->GetInitArray().size());
    for (uint32_t init_func : m_elf_loader->GetInitArray()) {
        RunFunction(init_func);
    }

    LOGI("Game ELF loaded from memory and initialized successfully!");
    return true;
}

bool DynarmicHost::LoadGuestLibraries(const std::vector<GuestLibrary>& libs) {
    if (libs.empty()) return false;

    // Each image gets its own 32 MiB slot so relocations never overlap.
    uint32_t base = 0x40000000;
    for (const auto& lib : libs) {
        auto loader = std::make_unique<ElfLoader>(m_memory, base);
        if (!loader->Load(lib.path)) {
            LOGE("Failed to load %s from %s", lib.name.c_str(), lib.path.c_str());
            return false;
        }
        LOGI("Loaded %s at 0x%08x (%u bytes)", lib.name.c_str(), base, loader->GetImageSize());
        m_libs.emplace_back(lib.name, std::move(loader));
        base += 0x02000000;
    }

    // Undefined symbols resolve against every loaded image first, then fall back to a
    // thunk trampoline. Search order is reverse of load order so a later library can
    // override an earlier one, matching how the dynamic linker would behave here.
    // Anything that reaches the thunk fallback is an import the bridge has to stand
    // in for. Listing them once is the difference between "the game misbehaves" and
    // "these 40 symbols are missing" - a data symbol landing here is silently fatal,
    // because the guest then reads a trampoline instruction as its value.
    std::set<std::string> unresolved;
    auto resolver = [this, &unresolved](const std::string& name) -> uint32_t {
        for (auto it = m_libs.rbegin(); it != m_libs.rend(); ++it) {
            uint32_t addr = it->second->GetSymbolAddress(name);
            if (addr != 0) return addr;
        }
        if (!m_thunk_manager->IsRegistered(name)) unresolved.insert(name);
        return m_thunk_manager->GetOrCreateThunkAddress(name);
    };

    for (auto& entry : m_libs) {
        if (!entry.second->Relocate(resolver)) {
            LOGE("Failed to relocate %s", entry.first.c_str());
            return false;
        }
    }

    // Mono reaches most metadata through bsearch. The two comparators below are
    // tiny, but calling either as guest code creates a second Dynarmic Jit inside
    // the libc thunk. Two rare cold-start stalls were captured in exactly those
    // nested calls. Execute the pure metadata readers and comparators directly
    // against the guest arena: that removes the fragile nesting point and avoids
    // translating thousands of prologues during startup.
    auto install_fast_path_at = [this](uint32_t address, const char* label,
                                       const char* thunk_name, uint32_t expected0,
                                       uint32_t expected1, ThunkHandler handler) {
        if (!address) {
            LOGE("Mono fast path skipped: %s was not found", label);
            return false;
        }
        uint32_t original0 = MemoryRead32(address);
        uint32_t original1 = MemoryRead32(address + 4);
        if (original0 != expected0 || original1 != expected1) {
            LOGE("Mono fast path skipped for %s: unexpected prologue %08X %08X",
                 label, original0, original1);
            return false;
        }
        m_thunk_manager->RegisterThunk(thunk_name, std::move(handler));
        uint32_t trampoline = m_thunk_manager->GetOrCreateThunkAddress(thunk_name);
        MemoryWrite32(address, MemoryRead32(trampoline));  // SVC #thunk-id
        MemoryWrite32(address + 4, 0xE12FFF1E);             // bx lr
        InvalidateGuestCode(address, 8);
        LOGI("Mono fast path installed: %s at 0x%08X", label, address);
        return true;
    };
    auto install_fast_path = [this, &install_fast_path_at](
                                 const char* symbol, const char* thunk_name,
                                 uint32_t expected0, uint32_t expected1,
                                 ThunkHandler handler) {
        return install_fast_path_at(FindSymbol(symbol), symbol, thunk_name,
                                    expected0, expected1, std::move(handler));
    };

    auto read16 = [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        uint32_t address = jit.Regs()[0];
        jit.Regs()[0] = static_cast<uint64_t>(address) + 2 <= host.GetGuestMemorySize()
                            ? host.MemoryRead16(address)
                            : 0;
    };
    auto read32 = [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        uint32_t address = jit.Regs()[0];
        jit.Regs()[0] = static_cast<uint64_t>(address) + 4 <= host.GetGuestMemorySize()
                            ? host.MemoryRead32(address)
                            : 0;
    };
    auto decode_metadata = [](DynarmicHost& host, uint32_t table, uint32_t row,
                              uint32_t column, uint32_t& value) -> const char* {
        if (static_cast<uint64_t>(table) + 12 > host.GetGuestMemorySize()) {
            return "table is outside guest memory";
        }
        const uint32_t base = host.MemoryRead32(table);
        const uint32_t rows_and_size = host.MemoryRead32(table + 4);
        const uint32_t layout = host.MemoryRead32(table + 8);
        const uint32_t rows = rows_and_size & 0x00FFFFFFu;
        const uint32_t row_size = host.MemoryRead8(table + 7);
        const uint32_t columns = layout >> 24;
        if (row >= rows || column >= columns || column >= 12 || row_size == 0) {
            return "row or column is outside the metadata table";
        }

        uint64_t address = static_cast<uint64_t>(base) +
                           static_cast<uint64_t>(row) * row_size;
        for (uint32_t i = 0; i < column; ++i)
            address += ((layout >> (i * 2)) & 3u) + 1u;
        const uint32_t width = ((layout >> (column * 2)) & 3u) + 1u;
        if (address + width > host.GetGuestMemorySize()) {
            return "column data is outside guest memory";
        }
        switch (width) {
            case 1: value = host.MemoryRead8(static_cast<uint32_t>(address)); break;
            case 2: value = host.MemoryRead16(static_cast<uint32_t>(address)); break;
            case 4: value = host.MemoryRead32(static_cast<uint32_t>(address)); break;
            default: return "metadata column has unsupported width";
        }
        return nullptr;
    };
    auto report_metadata_error = [](const char* operation, uint32_t table,
                                    uint32_t row, uint32_t column, const char* why) {
        static std::atomic<uint32_t> reports{0};
        if (reports.fetch_add(1, std::memory_order_relaxed) < 8) {
            LOGE("%s(%08X, %u, %u): %s", operation, table, row, column, why);
        }
    };
    auto decode_row_col = [decode_metadata, report_metadata_error](
                              Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const uint32_t table = jit.Regs()[0];
        const uint32_t row = jit.Regs()[1];
        const uint32_t column = jit.Regs()[2];
        uint32_t value = 0;
        if (const char* why = decode_metadata(host, table, row, column, value)) {
            report_metadata_error("mono_metadata_decode_row_col", table, row, column, why);
        }
        jit.Regs()[0] = value;
    };

    auto metadata_row = [](DynarmicHost& host, uint32_t key, uint32_t elem,
                           uint32_t& table, uint32_t& row, uint32_t& column,
                           uint32_t& target) -> const char* {
        if (static_cast<uint64_t>(key) + 16 > host.GetGuestMemorySize())
            return "search key is outside guest memory";
        target = host.MemoryRead32(key);
        column = host.MemoryRead32(key + 4);
        table = host.MemoryRead32(key + 8);
        if (static_cast<uint64_t>(table) + 12 > host.GetGuestMemorySize())
            return "table is outside guest memory";
        const uint32_t base = host.MemoryRead32(table);
        const uint32_t row_size = host.MemoryRead8(table + 7);
        if (!row_size || elem < base) return "element is outside the metadata table";
        row = (elem - base) / row_size;
        return nullptr;
    };

    // Hidden comparator at libmono .text+0x1CA220. It finds the metadata row
    // whose start value is <= the key and whose next start value is > the key.
    auto compare_metadata_range = [decode_metadata, metadata_row, report_metadata_error](
                                      Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const uint32_t key = jit.Regs()[0];
        uint32_t table = 0, row = 0, column = 0, target = 0, current = 0;
        const char* why = metadata_row(host, key, jit.Regs()[1], table, row, column, target);
        if (!why) why = decode_metadata(host, table, row, column, current);
        if (why) {
            report_metadata_error("mono_metadata_range_compare", table, row, column, why);
            jit.Regs()[0] = 0;
            return;
        }
        if (target < current) {
            jit.Regs()[0] = 0xFFFFFFFFu;
            return;
        }
        const uint32_t rows = host.MemoryRead32(table + 4) & 0x00FFFFFFu;
        if (row + 1 < rows) {
            uint32_t next = 0;
            if (const char* next_why = decode_metadata(host, table, row + 1, column, next)) {
                report_metadata_error("mono_metadata_range_compare", table, row + 1,
                                      column, next_why);
                jit.Regs()[0] = 0;
                return;
            }
            if (target >= next || current == next) {
                jit.Regs()[0] = 1;
                return;
            }
        }
        host.MemoryWrite32(key + 12, row);
        jit.Regs()[0] = 0;
    };

    // Hidden comparator at libmono .text+0x1CA35C. It is the exact-match form
    // used by custom-attribute metadata lookups.
    auto compare_metadata_exact = [decode_metadata, metadata_row, report_metadata_error](
                                      Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const uint32_t key = jit.Regs()[0];
        uint32_t table = 0, row = 0, column = 0, target = 0, current = 0;
        const char* why = metadata_row(host, key, jit.Regs()[1], table, row, column, target);
        if (!why) why = decode_metadata(host, table, row, column, current);
        if (why) {
            report_metadata_error("mono_metadata_exact_compare", table, row, column, why);
            jit.Regs()[0] = 0;
            return;
        }
        if (target == current) {
            host.MemoryWrite32(key + 12, row);
            jit.Regs()[0] = 0;
        } else {
            jit.Regs()[0] = target < current ? 0xFFFFFFFFu : 1u;
        }
    };

    const uint32_t metadata_decoder = FindSymbol("mono_metadata_decode_row_col");
    install_fast_path("mono_read16", "dxtf_fast_mono_read16",
                      0xE52DB004, 0xE28DB000, read16);
    install_fast_path("mono_read32", "dxtf_fast_mono_read32",
                      0xE52DB004, 0xE28DB000, read32);
    install_fast_path("mono_metadata_decode_row_col", "dxtf_fast_mono_decode_row_col",
                      0xE92D4830, 0xE28DB00C, decode_row_col);
    if (metadata_decoder) {
        // Relative positions and prologues are checked against this exact libmono
        // before anything is overwritten; a different build safely keeps guest code.
        install_fast_path_at(metadata_decoder + 0x5424, "mono_metadata_range_compare",
                             "dxtf_fast_mono_metadata_range_compare",
                             0xE92D4800, 0xE28DB004, compare_metadata_range);
        install_fast_path_at(metadata_decoder + 0x5560, "mono_metadata_exact_compare",
                             "dxtf_fast_mono_metadata_exact_compare",
                             0xE92D4800, 0xE28DB004, compare_metadata_exact);
    }

    LOGI("%zu imports have no implementation:", unresolved.size());
    for (const std::string& name : unresolved) LOGI("  unresolved: %s", name.c_str());

    for (auto& entry : m_libs) {
        const auto& init = entry.second->GetInitArray();
        LOGI("Running %zu initializers for %s...", init.size(), entry.first.c_str());
        for (uint32_t fn : init) RunFunction(fn);
    }

    LOGI("All %zu guest libraries loaded, relocated and initialized", m_libs.size());
    return true;
}

ElfLoader* DynarmicHost::GetLibrary(const std::string& name) {
    for (auto& entry : m_libs) {
        if (entry.first == name) return entry.second.get();
    }
    return nullptr;
}

uint32_t DynarmicHost::FindSymbol(const std::string& name) {
    for (auto it = m_libs.rbegin(); it != m_libs.rend(); ++it) {
        uint32_t addr = it->second->GetSymbolAddress(name);
        if (addr) return addr;
    }
    if (m_elf_loader) return m_elf_loader->GetSymbolAddress(name);
    return 0;
}

bool DynarmicHost::IsJitRunning() const { return t_thread.running; }

int DynarmicHost::NestingDepth() const { return t_thread.depth; }

Dynarmic::A32::Jit* DynarmicHost::MakeJit(uint32_t* out_index) {
    BridgeSetPhase("makejit-lock");
    std::lock_guard<std::mutex> lock(m_jits_mutex);
    BridgeSetPhase("makejit");
    const uint32_t index = static_cast<uint32_t>(m_jits.size());
    if (index >= kMaxJits) {
        LOGE("more than %zu Jits; ldrex/strex would index past the exclusive monitor",
             kMaxJits);
    }
    Dynarmic::A32::UserConfig config = m_jit_config;
    config.processor_id = index < kMaxJits ? index : static_cast<uint32_t>(kMaxJits - 1);
    if (index == 0 || (index % 16) == 0) LOGI("guest Jit #%u created", index);
    m_jits.push_back(std::make_unique<Dynarmic::A32::Jit>(config));
    if (out_index) *out_index = index;
    return m_jits.back().get();
}

Dynarmic::A32::Jit& DynarmicHost::CurrentJit() {
    // During a nested call the innermost Jit is the one whose registers a thunk
    // must read and write. Returning the thread's outer Jit here would make the
    // guest comparator's arguments come from the frame that is suspended.
    if (t_thread.active) return *t_thread.active;
    if (t_thread.jit) return *t_thread.jit;

    // Stack slots are counted separately from Jits: nested-call Jits share the
    // stack of the thread that owns them, and numbering by m_jits.size() would
    // hand out a slot to each of them and shift every later thread's stack.
    //
    // ponytail: slots are never recycled, so a thread that exits keeps its 1 MiB
    // and its Jit for the life of the process. A 3.5 minute run reaches slot 35 of
    // kMaxGuestThreads; past 64 the region below kThreadStackTop is only free
    // because nothing else is mapped there, not because anything checks. Recycle
    // on thread exit if a longer session ever walks past the cap.
    const uint32_t slot = m_next_stack_slot++;
    t_thread.jit = MakeJit(nullptr);
    t_thread.active = t_thread.jit;
    t_thread.stack_top = kThreadStackTop - slot * kThreadStackSize;
    // Boehm addresses threads by the value gettid() gave it, so the signal table
    // is keyed the same way.
    GuestSignalRegisterThread(gettid(), t_thread.jit);
    LOGI("guest thread %u attached: stack top 0x%08X", slot, t_thread.stack_top);
    return *t_thread.jit;
}

// Runs guest code from inside a thunk, i.e. from inside Jit::Run.
//
// Jit::Run is not re-entrant, but that is a property of the Jit object, not of
// the thread - so the nested call gets a Jit of its own. Without this, any libc
// function that takes a guest callback has to be stubbed out, and a stubbed
// bsearch is not "slower", it answers "not found" to every metadata lookup Mono
// makes.
//
// The nested stack is carved below the suspended frame's own stack pointer,
// which is memory the interrupted function has already finished with.
// Clears every halt reason Run came back for and does what each one asks. Returns
// true if the Jit should be re-entered.
//
// One reason per return trip is the wrong shape: Run reports whatever accumulated
// while it was inside, and handling only the first of them leaves the rest set. The
// next Run then returns for those instead of executing anything - and if the first
// reason keeps arriving, the rest never get their turn at all.
bool DynarmicHost::HandleHalt(Dynarmic::A32::Jit& jit, Dynarmic::HaltReason hr) {
    bool resume = false;
    if (Dynarmic::Has(hr, Dynarmic::HaltReason::CacheInvalidation)) {
        jit.ClearHalt(Dynarmic::HaltReason::CacheInvalidation);
        resume = true;
    }
    if (Dynarmic::Has(hr, Dynarmic::HaltReason::UserDefined3)) {
        jit.ClearHalt(Dynarmic::HaltReason::UserDefined3);
        LOGE("[pc] tid %d depth %d pc=0x%08X lr=0x%08X sp=0x%08X", gettid(), t_thread.depth,
             jit.Regs()[15], jit.Regs()[14], jit.Regs()[13]);
        resume = true;
    }
    // A signal can land while a nested call is running - the restart signal arrives
    // exactly like this, while the suspend handler is still on the stack. Delivering
    // it costs one more nesting level, which the depth cap bounds.
    if (Dynarmic::Has(hr, Dynarmic::HaltReason::UserDefined2)) {
        jit.ClearHalt(Dynarmic::HaltReason::UserDefined2);
        resume = true;
    }
    // Unconditional, not tied to the flag: the flag can be cleared by whichever Run
    // loop on this thread saw it first, while the queue entry it was announcing is
    // still there.
    if (GuestSignalPending()) GuestSignalDeliverPending(*this);
    return resume;
}

uint32_t DynarmicHost::CallGuestNested(uint32_t target_pc, const std::vector<uint32_t>& args) {
    constexpr int kMaxNestedDepth = 8;

    if (!target_pc) return 0;
    if (t_thread.depth >= kMaxNestedDepth) {
        LOGE("nested guest call to 0x%08X refused at depth %d", target_pc, t_thread.depth);
        return 0;
    }

    Dynarmic::A32::Jit* outer = t_thread.active ? t_thread.active : &CurrentJit();
    uint32_t outer_sp = outer->Regs()[13];

    const int depth = t_thread.depth;
    while (static_cast<int>(t_thread.nested.size()) <= depth) {
        t_thread.nested.push_back(MakeJit(nullptr));
        GuestSignalAddThreadJit(t_thread.nested.back());
    }
    Dynarmic::A32::Jit* jit = t_thread.nested[depth];

    // Just below the suspended frame: bytes under its stack pointer are dead, and
    // each deeper nested call starts below the one before it, so the frames never
    // overlap without reserving a slice per depth. Keeping the gap small matters
    // for signal handlers - Boehm records the handler's stack pointer as the top
    // of the region it scans conservatively.
    uint32_t sp = (outer_sp - 256) & ~7u;
    for (size_t i = 0; i < args.size() && i < 4; ++i) jit->Regs()[i] = args[i];
    // Arguments past the fourth go on the stack, same as RunFunction. Dropping them
    // is how nativeResize(env, self, w, h, fw, fh) used to arrive without its last
    // two words.
    if (args.size() > 4) {
        size_t extra_args = args.size() - 4;
        sp = (sp - static_cast<uint32_t>(extra_args * 4)) & ~7u;
        for (size_t i = 0; i < extra_args; ++i) {
            *reinterpret_cast<uint32_t*>(m_memory + sp + i * 4) = args[4 + i];
        }
    }
    jit->Regs()[13] = sp;

    bool is_thumb = (target_pc & 1) != 0;
    jit->Regs()[14] = is_thumb ? (EXIT_TRAP_THUMB | 1) : EXIT_TRAP_ARM;
    jit->Regs()[15] = target_pc & ~1u;
    jit->SetCpsr((jit->Cpsr() & ~0x20u) | (is_thumb ? 0x20u : 0u));

    t_thread.active = jit;
    ++t_thread.depth;
    jit->ClearHalt();  // same drop window as in RunFunction
    if (GuestSignalPending()) GuestSignalDeliverPending(*this);
    for (;;) {
        ApplyPendingInvalidations(*jit);
        BridgeSetPhase("nested-Run");
        Dynarmic::HaltReason hr = jit->Run();
        BridgeSetPhase("nested-halted");
        // Every reason Run came back for, not just the first one. Restarting after
        // the first match starved the others: Mono flushes the instruction cache
        // after every method it compiles, so CacheInvalidation arrives continuously
        // during start-up, and a thread whose next Run always came back for that
        // never got as far as its pending signal. It made no progress either - each
        // Run returned immediately - so it looked like guest code stuck in a loop.
        bool resume = HandleHalt(*jit, hr);
        if (!resume) break;
    }
    --t_thread.depth;
    t_thread.active = outer;

    return jit->Regs()[0];
}

uint32_t DynarmicHost::GuestStackTop() {
    CurrentJit();  // makes sure this thread has a stack assigned
    return t_thread.stack_top;
}

uint32_t DynarmicHost::GuestStackLimit() {
    CurrentJit();
    return t_thread.stack_top - kThreadStackSize;
}

// Deep enough for Dynarmic's IR passes on the largest block Unity has produced here;
// the default 1 MiB was not, and the overflow lands in the optimiser rather than
// anywhere that names the guest.
static constexpr size_t kHostThreadStack = 8u * 1024 * 1024;

void SpawnHostThread(std::function<void()> body) {
    auto* payload = new std::function<void()>(std::move(body));
    auto entry = [](void* p) -> void* {
        std::unique_ptr<std::function<void()>> fn(static_cast<std::function<void()>*>(p));
        (*fn)();
        return nullptr;
    };

    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, kHostThreadStack);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
    pthread_t tid{};
    int rc = pthread_create(&tid, &attr, entry, payload);
    pthread_attr_destroy(&attr);
    if (rc != 0) {
        LOGE("pthread_create for a guest thread failed (%d); falling back to the default stack", rc);
        std::thread([payload, entry] { entry(payload); }).detach();
    }
}

void DynarmicHost::SpawnGuestThread(uint32_t start_func, uint32_t arg) {
    SpawnHostThread([this, start_func, arg] {
        RunFunction(start_func, {arg});
    });
}

uint32_t DynarmicHost::RunFunction(uint32_t target_pc, const std::vector<uint32_t>& args) {
    if (t_thread.running) {
        // Jit::Run is not re-entrant on the same Jit (a32_interface.cpp asserts on it),
        // but that is a property of the Jit, not of the thread - so run it on a nested
        // one, which is what CallGuestNested exists for.
        //
        // This used to push the call onto a deferred queue instead, and nothing ever
        // drained that queue. Java calls nativeResize from inside a thread that is
        // already running guest code, so the engine was never told the surface size:
        // it kept the 128x128 default and drew every frame into the bottom-left corner
        // of the screen.
        return CallGuestNested(target_pc, args);
    }

    Dynarmic::A32::Jit& jit = CurrentJit();

    t_thread.running = true;
    struct JitRunningGuard {
        ~JitRunningGuard() { t_thread.running = false; }
    } running_guard;

    for (size_t i = 0; i < args.size() && i < 4; ++i) {
        jit.Regs()[i] = args[i];
    }

    // AAPCS wants SP 8-byte aligned at a public interface; an odd number of stacked
    // arguments would otherwise leave a doubleword argument straddling that boundary.
    uint32_t sp = t_thread.stack_top;
    if (args.size() > 4) {
        size_t extra_args = args.size() - 4;
        sp = (sp - static_cast<uint32_t>(extra_args * 4)) & ~7u;
        for (size_t i = 0; i < extra_args; ++i) {
            *reinterpret_cast<uint32_t*>(m_memory + sp + i * 4) = args[4 + i];
        }
    }
    jit.Regs()[13] = sp;

    // Return address -> exit trap (with Thumb bit if target is Thumb)
    bool is_thumb = (target_pc & 1) != 0;
    jit.Regs()[14] = is_thumb ? (EXIT_TRAP_THUMB | 1) : EXIT_TRAP_ARM;

    // Set PC
    jit.Regs()[15] = target_pc & ~1u;
    jit.SetCpsr((jit.Cpsr() & ~0x20u) | (is_thumb ? 0x20u : 0u));

    jit.ClearHalt();
    // ClearHalt wipes a UserDefined2 that GuestSignalSend set moments ago, so a signal
    // aimed at a thread that was still being set up was dropped: the deque kept the
    // entry and nothing ever looked at it again. The thread is visible in the signal
    // table from CurrentJit() above, which is before this point. Boehm's stop-the-world
    // then waited forever for an acknowledgement from a thread it had already reached.
    if (GuestSignalPending()) GuestSignalDeliverPending(*this);
    Dynarmic::HaltReason hr{};
    for (;;) {
        ApplyPendingInvalidations(jit);
        BridgeSetPhase("Run");
        hr = jit.Run();
        BridgeSetPhase("halted");
        // InvalidateCacheRange() called from inside Run - a thunk minting a new
        // trampoline (dlsym), or the guest's own cacheflush after Mono emits code -
        // stops execution instead of resuming it. The stale block is already gone by
        // the time Run returns, so re-entering continues from the same PC. Signals
        // and PC dumps come back the same way; HandleHalt takes all of them at once.
        if (!HandleHalt(jit, hr)) break;
    }
    uint32_t final_pc = jit.Regs()[15];
    if (final_pc > kTrapPageEnd) {
        LOGI("RunFunction(0x%08X) unexpected halt with reason=0x%X, PC=0x%08X, LR=0x%08X, R0=0x%08X, SP=0x%08X",
             target_pc, static_cast<unsigned>(hr), final_pc, jit.Regs()[14], jit.Regs()[0], jit.Regs()[13]);
    }

    return jit.Regs()[0];
}

// diag: see InvalidateGuestCode. kClearEverything is the size that means "not a
// range - drop the lot"; no real flush can ask for it, the guest address space is
// 4 GiB and this is one byte short of it.
static constexpr uint32_t kClearEverything = 0xFFFFFFFFu;
// Answer, 27.08: no. The IMT misdispatch reproduces with this on, so the guest is
// not running code the flush should have killed. Left in place - it is the cheapest
// way to rule invalidation out, and it costs a run to rediscover.
static constexpr bool kFlushClearsEverything = false;
// diag: recognise the IMT thunks Mono writes, dump them, and patch a trace into
// every entry and every exit so a dispatch can be checked against the thunk's own
// keys. This is what named the qsort defect; it patches guest code, so it is off
// unless something is being hunted.
static constexpr bool kDumpSmallFlushes = false;

// diag: the IMT thunks seen so far, by start address, with the end Mono flushed.
static std::map<uint32_t, uint32_t> g_imt_thunks;
// diag: for every patched exit, the keys of its thunk and whether it is the entry
// taken without comparing anything - the default.
struct ImtExitSite { std::vector<uint32_t> keys; bool is_default; };
static std::map<uint32_t, ImtExitSite> g_imt_exits;
static std::mutex g_imt_mutex;

// diag: catch an interface call that is about to be dispatched to the wrong method.
//
// A Mono ARM IMT thunk takes the interface method out of the caller's code stream
// (`ldr r0, [lr, #-4]`, zero meaning "use r8"), compares it against the key of every
// entry but the last, and takes the last one without comparing - the runtime knows
// the value has to be one of them. So a value that matches no key is, by itself,
// proof of a misdispatch: the thunk is about to jump to whatever its last entry is.
// diag: MonoMethod on 32-bit is { u16 flags; u16 iflags; u32 token; MonoClass *klass;
// MonoMethodSignature *signature; const char *name; ... }, so the name is the sixth
// word. Printing it turns a list of pointers into a list of methods.
static std::string GuestMethodName(DynarmicHost& host, uint32_t method) {
    if (method < 0x40000000u) return "?";
    uint32_t name_ptr = host.MemoryRead32(method + 16);
    if (name_ptr < 0x40000000u) return "?";
    std::string out;
    for (int i = 0; i < 64; ++i) {
        char c = static_cast<char>(host.MemoryRead8(name_ptr + i));
        if (!c) break;
        out += (c >= 32 && c < 127) ? c : '.';
    }
    return out.empty() ? "?" : out;
}

// diag: the branch the thunk really takes. Every entry ends `pop {r0, r1}` followed
// by `ldr pc, [r12, #slot]`, so patching the pop reports the chosen slot from inside
// the guest's own execution - the static reading of the thunk cannot, and it is the
// static reading that is in question.
static void ImtTraceExit(Dynarmic::A32::Jit& jit, DynarmicHost& host) {
    uint32_t site = (jit.Regs()[15] & ~3u) - 4;
    uint32_t value = jit.Regs()[0];
    uint32_t next = host.MemoryRead32(site + 4);
    if ((next & 0xFFFFF000) != 0xE59CF000) next = host.MemoryRead32(site + 8);
    // This sits on a call made every frame: a line per dispatch drowns logd, and a
    // drowned logd stops the render thread inside writev - which reads as a hang.
    std::string name = GuestMethodName(host, value);

    // The check this trace exists for. A thunk compares the value against every key
    // but the last entry's, and jumps the moment one matches; reaching the last
    // entry therefore means no key matched. If the value *does* match a key and the
    // guest still arrives here, the comparison that should have caught it did not -
    // and that is the misdispatch, in the act.
    bool matched_key = false;
    bool is_default = false;
    {
        std::lock_guard<std::mutex> lock(g_imt_mutex);
        auto it = g_imt_exits.find(site);
        if (it != g_imt_exits.end()) {
            is_default = it->second.is_default;
            for (uint32_t key : it->second.keys) if (key == value) matched_key = true;
        }
    }
    if (is_default && matched_key) {
        static std::atomic<uint32_t> shouted{0};
        if (shouted.fetch_add(1) < 20) {
            LOGE("[imt] MISDISPATCH at 0x%08X: value 0x%08X '%s' matches a key of this "
                 "thunk, yet the default entry (slot +0x%X) was taken",
                 site, value, name.c_str(), next & 0xFFF);
        }
    }

    static std::atomic<uint32_t> told{0};
    if (told.fetch_add(1) < 40) {
        // r12 is the vtable the thunk loaded, so the slot it is about to jump
        // through can be read here - and, next to it, its neighbours. A slot that
        // ends up holding the same code as another slot is the corruption itself.
        uint32_t vt = jit.Regs()[12];
        uint32_t slot = next & 0xFFF;
        LOGE("[imt-exit] 0x%08X takes slot +0x%X with r0 0x%08X '%s' | vtable 0x%08X "
             "slot=0x%08X, neighbours -8:0x%08X -4:0x%08X +4:0x%08X",
             site, slot, value, name.c_str(), vt,
             host.MemoryRead32(vt + slot), host.MemoryRead32(vt + slot - 8),
             host.MemoryRead32(vt + slot - 4), host.MemoryRead32(vt + slot + 4));
    }

    uint32_t sp = jit.Regs()[13];
    jit.Regs()[0] = host.MemoryRead32(sp);
    jit.Regs()[1] = host.MemoryRead32(sp + 4);
    jit.Regs()[13] = sp + 8;
}

static void ImtTraceEntry(Dynarmic::A32::Jit& jit, DynarmicHost& host) {
    // R15 inside an SVC callback is the instruction after the one that trapped, so
    // the thunk starts one word back - checked against the table rather than assumed.
    uint32_t pc = jit.Regs()[15] & ~3u;
    uint32_t lr = jit.Regs()[14];
    uint32_t literal = host.MemoryRead32(lr - 4);
    uint32_t value = literal ? literal : jit.Regs()[8];

    uint32_t thunk = 0, end = 0;
    {
        std::lock_guard<std::mutex> lock(g_imt_mutex);
        for (uint32_t candidate : {pc - 4, pc, pc - 8}) {
            auto it = g_imt_thunks.find(candidate);
            if (it == g_imt_thunks.end()) continue;
            thunk = candidate;
            end = it->second;
            break;
        }
    }
    if (!thunk) {
        LOGE("[imt] trace fired at pc 0x%08X with no thunk recorded", pc);
        thunk = pc - 4;
    }

    // Walk the thunk the way it walks itself: every `ldr r1, [pc, #imm]` loads the
    // key of the entry that follows, and the entry after the last comparison is
    // taken unconditionally. Reporting which slot this call will end up in, with
    // names, is the whole point - a pointer tells nobody anything.
    std::string keys;
    uint32_t chosen_slot = 0;
    uint32_t vtable = 0;
    {
        uint32_t insn = host.MemoryRead32(thunk + 8);          // ldr r12, [pc, #imm]
        if ((insn & 0xFFFFF000) == 0xE59FC000)
            vtable = host.MemoryRead32(thunk + 8 + 8 + (insn & 0xFFF));
    }
    bool decided = false;
    uint32_t pending_key = 0;
    for (uint32_t a = thunk + 8; a + 4 <= end; a += 4) {
        uint32_t insn = host.MemoryRead32(a);
        if ((insn & 0xFFFFF000) == 0xE59F1000) {          // ldr r1, [pc, #imm]
            pending_key = host.MemoryRead32(a + 8 + (insn & 0xFFF));
            keys += GuestMethodName(host, pending_key) + " ";
            continue;
        }
        if ((insn & 0xFFFFF000) == 0xE59CF000 && !decided) {  // ldr pc, [r12, #imm]
            // The entry this belongs to is taken when its key matched, and the one
            // with no key before it is the default.
            if (pending_key == 0 || pending_key == value) {
                chosen_slot = insn & 0xFFF;
                decided = true;
            }
            pending_key = 0;
        }
    }

    // Where that slot actually leads. A vtable entry that has not been compiled yet
    // holds Mono's specific trampoline - `push {r0-r12,lr}; bl generic; .word method`
    // - and that third word names the method this call is really about to enter.
    // Asking for one method and arriving at another is the defect itself.
    std::string want = GuestMethodName(host, value);
    std::string got = "?";
    uint32_t target = 0;
    if (vtable && chosen_slot) {
        target = host.MemoryRead32(vtable + chosen_slot);
        if (target && host.MemoryRead32(target) == 0xE92D5FFF)
            got = GuestMethodName(host, host.MemoryRead32(target + 8));
    }
    bool agrees = got == "?" || want == "?" ||
                  want.find(got) != std::string::npos || got.find(want) != std::string::npos;

    // The call this whole hunt is about is IUIObject::get_gameObject, so that one is
    // always reported; everything else only when the names disagree.
    bool interesting = want.find("gameObject") != std::string::npos ||
                       keys.find("gameObject") != std::string::npos;
    static std::atomic<uint32_t> told{0};
    static std::atomic<uint32_t> shown{0};
    if (!agrees || interesting ? (told.fetch_add(1) < 40) : (shown.fetch_add(1) < 6)) {
        LOGE("[imt]%s thunk 0x%08X: want '%s' (0x%08X, literal 0x%08X, r8 0x%08X) "
             "keys [%s] -> slot +0x%X of vtable 0x%08X -> 0x%08X '%s' [%08X %08X %08X] "
             "| caller %08X %08X %08X",
             agrees ? "" : " MISDISPATCH", thunk, want.c_str(), value, literal, jit.Regs()[8],
             keys.c_str(), chosen_slot, vtable, target, got.c_str(),
             target ? host.MemoryRead32(target) : 0, target ? host.MemoryRead32(target + 4) : 0,
             target ? host.MemoryRead32(target + 8) : 0,
             host.MemoryRead32(lr - 12), host.MemoryRead32(lr - 8), host.MemoryRead32(lr - 4));
    }

    // The overwritten instruction, executed here: push {r0, r1}.
    uint32_t sp = jit.Regs()[13] - 8;
    jit.Regs()[13] = sp;
    host.MemoryWrite32(sp, jit.Regs()[0]);
    host.MemoryWrite32(sp + 4, jit.Regs()[1]);
}

void DynarmicHost::InstallImtTrace(uint32_t start, uint32_t end) {
    static uint32_t svc_word = 0;
    if (!svc_word) {
        m_thunk_manager->RegisterThunk("dxtf_imt_trace", ImtTraceEntry);
        uint32_t tramp = m_thunk_manager->GetOrCreateThunkAddress("dxtf_imt_trace");
        svc_word = MemoryRead32(tramp);  // svc #id, the trampoline's own first word
    }
    static uint32_t exit_word = 0;
    if (!exit_word) {
        m_thunk_manager->RegisterThunk("dxtf_imt_exit", ImtTraceExit);
        exit_word = MemoryRead32(m_thunk_manager->GetOrCreateThunkAddress("dxtf_imt_exit"));
    }
    {
        std::lock_guard<std::mutex> lock(g_imt_mutex);
        g_imt_thunks[start] = end;
    }
    MemoryWrite32(start, svc_word);
    InvalidateGuestCode(start, 4);

    std::vector<uint32_t> keys;
    for (uint32_t a = start + 8; a + 4 <= end; a += 4) {
        uint32_t insn = MemoryRead32(a);
        if ((insn & 0xFFFFF000) != 0xE59F1000) continue;   // ldr r1, [pc, #imm]
        keys.push_back(MemoryRead32(a + 8 + (insn & 0xFFF)));
    }
    std::vector<uint32_t> sites;
    for (uint32_t a = start; a + 8 <= end; a += 4) {
        if (MemoryRead32(a) != 0xE8BD0003) continue;       // pop {r0, r1}
        uint32_t next = MemoryRead32(a + 4);
        bool jumps = (next & 0xFFFFF000) == 0xE59CF000 ||
                     (next == 0xE28DD008 && (MemoryRead32(a + 8) & 0xFFFFF000) == 0xE59CF000);
        if (jumps) sites.push_back(a);
    }
    {
        std::lock_guard<std::mutex> lock(g_imt_mutex);
        for (size_t i = 0; i < sites.size(); ++i)
            g_imt_exits[sites[i]] = ImtExitSite{keys, i + 1 == sites.size()};
    }
    for (uint32_t site : sites) {
        MemoryWrite32(site, exit_word);
        InvalidateGuestCode(site, 4);
    }
}

void DynarmicHost::InvalidateGuestCode(uint32_t addr, size_t size) {
    // diag: answers one question and nothing else - "does the guest ever execute a
    // translation the flush was supposed to kill?". With this on, a flush of any
    // range throws away *every* translation on every Jit, so a stale block cannot
    // survive; anything that still misbehaves is not an invalidation problem.
    // Ruinously slow (Mono flushes after every compiled method), diagnostic only.
    if (kFlushClearsEverything) size = kClearEverything;
    // Every thread holds its own translation of the range, so all of them have to
    // drop it - otherwise a thread that already ran the old code keeps running it.
    //
    // But InvalidateCacheRange rewrites the Jit's own block tables, and doing that
    // to a Jit another thread is currently executing corrupts them. Mono flushes
    // the instruction cache after every method it compiles, so once the managed
    // code got going this was happening constantly, and the process died in
    // Dynarmic's own emitter - a different place each run, which is what a
    // corrupted structure looks like.
    //
    // So each Jit is invalidated by the thread that owns it. Other threads are
    // handed the range and stopped; HaltExecution is an atomic flag and is safe
    // to set from here. They apply it before they next execute anything.
    std::vector<Dynarmic::A32::Jit*> mine = OwnedJits();
    BridgeSetPhase("invalidate-own");
    for (auto* jit : mine) {
        if (size == kClearEverything) jit->ClearCache();
        else jit->InvalidateCacheRange(addr, size);
    }

    BridgeSetPhase("invalidate-lock");
    std::lock_guard<std::mutex> lock(m_jits_mutex);
    BridgeSetPhase("invalidate");
    for (auto& jit : m_jits) {
        Dynarmic::A32::Jit* raw = jit.get();
        if (std::find(mine.begin(), mine.end(), raw) != mine.end()) continue;
        m_pending_invalidations[raw].emplace_back(addr, static_cast<uint32_t>(size));
        raw->HaltExecution(Dynarmic::HaltReason::CacheInvalidation);
    }
}

// The Jits this thread owns: its own, plus any it is using for nested calls.
std::vector<Dynarmic::A32::Jit*> DynarmicHost::OwnedJits() {
    std::vector<Dynarmic::A32::Jit*> out;
    if (t_thread.jit) out.push_back(t_thread.jit);
    for (auto* jit : t_thread.nested) out.push_back(jit);
    return out;
}

// Applied by the owning thread, immediately before it enters Run.
void DynarmicHost::ApplyPendingInvalidations(Dynarmic::A32::Jit& jit) {
    std::vector<std::pair<uint32_t, uint32_t>> ranges;
    {
        BridgeSetPhase("apply-inval-lock");
        std::lock_guard<std::mutex> lock(m_jits_mutex);
        BridgeSetPhase("apply-inval");
        auto it = m_pending_invalidations.find(&jit);
        if (it == m_pending_invalidations.end() || it->second.empty()) return;
        ranges.swap(it->second);
    }
    for (auto& [addr, size] : ranges) {
        // The sentinel a full flush leaves behind: drop every translation this Jit
        // holds rather than one range of it.
        if (size == kClearEverything) { jit.ClearCache(); continue; }
        jit.InvalidateCacheRange(addr, size);
    }
}


// The guest heap runs from 0x50000000 up to the guest stacks at 0x7E000000.
// A bump allocator was enough to reach Mono's start-up and no further: Mono
// allocates and frees continuously once it is running, so with free() a no-op
// the 736 MiB arena drains in seconds and every allocation after that returns
// null.
//
// Blocks carry an eight-byte header - their own size and a magic word - and go
// back onto a free list bucketed by that size when released. Sizes are rounded
// so buckets actually get hits: powers of two up to 64 KiB, multiples of 64 KiB
// above it, which keeps the waste on Unity's large buffers bounded at 64 KiB
// instead of doubling them.
//
// The magic word is what makes free() safe. The guest also frees pointers that
// never came from here - static buffers, addresses inside a loaded image - and
// reading a "size" out of those would poison the free list.
namespace {
constexpr uint32_t kHeapBase = 0x50000000;
constexpr uint32_t kHeapLimit = 0x7E000000;
constexpr uint32_t kBlockMagic = 0x4246;  // 'BF', in the top half of the tag word
constexpr uint32_t kHeaderSize = 8;

// The tag word carries the magic and how strictly the block was aligned, so
// free() can put it back on the list it came from without the caller saying.
uint32_t MakeTag(size_t alignment) {
    uint32_t log2 = 3;
    while ((static_cast<size_t>(1) << log2) < alignment) ++log2;
    return (kBlockMagic << 16) | log2;
}
uint64_t BlockKey(uint32_t block, uint32_t tag) {
    return (static_cast<uint64_t>(tag & 0xFFFFu) << 32) | block;
}

uint32_t RoundBlockSize(size_t size) {
    uint32_t n = static_cast<uint32_t>(size < 16 ? 16 : size);
    if (n <= 64 * 1024) {
        uint32_t p = 16;
        while (p < n) p <<= 1;
        return p;
    }
    return (n + 0xFFFFu) & ~0xFFFFu;
}
}  // namespace

void DynarmicHost::ZeroGuestMemory(uint32_t addr, size_t size) {
    if (!m_memory || !size || static_cast<size_t>(addr) >= m_memory_size) return;

    const size_t bounded = std::min(size, m_memory_size - static_cast<size_t>(addr));
#if defined(_WIN32)
    memset(m_memory + addr, 0, bounded);
#else
    // MADV_DONTNEED on this private anonymous arena has exactly the zero-fill
    // semantics the 32-bit guest expects from a new mmap, but it leaves the pages
    // uncommitted until they are actually used.  Touching a multi-gigabyte
    // MAP_FIXED range with memset instead made Android account every page in RSS
    // and Xiaomi's lmkd killed the foreground game during cold start.
    static const size_t page_size = [] {
        long value = sysconf(_SC_PAGESIZE);
        return value > 0 ? static_cast<size_t>(value) : static_cast<size_t>(4096);
    }();
    const uint64_t begin = addr;
    const uint64_t end = begin + bounded;
    const uint64_t page_begin = ((begin + page_size - 1) / page_size) * page_size;
    const uint64_t page_end = (end / page_size) * page_size;

    if (page_begin >= page_end) {
        memset(m_memory + begin, 0, bounded);
        return;
    }

    if (page_begin > begin)
        memset(m_memory + begin, 0, static_cast<size_t>(page_begin - begin));
    const size_t whole_pages = static_cast<size_t>(page_end - page_begin);
    if (madvise(m_memory + page_begin, whole_pages, MADV_DONTNEED) != 0)
        memset(m_memory + page_begin, 0, whole_pages);
    if (end > page_end)
        memset(m_memory + page_end, 0, static_cast<size_t>(end - page_end));
#endif
}

uint32_t DynarmicHost::GuestAlloc(size_t size, size_t alignment) {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    if (alignment < 8) alignment = 8;
    if (size > static_cast<size_t>(kHeapLimit - kHeapBase - kHeaderSize)) {
        LOGE("GuestAlloc request is larger than the guest heap: %zu bytes", size);
        return 0;
    }
    uint32_t block = RoundBlockSize(size);

    // A recycled block keeps the alignment it was first handed out with, so the
    // free lists are keyed by both. Ignoring alignment here would either hand a
    // 4 KiB-aligned caller an 8-aligned block, or - as it did - refuse to recycle
    // page-aligned blocks at all, which is most of what Mono's JIT asks for.
    uint32_t tag = MakeTag(alignment);
    uint64_t key = BlockKey(block, tag);
    {
        auto it = m_free_blocks.find(key);
        if (it != m_free_blocks.end() && !it->second.empty()) {
            uint32_t addr = it->second.back();
            it->second.pop_back();
            // Fresh arena pages read as zero, so recycled ones have to as well:
            // callers that fill a structure only partly would otherwise see the
            // previous tenant's bytes.
            ZeroGuestMemory(addr, block);
            *reinterpret_cast<uint32_t*>(m_memory + addr - 8) = block;
            *reinterpret_cast<uint32_t*>(m_memory + addr - 4) = tag;
            return addr;
        }
    }

    uint32_t payload = (m_heap_current + kHeaderSize + static_cast<uint32_t>(alignment - 1)) &
                       ~static_cast<uint32_t>(alignment - 1);
    if (static_cast<uint64_t>(payload) + block > kHeapLimit) {
        LOGE("GuestAlloc out of guest heap: at 0x%08X, want %zu, limit 0x%08X", payload, size, kHeapLimit);
        return 0;
    }
    m_heap_current = payload + block;
    *reinterpret_cast<uint32_t*>(m_memory + payload - 8) = block;
    *reinterpret_cast<uint32_t*>(m_memory + payload - 4) = tag;
    return payload;
}

uint32_t DynarmicHost::GuestAllocSize(uint32_t addr) {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    if (addr < kHeapBase + kHeaderSize || addr >= kHeapLimit) return 0;
    uint32_t tag = *reinterpret_cast<uint32_t*>(m_memory + addr - 4);
    if ((tag >> 16) != kBlockMagic) return 0;
    return *reinterpret_cast<uint32_t*>(m_memory + addr - 8);
}

void DynarmicHost::GuestFree(uint32_t addr) {
    if (!addr) return;
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    uint32_t block = GuestAllocSize(addr);
    if (!block) return;  // not ours, or freed twice
    // Any guest lock that lived in this block dies with it.
    GuestSyncForget(addr, block);
    uint32_t tag = *reinterpret_cast<uint32_t*>(m_memory + addr - 4);
    // Clearing the magic is what makes a double free a no-op instead of putting
    // the same address on the list twice.
    *reinterpret_cast<uint32_t*>(m_memory + addr - 4) = 0;
    m_free_blocks[BlockKey(block, tag)].push_back(addr);
}

std::optional<std::uint32_t> DynarmicHost::MemoryReadCode(std::uint32_t vaddr) {
    if (static_cast<uint64_t>(vaddr) + 4 <= m_memory_size) {
        return *reinterpret_cast<uint32_t*>(m_memory + vaddr);
    }
    return std::nullopt;
}

std::uint8_t DynarmicHost::MemoryRead8(std::uint32_t vaddr) {
    if (static_cast<uint64_t>(vaddr) < m_memory_size) return m_memory[vaddr];
    return 0;
}

std::uint16_t DynarmicHost::MemoryRead16(std::uint32_t vaddr) {
    if (static_cast<uint64_t>(vaddr) + 2 <= m_memory_size) return *reinterpret_cast<uint16_t*>(m_memory + vaddr);
    return 0;
}

std::uint32_t DynarmicHost::MemoryRead32(std::uint32_t vaddr) {
    if (static_cast<uint64_t>(vaddr) + 4 <= m_memory_size) return *reinterpret_cast<uint32_t*>(m_memory + vaddr);
    return 0;
}

std::uint64_t DynarmicHost::MemoryRead64(std::uint32_t vaddr) {
    if (static_cast<uint64_t>(vaddr) + 8 <= m_memory_size) return *reinterpret_cast<uint64_t*>(m_memory + vaddr);
    return 0;
}

void DynarmicHost::MemoryWrite8(std::uint32_t vaddr, std::uint8_t value) {
    if (static_cast<uint64_t>(vaddr) < m_memory_size) m_memory[vaddr] = value;
}

void DynarmicHost::MemoryWrite16(std::uint32_t vaddr, std::uint16_t value) {
    if (static_cast<uint64_t>(vaddr) + 2 <= m_memory_size) *reinterpret_cast<uint16_t*>(m_memory + vaddr) = value;
}

void DynarmicHost::MemoryWrite32(std::uint32_t vaddr, std::uint32_t value) {
    if (static_cast<uint64_t>(vaddr) + 4 <= m_memory_size) *reinterpret_cast<uint32_t*>(m_memory + vaddr) = value;
}

void DynarmicHost::MemoryWrite64(std::uint32_t vaddr, std::uint64_t value) {
    if (static_cast<uint64_t>(vaddr) + 8 <= m_memory_size) *reinterpret_cast<uint64_t*>(m_memory + vaddr) = value;
}

void DynarmicHost::InterpreterFallback(std::uint32_t pc, std::size_t num_instructions) {
    LOGE("Interpreter Fallback at 0x%08X", pc);
    CurrentJit().HaltExecution(Dynarmic::HaltReason::UserDefined1);
}

// The ARM trap word is `svc #0` only when it is executed as ARM; as Thumb it is
// two harmless halfwords and the guest walks straight off the end of it. So the
// mode has to be set along with the PC, not left as whatever the guest was in.
void GuestJumpToExitTrap(Dynarmic::A32::Jit& jit) {
    jit.SetCpsr(jit.Cpsr() & ~0x20u);
    jit.Regs()[15] = EXIT_TRAP_ARM;
}

// ARM private syscalls, issued as "svc #0" with the number in r7.
static constexpr uint32_t kArmNrCacheflush = 0x0f0002;
static constexpr uint32_t kArmNrSetTls     = 0x0f0005;

void DynarmicHost::CallSVC(std::uint32_t swi) {
    Dynarmic::A32::Jit& jit = CurrentJit();

    if (swi == kNullCallSvc) {
        LOGE("guest called through a null function pointer, return address 0x%08X",
             jit.Regs()[14]);
        GuestJumpToExitTrap(jit);
        return;
    }

    // Any SVC executed inside the trap page is a guest function returning, whatever
    // immediate the word happened to decode to. The two traps hold different words
    // and each is only an SVC 0 in one instruction set: read the other way round,
    // the Thumb trap is `svcle #0xDF00` and the ARM trap is not an SVC at all. A
    // guest that returns in the mode the trap was not written for therefore used to
    // be dispatched as thunk 57088 and then carried on past the trap - into the dead
    // page, where nothing could stop it again. Deciding by PC covers every
    // combination without caring which of them arrived.
    if (jit.Regs()[15] <= kTrapPageEnd) {
        // diag: how often the guest comes back in the other instruction set at all.
        // Once per call would drown the log - the traps are hit as often as guest
        // functions return.
        if (swi != 0) {
            static std::atomic<uint32_t> mismatched{0};
            uint32_t n = mismatched.fetch_add(1);
            if (n < 4 || (n % 256) == 0) {
                LOGI("[trap] guest returned to the exit trap in the other instruction set "
                     "(svc %u, PC 0x%08X, lr 0x%08X, #%u)", swi, jit.Regs()[15], jit.Regs()[14], n);
            }
        }
        jit.HaltExecution(Dynarmic::HaltReason::UserDefined1);
        return;
    }

    // swi != 0 is a bridge trampoline.
    if (swi != 0) {
        m_thunk_manager->DispatchThunk(swi, jit);
        return;
    }

    // A real "svc #0" from guest code. Mono emits code at runtime and then asks the
    // kernel to flush the icache for it; without honouring that, the JIT keeps
    // executing the translation of whatever used to be at those addresses, which
    // shows up much later as random crashes once the managed JIT warms up.
    const uint32_t nr = jit.Regs()[7];
    switch (nr) {
        case kArmNrCacheflush: {
            uint32_t start = jit.Regs()[0];
            uint32_t end = jit.Regs()[1];
            // diag: an IMT thunk starts `push {r0,r1}` / `push {r0,r1,pc}` followed
            // by `ldr r0, [lr, #-4]` - it takes the interface method out of the
            // caller's code stream. Nothing else in this runtime opens that way, so
            // the shape is a better filter than the size.
            if (kDumpSmallFlushes && end > start && end - start <= 4096) {
                uint32_t w0 = MemoryRead32(start);
                uint32_t w1 = MemoryRead32(start + 4);
                bool imt = (w0 == 0xE92D0003 || w0 == 0xE92D9003) && w1 == 0xE51E0004;
                static std::atomic<uint32_t> seen{0};
                uint32_t n = imt ? seen.fetch_add(1) : 0xFFFFFFFF;
                if (imt && n < 400) {
                    InstallImtTrace(start, end);
                    uint32_t shown = std::min(end, start + 160);
                    std::string words;
                    for (uint32_t a = start; a < shown; a += 4) {
                        char buf[16];
                        snprintf(buf, sizeof(buf), "%08X ", MemoryRead32(a));
                        words += buf;
                    }
                    LOGE("[flush] #%u 0x%08X..0x%08X: %s", n, start, end, words.c_str());
                }
            }
            if (end > start) InvalidateGuestCode(start, end - start);
            jit.Regs()[0] = 0;
            return;
        }
        case kArmNrSetTls:
            // Guest code does read this back, through `mrc p15, 0, Rt, c13, c0, 3`,
            // so the syscall and the coprocessor have to share one value.
            GuestCp15SetCurrentTls(jit.Regs()[0]);
            jit.Regs()[0] = 0;
            return;
        default:
            LOGE("unhandled guest syscall %u (0x%X) at PC 0x%08X", nr, nr, jit.Regs()[15]);
            jit.Regs()[0] = static_cast<uint32_t>(-38);  // -ENOSYS
            return;
    }
}

void DynarmicHost::ExceptionRaised(std::uint32_t pc, Dynarmic::A32::Exception exception) {
    LOGE("Exception at PC: 0x%08X (type %d)", pc, static_cast<int>(exception));
    CurrentJit().HaltExecution(Dynarmic::HaltReason::UserDefined1);
}

void DynarmicHost::AddTicks(std::uint64_t ticks) {
    if (m_ticks_remaining > ticks) m_ticks_remaining -= ticks;
    else m_ticks_remaining = 0;
}

std::uint64_t DynarmicHost::GetTicksRemaining() {
    return m_ticks_remaining;
}

namespace {

// diag: strex is only used by lock primitives in these images - Boehm's
// GC_test_and_set, Unity's atomic flags - so remembering the last thread to win one
// per address names the owner of a lock that is never released. A lock word stuck at
// 1 with an owner that is gone reads, from the outside, exactly like a lock word
// stuck at 1 with an owner that is merely slow.
// Boehm's allocator lock in this libmono. Watched by address because its release is
// a plain store, which fastmem never routes through a callback.
//
// This whole block is off by default now that it has done its job: it costs a read
// and a ring-buffer write on every exclusive access to the word, plus a stack walk on
// every acquisition, and 0x423B5D34 is only that lock in this exact build of libmono.
// Set kWatchWord to 1 to bring it back for the next lock that will not come free.
constexpr bool kWatchWord = false;
constexpr uint32_t kWatchedWord = 0x423B5D34;
// Whoever currently holds it, or 0 while it is free. Set on the acquiring strex,
// cleared by the poller when the plain store releases it.
std::atomic<int> g_word_owner{0};
// Bumped on every acquisition, so a trace of what the owner did can be reset per hold
// rather than per owner - the same thread takes and releases it thousands of times.
std::atomic<uint32_t> g_word_epoch{0};
// The acquiring frame: only that frame ever stores the zero that releases the lock,
// so an owner whose stack pointer later sits above this one is not holding the lock
// on purpose - it left the function that took it without reaching its unlock.
std::atomic<uint32_t> g_word_owner_sp{0};
std::atomic<uint32_t> g_word_owner_lr{0};
// ...and how it got there. The unlock belongs to whichever function took it, so
// naming that function is the whole question.
constexpr int kOwnerFrames = 10;
std::atomic<uint32_t> g_word_owner_stack[kOwnerFrames];
std::atomic<int> g_word_owner_depth{0};

// diag: every exclusive op on the watched word, failures included. GC_test_and_set
// stores 1 unconditionally and returns what its ldrex read, so the guest's view of
// the lock is entirely decided here - and "the owner is the thread waiting for it"
// is only explicable if that view disagrees with memory.
struct StrexRecord {
    int tid;
    uint32_t expected;
    uint32_t value;
    uint32_t in_memory;
    uint32_t lr;
    bool ok;
};
StrexRecord g_strex_log[64];
std::atomic<uint32_t> g_strex_count{0};
// The spin writes 1 over 1 hundreds of times a second and would push the interesting
// entries out of a plain ring, so anything that is not that is kept separately.
StrexRecord g_strex_notable[64];
std::atomic<uint32_t> g_strex_notable_count{0};

struct ExclusiveWitness {
    std::atomic<uint32_t> addr{0};
    std::atomic<uint32_t> value{0};
    std::atomic<int32_t> tid{0};
    std::atomic<uint32_t> count{0};
};
constexpr size_t kExclusiveWitnesses = 1024;
ExclusiveWitness g_excl[kExclusiveWitnesses];

int WitnessTidFor(uint32_t vaddr) {
    uint32_t h = vaddr * 2654435761u;
    for (uint32_t probe = 0; probe < 8; ++probe) {
        ExclusiveWitness& w = g_excl[(h + probe) % kExclusiveWitnesses];
        if (w.addr.load(std::memory_order_relaxed) == vaddr) return w.tid.load();
    }
    return 0;
}

void NoteExclusiveWrite(uint32_t vaddr, uint32_t value, uint32_t was) {
    // Every guest atomic goes through here, so it stays off with the rest of the
    // watched-word tooling rather than probing a hash table per lock operation.
    if (!kWatchWord) return;
    uint32_t h = vaddr * 2654435761u;
    for (uint32_t probe = 0; probe < 8; ++probe) {
        ExclusiveWitness& w = g_excl[(h + probe) % kExclusiveWitnesses];
        uint32_t cur = w.addr.load(std::memory_order_relaxed);
        if (cur == 0) {
            uint32_t expected = 0;
            if (!w.addr.compare_exchange_strong(expected, vaddr)) continue;
        } else if (cur != vaddr) {
            continue;
        }
        // Only a 0 -> non-zero transition is an acquisition. GC_test_and_set stores 1
        // whether or not it won, so recording every writer would name a thread that is
        // still waiting for the lock as its owner.
        if (was == 0 && value != 0) {
            w.value.store(value, std::memory_order_relaxed);
            w.tid.store(gettid(), std::memory_order_relaxed);
            if (vaddr == kWatchedWord) {
                g_word_owner.store(gettid());
                g_word_epoch.fetch_add(1);
            }
        }
        w.count.fetch_add(1, std::memory_order_relaxed);
        return;
    }
}

}  // namespace

namespace {

// diag: Boehm's allocator lock ends every run held by a thread that is itself waiting
// for it, which is impossible if the only writers are GC_test_and_set (strex, visible
// above) and GC_clear (a plain store, invisible under fastmem). Polling the word
// records the release side too, so the last transition before everything stopped is
// on the record rather than inferred.
struct WordTransition {
    int64_t at_ms;
    uint32_t from;
    uint32_t to;
    int witness_tid;
};
WordTransition g_transitions[64];
std::atomic<uint32_t> g_transition_count{0};

int64_t WatchNowMs() {
    struct timespec ts {};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<int64_t>(ts.tv_sec) * 1000 + ts.tv_nsec / 1000000;
}

}  // namespace

void DynarmicHost::StartWordWatch() {
    if (!kWatchWord || !m_memory) return;
    std::thread([this] {
        volatile uint32_t* word = reinterpret_cast<volatile uint32_t*>(m_memory + kWatchedWord);
        uint32_t previous = *word;
        for (;;) {
            uint32_t now = *word;
            if (now != previous) {
                uint32_t slot = g_transition_count.fetch_add(1) % 64;
                g_transitions[slot] = {WatchNowMs(), previous, now, WitnessTidFor(kWatchedWord)};
                if (now == 0) g_word_owner.store(0);
                previous = now;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(200));
        }
    }).detach();
}

int DynarmicHost::WatchedWordOwner() { return g_word_owner.load(std::memory_order_relaxed); }

uint32_t DynarmicHost::WatchedWordEpoch() { return g_word_epoch.load(std::memory_order_relaxed); }

uint32_t DynarmicHost::WatchedWordOwnerSp() { return g_word_owner_sp.load(std::memory_order_relaxed); }

uint32_t DynarmicHost::WatchedWordOwnerLr() { return g_word_owner_lr.load(std::memory_order_relaxed); }

void DynarmicHost::DumpStrexLog() {
    uint32_t total = g_strex_count.load();
    uint32_t shown = total < 16 ? total : 16;
    LOGE("[strex] last %u of %u exclusive writes to 0x%08X:", shown, total, kWatchedWord);
    for (uint32_t i = total - shown; i < total; ++i) {
        const StrexRecord& r = g_strex_log[i % 64];
        LOGE("[strex]   #%u tid %d lr 0x%08X saw %u, memory held %u, wrote %u -> %s", i, r.tid,
             r.lr, r.expected, r.in_memory, r.value, r.ok ? "ok" : "FAILED");
    }
    uint32_t notable = g_strex_notable_count.load();
    uint32_t notable_shown = notable < 16 ? notable : 16;
    LOGE("[strex] last %u of %u that were not the spin:", notable_shown, notable);
    for (uint32_t i = notable - notable_shown; i < notable; ++i) {
        const StrexRecord& r = g_strex_notable[i % 64];
        LOGE("[strex]   ! #%u tid %d lr 0x%08X saw %u, memory held %u, wrote %u -> %s", i, r.tid,
             r.lr, r.expected, r.in_memory, r.value, r.ok ? "ok" : "FAILED");
    }
}

void DynarmicHost::DumpWordWatch() {
    uint32_t total = g_transition_count.load();
    LOGE("[word] 0x%08X = %u after %u transitions, now %lld ms", kWatchedWord,
         MemoryRead32(kWatchedWord), total, static_cast<long long>(WatchNowMs()));
    uint32_t shown = total < 8 ? total : 8;
    for (uint32_t i = total - shown; i < total; ++i) {
        const WordTransition& t = g_transitions[i % 64];
        LOGE("[word]   #%u at %lld ms: %u -> %u (strex owner tid %d)", i,
             static_cast<long long>(t.at_ms), t.from, t.to, t.witness_tid);
    }
}

void DynarmicHost::DumpExclusiveWitnesses() {
    std::vector<std::pair<uint32_t, size_t>> ranked;
    for (size_t i = 0; i < kExclusiveWitnesses; ++i) {
        if (g_excl[i].addr.load()) ranked.emplace_back(g_excl[i].count.load(), i);
    }
    std::sort(ranked.rbegin(), ranked.rend());
    for (size_t i = 0; i < ranked.size() && i < 20; ++i) {
        ExclusiveWitness& w = g_excl[ranked[i].second];
        uint32_t addr = w.addr.load();
        LOGE("[excl] 0x%08X now=%u taken by tid %d (x%u)", addr, MemoryRead32(addr),
             w.tid.load(), ranked[i].first);
    }
}

// STREX: compare-and-set against guest memory.
//
// This has to be one atomic operation, not a compare followed by a store. The
// ExclusiveMonitor's lock only serialises exclusive accesses against each other,
// and the other half of a guest lock is a plain store that fastmem writes straight
// into the arena without passing through here at all. Boehm releases its allocator
// lock exactly that way - GC_clear is `dmb sy; str #0, [r3]`.
//
// So with a read-compare-write, a spinning GC_test_and_set (which stores 1
// unconditionally and reports what it read) could compare against 1, have the
// owner's release land in the gap, and then put its own 1 back on top of the zero.
// The lock came back held with nobody holding it, and every thread queued on it
// spun until the process was killed. It took a few thousand strex/release pairs a
// second to hit, which is why roughly one launch in eight got through to the menu.
template<typename T>
static bool GuestCompareAndSwap(uint8_t* mem, size_t mem_size, uint32_t vaddr, T value, T expected) {
    if (static_cast<uint64_t>(vaddr) + sizeof(T) > mem_size) return false;
    // LDREX/STREX fault on an unaligned address on real hardware, and so would the
    // host's own atomics; there is nothing to be correct about in that case.
    if ((vaddr & (sizeof(T) - 1)) != 0) return false;
    auto* p = reinterpret_cast<std::atomic<T>*>(mem + vaddr);
    return p->compare_exchange_strong(expected, value, std::memory_order_seq_cst);
}

bool DynarmicHost::MemoryWriteExclusive8(std::uint32_t vaddr, std::uint8_t value, std::uint8_t expected) {
    return GuestCompareAndSwap(m_memory, m_memory_size, vaddr, value, expected);
}

bool DynarmicHost::MemoryWriteExclusive16(std::uint32_t vaddr, std::uint16_t value, std::uint16_t expected) {
    return GuestCompareAndSwap(m_memory, m_memory_size, vaddr, value, expected);
}

bool DynarmicHost::MemoryWriteExclusive32(std::uint32_t vaddr, std::uint32_t value, std::uint32_t expected) {
    const bool watched = kWatchWord && vaddr == kWatchedWord;
    uint32_t before = watched ? MemoryRead32(vaddr) : 0;
    bool ok = GuestCompareAndSwap(m_memory, m_memory_size, vaddr, value, expected);
    if (watched) {
        StrexRecord record{gettid(), expected, value, before, CurrentJit().Regs()[14], ok};
        g_strex_log[g_strex_count.fetch_add(1) % 64] = record;
        if (!ok || expected == 0 || before == 0 || value == 0) {
            g_strex_notable[g_strex_notable_count.fetch_add(1) % 64] = record;
        }
    }
    if (ok) NoteExclusiveWrite(vaddr, value, expected);
    if (ok && watched && before == 0 && value != 0) {
        Dynarmic::A32::Jit& jit = CurrentJit();
        uint32_t sp = jit.Regs()[13];
        uint32_t lr = jit.Regs()[14];
        g_word_owner_sp.store(sp);
        g_word_owner_lr.store(lr);
        g_word_owner_depth.store(t_thread.depth);
        uint32_t frames[kOwnerFrames];
        GuestBacktrace(*this, sp, lr, frames, kOwnerFrames);
        for (int i = 0; i < kOwnerFrames; ++i) g_word_owner_stack[i].store(frames[i]);
    }
    return ok;
}

void DynarmicHost::DumpWatchedWordAcquireStack() {
    std::string line;
    for (int i = 0; i < kOwnerFrames; ++i) {
        uint32_t f = g_word_owner_stack[i].load();
        if (!f) break;
        char buf[16];
        snprintf(buf, sizeof(buf), "%08X ", f);
        line += buf;
    }
    LOGE("[owner] taken at nesting depth %d, guest stack: %s", g_word_owner_depth.load(),
         line.c_str());
}

bool DynarmicHost::MemoryWriteExclusive64(std::uint32_t vaddr, std::uint64_t value, std::uint64_t expected) {
    return GuestCompareAndSwap(m_memory, m_memory_size, vaddr, value, expected);
}
