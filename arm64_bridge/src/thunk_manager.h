// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <functional>
#include <mutex>

namespace Dynarmic::A32 {
    class Jit;
}

class DynarmicHost;

using ThunkHandler = std::function<void(Dynarmic::A32::Jit& jit, DynarmicHost& host)>;

class ThunkManager {
public:
    ThunkManager(DynarmicHost& host);
    ~ThunkManager();

    void RegisterAllThunks();
    void RegisterThunk(const std::string& name, ThunkHandler handler);
    void RegisterSymbol(const std::string& name, uint32_t address);

    uint32_t GetOrCreateThunkAddress(const std::string& name);
    bool IsRegistered(const std::string& name);
    void DispatchThunk(uint32_t thunk_id, Dynarmic::A32::Jit& jit);

    const std::unordered_map<std::string, uint32_t>& GetThunkSymbolTable() const {
        return m_symbol_to_address;
    }

    void SetupGuestJNIEnv();
    uint32_t GetGuestJNIEnv() const { return m_guest_jni_env; }

    static uint32_t s_screenWidth;
    static uint32_t s_screenHeight;
    static uint32_t s_forced_dpi;
    static uint32_t s_ui_inset;

private:
    void RegisterGLThunks();
    void RegisterLibcThunks();
    void RegisterAndroidThunks();
    void RegisterExtraLibcThunks();
    void RegisterZlibThunks();
    void RegisterEglThunks();
    void RegisterOpenSLThunks();
    void InstallFastMemcpy();
    void RegisterDataSymbols();
    // dlsym mints trampolines on whichever guest thread asks, so the tables below are
    // mutated after start-up and from more than one thread.
    std::recursive_mutex m_mutex;
    DynarmicHost& m_host;
    uint32_t m_next_thunk_addr = 0x00010000; // Base address for thunk trampolines
    uint32_t m_guest_jni_env = 0;
    std::unordered_map<std::string, uint32_t> m_symbol_to_address;
    std::unordered_map<uint32_t, ThunkHandler> m_id_to_handler;
    std::unordered_map<uint32_t, std::string> m_id_to_name;
};


// Asset overriding from <externalFilesDir>/... (thunks_libc.cpp)
void SetAssetOverrideRoot(const std::string& root);
void InstallAssetOverrideHook(DynarmicHost& host);
