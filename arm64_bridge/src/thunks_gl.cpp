// SPDX-License-Identifier: GPL-3.0-or-later
#include "thunk_manager.h"
#include "dynarmic_host.h"
#include "thunk_types.h"
#include "bridge_config.h"
#include <GLES2/gl2.h>
#include <android/log.h>
#include <vector>
#include <cstring>
#include <mutex>
#include <unordered_map>


// Mirror of the current buffer bindings. Querying the driver with glGetIntegerv inside
// glDrawElements/glVertexAttribPointer cost a round-trip on every draw call.
static GLuint s_bound_array_buffer = 0;
static GLuint s_bound_element_buffer = 0;
static GLuint s_bound_framebuffer = 0;

// With a custom UI scale the guest believes the screen is GetVirtual{Width,Height}() and sizes
// its viewport accordingly, but the real default framebuffer is still the surface size. Offscreen
// targets are allocated in the guest's own units, so only the window-system framebuffer is mapped.
static void ScaleViewportToSurface(GLint& x, GLint& y, GLsizei& w, GLsizei& h) {
    if (s_bound_framebuffer != 0) return;
    const uint32_t vw = GetVirtualWidth(), vh = GetVirtualHeight();
    const uint32_t rw = ThunkManager::s_screenWidth, rh = ThunkManager::s_screenHeight;
    if (!vw || !vh || !rw || !rh || (vw == rw && vh == rh)) return;
    x = static_cast<GLint>((float)x * rw / vw);
    w = static_cast<GLsizei>((float)w * rw / vw);
    y = static_cast<GLint>((float)y * rh / vh);
    h = static_cast<GLsizei>((float)h * rh / vh);
}
#define LOG_TAG "GOF2_GLThunks"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

static uint32_t g_draw_calls = 0;

void ThunkManager::RegisterGLThunks() {
    uint8_t* mem = m_host.GetGuestMemoryBase();

    RegisterThunk("glActiveTexture", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glActiveTexture(static_cast<GLenum>(jit.Regs()[0]));
    });

    RegisterThunk("glAttachShader", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glAttachShader(jit.Regs()[0], jit.Regs()[1]);
    });

    RegisterThunk("glBindBuffer", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const GLenum target = static_cast<GLenum>(jit.Regs()[0]);
        const GLuint buffer = jit.Regs()[1];
        if (target == GL_ARRAY_BUFFER) s_bound_array_buffer = buffer;
        else if (target == GL_ELEMENT_ARRAY_BUFFER) s_bound_element_buffer = buffer;
        glBindBuffer(target, buffer);
    });

    RegisterThunk("glBindFramebuffer", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        s_bound_framebuffer = jit.Regs()[1];
        glBindFramebuffer(static_cast<GLenum>(jit.Regs()[0]), jit.Regs()[1]);
    });

    RegisterThunk("glBindRenderbuffer", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glBindRenderbuffer(static_cast<GLenum>(jit.Regs()[0]), jit.Regs()[1]);
    });

    RegisterThunk("glBindTexture", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glBindTexture(static_cast<GLenum>(jit.Regs()[0]), jit.Regs()[1]);
    });

    RegisterThunk("glBlendFunc", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glBlendFunc(static_cast<GLenum>(jit.Regs()[0]), static_cast<GLenum>(jit.Regs()[1]));
    });

    RegisterThunk("glBufferData", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        GLenum target = static_cast<GLenum>(jit.Regs()[0]);
        GLsizeiptr size = static_cast<GLsizeiptr>(jit.Regs()[1]);
        const void* data = GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[2]);
        GLenum usage = static_cast<GLenum>(jit.Regs()[3]);
        glBufferData(target, size, data, usage);
    });

    RegisterThunk("glCheckFramebufferStatus", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        jit.Regs()[0] = glCheckFramebufferStatus(static_cast<GLenum>(jit.Regs()[0]));
    });

    RegisterThunk("glClear", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glClear(static_cast<GLbitfield>(jit.Regs()[0]));
    });

    RegisterThunk("glClearColor", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        float r = *reinterpret_cast<float*>(&jit.Regs()[0]);
        float g = *reinterpret_cast<float*>(&jit.Regs()[1]);
        float b = *reinterpret_cast<float*>(&jit.Regs()[2]);
        float a = *reinterpret_cast<float*>(&jit.Regs()[3]);
        glClearColor(r, g, b, a);
    });

    RegisterThunk("glColorMask", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glColorMask(jit.Regs()[0], jit.Regs()[1], jit.Regs()[2], jit.Regs()[3]);
    });

    RegisterThunk("glCompileShader", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        GLuint shader = jit.Regs()[0];
        glCompileShader(shader);
        GLint compiled = 0;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
        if (!compiled) {
            char log[1024];
            glGetShaderInfoLog(shader, sizeof(log), nullptr, log);
            LOGE("Shader %u compile failed: %s", shader, log);
        }
    });

    RegisterThunk("glCompressedTexImage2D", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        GLenum target = jit.Regs()[0];
        GLint level = jit.Regs()[1];
        GLenum internalformat = jit.Regs()[2];
        GLsizei width = jit.Regs()[3];
        uint32_t sp = jit.Regs()[13];
        GLsizei height = host.MemoryRead32(sp);
        GLint border = host.MemoryRead32(sp + 4);
        GLsizei imageSize = host.MemoryRead32(sp + 8);
        const void* data = GuestToHost(host.GetGuestMemoryBase(), host.MemoryRead32(sp + 12));
        glCompressedTexImage2D(target, level, internalformat, width, height, border, imageSize, data);
    });

    RegisterThunk("glCreateProgram", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        jit.Regs()[0] = glCreateProgram();
    });

    RegisterThunk("glCreateShader", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        jit.Regs()[0] = glCreateShader(static_cast<GLenum>(jit.Regs()[0]));
    });

    RegisterThunk("glCullFace", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glCullFace(static_cast<GLenum>(jit.Regs()[0]));
    });

    RegisterThunk("glDeleteBuffers", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glDeleteBuffers(jit.Regs()[0], reinterpret_cast<const GLuint*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[1])));
    });

    RegisterThunk("glDeleteFramebuffers", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glDeleteFramebuffers(jit.Regs()[0], reinterpret_cast<const GLuint*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[1])));
    });

    RegisterThunk("glDeleteProgram", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glDeleteProgram(jit.Regs()[0]);
    });

    RegisterThunk("glDeleteRenderbuffers", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glDeleteRenderbuffers(jit.Regs()[0], reinterpret_cast<const GLuint*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[1])));
    });

    RegisterThunk("glDeleteShader", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glDeleteShader(jit.Regs()[0]);
    });

    RegisterThunk("glDeleteTextures", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glDeleteTextures(jit.Regs()[0], reinterpret_cast<const GLuint*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[1])));
    });

    RegisterThunk("glDepthFunc", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glDepthFunc(static_cast<GLenum>(jit.Regs()[0]));
    });

    RegisterThunk("glDepthMask", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glDepthMask(static_cast<GLboolean>(jit.Regs()[0]));
    });

    RegisterThunk("glDisable", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glDisable(static_cast<GLenum>(jit.Regs()[0]));
    });

    RegisterThunk("glDisableVertexAttribArray", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glDisableVertexAttribArray(jit.Regs()[0]);
    });

    RegisterThunk("glDrawArrays", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glDrawArrays(static_cast<GLenum>(jit.Regs()[0]), static_cast<GLint>(jit.Regs()[1]), static_cast<GLsizei>(jit.Regs()[2]));
    });

    RegisterThunk("glDrawElements", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        uint32_t indices_gaddr = jit.Regs()[3];
        const void* indices = (s_bound_element_buffer != 0) ?
            reinterpret_cast<const void*>(static_cast<uintptr_t>(indices_gaddr)) :
            GuestToHost(host.GetGuestMemoryBase(), indices_gaddr);
        glDrawElements(static_cast<GLenum>(jit.Regs()[0]), static_cast<GLsizei>(jit.Regs()[1]),
                       static_cast<GLenum>(jit.Regs()[2]), indices);
    });

    RegisterThunk("glEnable", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glEnable(static_cast<GLenum>(jit.Regs()[0]));
    });

    RegisterThunk("glEnableVertexAttribArray", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glEnableVertexAttribArray(jit.Regs()[0]);
    });

    RegisterThunk("glFlush", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glFlush();
    });

    RegisterThunk("glFramebufferRenderbuffer", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glFramebufferRenderbuffer(jit.Regs()[0], jit.Regs()[1], jit.Regs()[2], jit.Regs()[3]);
    });

    RegisterThunk("glFramebufferTexture2D", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glFramebufferTexture2D(jit.Regs()[0], jit.Regs()[1], jit.Regs()[2], jit.Regs()[3], host.MemoryRead32(jit.Regs()[13]));
    });

    RegisterThunk("glFrontFace", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glFrontFace(static_cast<GLenum>(jit.Regs()[0]));
    });

    RegisterThunk("glGenBuffers", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glGenBuffers(jit.Regs()[0], reinterpret_cast<GLuint*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[1])));
    });

    RegisterThunk("glGenFramebuffers", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glGenFramebuffers(jit.Regs()[0], reinterpret_cast<GLuint*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[1])));
    });

    RegisterThunk("glGenRenderbuffers", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glGenRenderbuffers(jit.Regs()[0], reinterpret_cast<GLuint*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[1])));
    });

    RegisterThunk("glGenTextures", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glGenTextures(jit.Regs()[0], reinterpret_cast<GLuint*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[1])));
    });

    RegisterThunk("glGenerateMipmap", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glGenerateMipmap(static_cast<GLenum>(jit.Regs()[0]));
    });

    RegisterThunk("glGetActiveAttrib", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        uint32_t sp = jit.Regs()[13];
        GLsizei* length = reinterpret_cast<GLsizei*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[3]));
        GLint* size = reinterpret_cast<GLint*>(GuestToHost(host.GetGuestMemoryBase(), host.MemoryRead32(sp)));
        GLenum* type = reinterpret_cast<GLenum*>(GuestToHost(host.GetGuestMemoryBase(), host.MemoryRead32(sp + 4)));
        GLchar* name = reinterpret_cast<GLchar*>(GuestToHost(host.GetGuestMemoryBase(), host.MemoryRead32(sp + 8)));
        glGetActiveAttrib(jit.Regs()[0], jit.Regs()[1], jit.Regs()[2], length, size, type, name);
    });

    RegisterThunk("glGetActiveUniform", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        uint32_t sp = jit.Regs()[13];
        GLsizei* length = reinterpret_cast<GLsizei*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[3]));
        GLint* size = reinterpret_cast<GLint*>(GuestToHost(host.GetGuestMemoryBase(), host.MemoryRead32(sp)));
        GLenum* type = reinterpret_cast<GLenum*>(GuestToHost(host.GetGuestMemoryBase(), host.MemoryRead32(sp + 4)));
        GLchar* name = reinterpret_cast<GLchar*>(GuestToHost(host.GetGuestMemoryBase(), host.MemoryRead32(sp + 8)));
        glGetActiveUniform(jit.Regs()[0], jit.Regs()[1], jit.Regs()[2], length, size, type, name);
    });

    RegisterThunk("glGetAttribLocation", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        jit.Regs()[0] = glGetAttribLocation(jit.Regs()[0], reinterpret_cast<const GLchar*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[1])));
    });

    RegisterThunk("glGetError", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        jit.Regs()[0] = glGetError();
    });

    RegisterThunk("glGetIntegerv", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glGetIntegerv(static_cast<GLenum>(jit.Regs()[0]),
                      reinterpret_cast<GLint*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[1])));
    });

    RegisterThunk("glGetProgramInfoLog", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        GLsizei* length = jit.Regs()[2] ?
            reinterpret_cast<GLsizei*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[2])) : nullptr;
        glGetProgramInfoLog(jit.Regs()[0], jit.Regs()[1], length,
                           reinterpret_cast<GLchar*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[3])));
    });

    RegisterThunk("glGetProgramiv", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glGetProgramiv(jit.Regs()[0], static_cast<GLenum>(jit.Regs()[1]),
                       reinterpret_cast<GLint*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[2])));
    });

    RegisterThunk("glGetRenderbufferParameteriv", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glGetRenderbufferParameteriv(jit.Regs()[0], jit.Regs()[1],
                                     reinterpret_cast<GLint*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[2])));
    });

    RegisterThunk("glGetShaderInfoLog", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        GLsizei* length = jit.Regs()[2] ?
            reinterpret_cast<GLsizei*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[2])) : nullptr;
        glGetShaderInfoLog(jit.Regs()[0], jit.Regs()[1], length,
                          reinterpret_cast<GLchar*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[3])));
    });

    RegisterThunk("glGetShaderiv", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glGetShaderiv(jit.Regs()[0], static_cast<GLenum>(jit.Regs()[1]),
                      reinterpret_cast<GLint*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[2])));
    });

    // GL promises a pointer that stays valid for the life of the context, and
    // callers keep it. One block per enum, so a driver-capability check inside a
    // frame loop costs nothing after the first frame - GL_EXTENSIONS alone is a
    // couple of KiB.
    RegisterThunk("glGetString", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        static std::mutex m;
        static std::unordered_map<GLenum, uint32_t> cache;
        GLenum name = static_cast<GLenum>(jit.Regs()[0]);
        std::lock_guard<std::mutex> lock(m);
        auto it = cache.find(name);
        if (it == cache.end()) {
            const GLubyte* str = glGetString(name);
            uint32_t guest_addr = 0;
            if (str) {
                size_t len = strlen(reinterpret_cast<const char*>(str)) + 1;
                guest_addr = host.GuestAlloc(len);
                if (guest_addr) memcpy(host.GetGuestMemoryBase() + guest_addr, str, len);
            }
            it = cache.emplace(name, guest_addr).first;
        }
        jit.Regs()[0] = it->second;
    });

    RegisterThunk("glGetUniformLocation", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        jit.Regs()[0] = glGetUniformLocation(jit.Regs()[0], reinterpret_cast<const GLchar*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[1])));
    });

    RegisterThunk("glHint", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glHint(static_cast<GLenum>(jit.Regs()[0]), static_cast<GLenum>(jit.Regs()[1]));
    });

    RegisterThunk("glLineWidth", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glLineWidth(*reinterpret_cast<float*>(&jit.Regs()[0]));
    });

    RegisterThunk("glLinkProgram", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        GLuint prog = jit.Regs()[0];
        glLinkProgram(prog);
        GLint linked = 0;
        glGetProgramiv(prog, GL_LINK_STATUS, &linked);
        if (!linked) {
            char log[1024];
            glGetProgramInfoLog(prog, sizeof(log), nullptr, log);
            LOGE("Program %u link failed: %s", prog, log);
        }
    });

    RegisterThunk("glPixelStorei", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glPixelStorei(static_cast<GLenum>(jit.Regs()[0]), jit.Regs()[1]);
    });

    RegisterThunk("glPolygonOffset", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glPolygonOffset(*reinterpret_cast<float*>(&jit.Regs()[0]), *reinterpret_cast<float*>(&jit.Regs()[1]));
    });

    RegisterThunk("glReadPixels", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        uint32_t sp = jit.Regs()[13];
        GLenum format = host.MemoryRead32(sp);
        GLenum type = host.MemoryRead32(sp + 4);
        void* pixels = GuestToHost(host.GetGuestMemoryBase(), host.MemoryRead32(sp + 8));
        glReadPixels(jit.Regs()[0], jit.Regs()[1], jit.Regs()[2], jit.Regs()[3], format, type, pixels);
    });

    RegisterThunk("glRenderbufferStorage", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glRenderbufferStorage(jit.Regs()[0], jit.Regs()[1], jit.Regs()[2], jit.Regs()[3]);
    });

    RegisterThunk("glScissor", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        GLint x = jit.Regs()[0], y = jit.Regs()[1];
        GLsizei w = jit.Regs()[2], h = jit.Regs()[3];
        ScaleViewportToSurface(x, y, w, h);
        glScissor(x, y, w, h);
    });

    RegisterThunk("glShaderSource", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        GLuint shader = jit.Regs()[0];
        GLsizei count = jit.Regs()[1];
        uint32_t str_array_gaddr = jit.Regs()[2];
        uint32_t len_array_gaddr = jit.Regs()[3];

        std::vector<const char*> strings(count);
        for (GLsizei i = 0; i < count; ++i) {
            uint32_t str_gaddr = host.MemoryRead32(str_array_gaddr + i * 4);
            strings[i] = reinterpret_cast<const char*>(GuestToHost(host.GetGuestMemoryBase(), str_gaddr));
        }

        const GLint* lengths = len_array_gaddr ?
            reinterpret_cast<const GLint*>(GuestToHost(host.GetGuestMemoryBase(), len_array_gaddr)) : nullptr;

        glShaderSource(shader, count, strings.data(), lengths);
    });

    RegisterThunk("glTexImage2D", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        GLenum target = jit.Regs()[0];
        GLint level = jit.Regs()[1];
        GLint internalformat = jit.Regs()[2];
        GLsizei width = jit.Regs()[3];
        uint32_t sp = jit.Regs()[13];
        GLsizei height = host.MemoryRead32(sp);
        GLint border = host.MemoryRead32(sp + 4);
        GLenum format = host.MemoryRead32(sp + 8);
        GLenum type = host.MemoryRead32(sp + 12);
        uint32_t pixels_gaddr = host.MemoryRead32(sp + 16);
        const void* pixels = (pixels_gaddr < 0x10000) ?
            reinterpret_cast<const void*>(static_cast<uintptr_t>(pixels_gaddr)) :
            GuestToHost(host.GetGuestMemoryBase(), pixels_gaddr);
        glTexImage2D(target, level, internalformat, width, height, border, format, type, pixels);
    });

    RegisterThunk("glTexParameterf", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glTexParameterf(jit.Regs()[0], jit.Regs()[1], *reinterpret_cast<float*>(&jit.Regs()[2]));
    });

    RegisterThunk("glTexParameteri", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glTexParameteri(jit.Regs()[0], jit.Regs()[1], jit.Regs()[2]);
    });

    RegisterThunk("glUniform1f", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glUniform1f(jit.Regs()[0], *reinterpret_cast<float*>(&jit.Regs()[1]));
    });

    RegisterThunk("glUniform1i", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glUniform1i(jit.Regs()[0], jit.Regs()[1]);
    });

    RegisterThunk("glUniform2f", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glUniform2f(jit.Regs()[0], *reinterpret_cast<float*>(&jit.Regs()[1]), *reinterpret_cast<float*>(&jit.Regs()[2]));
    });

    RegisterThunk("glUniform3f", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glUniform3f(jit.Regs()[0], *reinterpret_cast<float*>(&jit.Regs()[1]),
                    *reinterpret_cast<float*>(&jit.Regs()[2]), *reinterpret_cast<float*>(&jit.Regs()[3]));
    });

    RegisterThunk("glUniform3fv", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glUniform3fv(jit.Regs()[0], jit.Regs()[1],
                     reinterpret_cast<const GLfloat*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[2])));
    });

    RegisterThunk("glUniform4f", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        uint32_t sp = jit.Regs()[13];
        uint32_t w_bits = host.MemoryRead32(sp);
        float w = *reinterpret_cast<float*>(&w_bits);
        glUniform4f(jit.Regs()[0], *reinterpret_cast<float*>(&jit.Regs()[1]),
                    *reinterpret_cast<float*>(&jit.Regs()[2]), *reinterpret_cast<float*>(&jit.Regs()[3]), w);
    });

    RegisterThunk("glUniform4fv", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glUniform4fv(jit.Regs()[0], jit.Regs()[1],
                     reinterpret_cast<const GLfloat*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[2])));
    });

    RegisterThunk("glUniformMatrix3fv", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glUniformMatrix3fv(jit.Regs()[0], jit.Regs()[1], jit.Regs()[2],
                           reinterpret_cast<const GLfloat*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[3])));
    });

    RegisterThunk("glUniformMatrix4fv", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glUniformMatrix4fv(jit.Regs()[0], jit.Regs()[1], jit.Regs()[2],
                           reinterpret_cast<const GLfloat*>(GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[3])));
    });

    RegisterThunk("glUseProgram", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glUseProgram(jit.Regs()[0]);
    });

    RegisterThunk("glVertexAttribPointer", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        GLuint index = jit.Regs()[0];
        GLint size = jit.Regs()[1];
        GLenum type = jit.Regs()[2];
        GLboolean normalized = jit.Regs()[3];
        uint32_t sp = jit.Regs()[13];
        GLsizei stride = host.MemoryRead32(sp);
        uint32_t ptr_gaddr = host.MemoryRead32(sp + 4);

        const void* ptr = (s_bound_array_buffer != 0) ?
            reinterpret_cast<const void*>(static_cast<uintptr_t>(ptr_gaddr)) :
            GuestToHost(host.GetGuestMemoryBase(), ptr_gaddr);

        glVertexAttribPointer(index, size, type, normalized, stride, ptr);
    });

    RegisterThunk("glViewport", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        GLint x = jit.Regs()[0], y = jit.Regs()[1];
        GLsizei w = jit.Regs()[2], h = jit.Regs()[3];
        ScaleViewportToSurface(x, y, w, h);
        // diag: the size the engine believes it is drawing into.
        static uint32_t reports = 0;
        if (reports++ < 12) LOGI("[viewport] %d,%d %dx%d fbo=%u", x, y, w, h, s_bound_framebuffer);
        glViewport(x, y, w, h);
    });

    // The rest of the GLES2 entry points libunity imports. These were all
    // landing on the default stub, which returns zero and draws nothing - a
    // stencilled UI element simply never appears, with no error anywhere.

    RegisterThunk("glFinish", [](Dynarmic::A32::Jit&, DynarmicHost&) { glFinish(); });
    RegisterThunk("glClearStencil", [](Dynarmic::A32::Jit& jit, DynarmicHost&) {
        glClearStencil(static_cast<GLint>(jit.Regs()[0]));
    });
    // Softfp: the float arrives as its bit pattern in a core register.
    RegisterThunk("glClearDepthf", [](Dynarmic::A32::Jit& jit, DynarmicHost&) {
        glClearDepthf(*reinterpret_cast<float*>(&jit.Regs()[0]));
    });
    RegisterThunk("glStencilFunc", [](Dynarmic::A32::Jit& jit, DynarmicHost&) {
        glStencilFunc(static_cast<GLenum>(jit.Regs()[0]), static_cast<GLint>(jit.Regs()[1]),
                      jit.Regs()[2]);
    });
    RegisterThunk("glStencilFuncSeparate", [](Dynarmic::A32::Jit& jit, DynarmicHost&) {
        glStencilFuncSeparate(static_cast<GLenum>(jit.Regs()[0]), static_cast<GLenum>(jit.Regs()[1]),
                              static_cast<GLint>(jit.Regs()[2]), jit.Regs()[3]);
    });
    RegisterThunk("glStencilMask", [](Dynarmic::A32::Jit& jit, DynarmicHost&) {
        glStencilMask(jit.Regs()[0]);
    });
    RegisterThunk("glStencilOp", [](Dynarmic::A32::Jit& jit, DynarmicHost&) {
        glStencilOp(static_cast<GLenum>(jit.Regs()[0]), static_cast<GLenum>(jit.Regs()[1]),
                    static_cast<GLenum>(jit.Regs()[2]));
    });
    RegisterThunk("glStencilOpSeparate", [](Dynarmic::A32::Jit& jit, DynarmicHost&) {
        glStencilOpSeparate(static_cast<GLenum>(jit.Regs()[0]), static_cast<GLenum>(jit.Regs()[1]),
                            static_cast<GLenum>(jit.Regs()[2]), static_cast<GLenum>(jit.Regs()[3]));
    });
    RegisterThunk("glBlendEquationSeparate", [](Dynarmic::A32::Jit& jit, DynarmicHost&) {
        glBlendEquationSeparate(static_cast<GLenum>(jit.Regs()[0]), static_cast<GLenum>(jit.Regs()[1]));
    });
    RegisterThunk("glBlendFuncSeparate", [](Dynarmic::A32::Jit& jit, DynarmicHost&) {
        glBlendFuncSeparate(static_cast<GLenum>(jit.Regs()[0]), static_cast<GLenum>(jit.Regs()[1]),
                            static_cast<GLenum>(jit.Regs()[2]), static_cast<GLenum>(jit.Regs()[3]));
    });
    RegisterThunk("glBindAttribLocation", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        const char* name = reinterpret_cast<const char*>(
            GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[2]));
        glBindAttribLocation(jit.Regs()[0], jit.Regs()[1], name);
    });
    RegisterThunk("glBufferSubData", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        glBufferSubData(static_cast<GLenum>(jit.Regs()[0]), static_cast<GLintptr>(jit.Regs()[1]),
                        static_cast<GLsizeiptr>(jit.Regs()[2]),
                        GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[3]));
    });
    RegisterThunk("glTexSubImage2D", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        uint32_t sp = jit.Regs()[13];
        glTexSubImage2D(static_cast<GLenum>(jit.Regs()[0]), static_cast<GLint>(jit.Regs()[1]),
                        static_cast<GLint>(jit.Regs()[2]), static_cast<GLint>(jit.Regs()[3]),
                        static_cast<GLsizei>(host.MemoryRead32(sp)),
                        static_cast<GLsizei>(host.MemoryRead32(sp + 4)),
                        static_cast<GLenum>(host.MemoryRead32(sp + 8)),
                        static_cast<GLenum>(host.MemoryRead32(sp + 12)),
                        GuestToHost(host.GetGuestMemoryBase(), host.MemoryRead32(sp + 16)));
    });
    RegisterThunk("glCompressedTexSubImage2D", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        uint32_t sp = jit.Regs()[13];
        glCompressedTexSubImage2D(static_cast<GLenum>(jit.Regs()[0]), static_cast<GLint>(jit.Regs()[1]),
                                  static_cast<GLint>(jit.Regs()[2]), static_cast<GLint>(jit.Regs()[3]),
                                  static_cast<GLsizei>(host.MemoryRead32(sp)),
                                  static_cast<GLsizei>(host.MemoryRead32(sp + 4)),
                                  static_cast<GLenum>(host.MemoryRead32(sp + 8)),
                                  static_cast<GLsizei>(host.MemoryRead32(sp + 12)),
                                  GuestToHost(host.GetGuestMemoryBase(), host.MemoryRead32(sp + 16)));
    });
    RegisterThunk("glCopyTexSubImage2D", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        uint32_t sp = jit.Regs()[13];
        glCopyTexSubImage2D(static_cast<GLenum>(jit.Regs()[0]), static_cast<GLint>(jit.Regs()[1]),
                            static_cast<GLint>(jit.Regs()[2]), static_cast<GLint>(jit.Regs()[3]),
                            static_cast<GLint>(host.MemoryRead32(sp)),
                            static_cast<GLint>(host.MemoryRead32(sp + 4)),
                            static_cast<GLsizei>(host.MemoryRead32(sp + 8)),
                            static_cast<GLsizei>(host.MemoryRead32(sp + 12)));
    });
    RegisterThunk("glGetFramebufferAttachmentParameteriv", [](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
        GLint value = 0;
        glGetFramebufferAttachmentParameteriv(static_cast<GLenum>(jit.Regs()[0]),
                                              static_cast<GLenum>(jit.Regs()[1]),
                                              static_cast<GLenum>(jit.Regs()[2]), &value);
        if (jit.Regs()[3]) host.MemoryWrite32(jit.Regs()[3], static_cast<uint32_t>(value));
    });

    // glUniform*v(location, count, value): `value` points at count*N words in
    // guest memory, which the arena makes directly addressable.
    auto uniform_v = [this](const char* name, void (*fn)(GLint, GLsizei, const void*)) {
        RegisterThunk(name, [fn](Dynarmic::A32::Jit& jit, DynarmicHost& host) {
            fn(static_cast<GLint>(jit.Regs()[0]), static_cast<GLsizei>(jit.Regs()[1]),
               GuestToHost(host.GetGuestMemoryBase(), jit.Regs()[2]));
        });
    };
    uniform_v("glUniform1fv", [](GLint l, GLsizei c, const void* v) {
        glUniform1fv(l, c, static_cast<const GLfloat*>(v)); });
    uniform_v("glUniform2fv", [](GLint l, GLsizei c, const void* v) {
        glUniform2fv(l, c, static_cast<const GLfloat*>(v)); });
    uniform_v("glUniform1iv", [](GLint l, GLsizei c, const void* v) {
        glUniform1iv(l, c, static_cast<const GLint*>(v)); });
    uniform_v("glUniform2iv", [](GLint l, GLsizei c, const void* v) {
        glUniform2iv(l, c, static_cast<const GLint*>(v)); });
    uniform_v("glUniform3iv", [](GLint l, GLsizei c, const void* v) {
        glUniform3iv(l, c, static_cast<const GLint*>(v)); });
    uniform_v("glUniform4iv", [](GLint l, GLsizei c, const void* v) {
        glUniform4iv(l, c, static_cast<const GLint*>(v)); });
}
