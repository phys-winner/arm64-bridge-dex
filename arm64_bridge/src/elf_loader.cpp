// SPDX-License-Identifier: GPL-3.0-or-later
#include "elf_loader.h"
#include <fstream>
#include <cstring>
#include <algorithm>
#include <unistd.h>
#include <android/log.h>

#include "apk_reader.h"

#define LOG_TAG "GOF2_ElfLoader"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

#define R_ARM_NONE        0
#define R_ARM_ABS32       2
#define R_ARM_REL32       3
#define R_ARM_GLOB_DAT    21
#define R_ARM_JUMP_SLOT   22
#define R_ARM_RELATIVE    23

ElfLoader::ElfLoader(uint8_t* guest_memory_base, uint32_t load_base_addr)
    : m_guest_mem(guest_memory_base), m_load_base(load_base_addr) {
}

ElfLoader::~ElfLoader() = default;

// The path may be a real file or an entry inside the package, written as
// "<...>/base.apk/assets/guestlib/libunity.so". The guest images are the latter:
// nothing unpacks them, they are read out of the APK straight into this buffer
// and from here into the guest arena. They are stored rather than deflated (see
// apktool.yml), so that read is one contiguous pread.
bool ElfLoader::Load(const std::string& filepath) {
    std::vector<uint8_t> buffer;

    std::ifstream file(filepath, std::ios::binary | std::ios::ate);
    if (file.is_open()) {
        buffer.resize(static_cast<size_t>(file.tellg()));
        file.seekg(0, std::ios::beg);
        if (!file.read(reinterpret_cast<char*>(buffer.data()), buffer.size())) {
            LOGE("Failed to read ELF file data: %s", filepath.c_str());
            return false;
        }
    } else {
        off_t entry_size = 0;
        int fd = ApkOpenEntry(filepath.c_str(), &entry_size);
        if (fd < 0) {
            LOGE("Failed to open ELF file: %s", filepath.c_str());
            return false;
        }
        buffer.resize(static_cast<size_t>(entry_size));
        size_t got = 0;
        while (got < buffer.size()) {
            ssize_t n = read(fd, buffer.data() + got, buffer.size() - got);
            if (n <= 0) break;
            got += static_cast<size_t>(n);
        }
        close(fd);
        if (got != buffer.size()) {
            LOGE("Short read of %s (%zu of %zu bytes)", filepath.c_str(), got, buffer.size());
            return false;
        }
    }

    return LoadFromMemory(buffer.data(), buffer.size());
}

bool ElfLoader::LoadFromMemory(const uint8_t* data, size_t size) {
    if (!ParseHeader(data, size)) return false;
    if (!LoadSegments(data, size)) return false;
    if (!ParseDynamic(data, size)) return false;
    return true;
}

bool ElfLoader::ParseHeader(const uint8_t* data, size_t size) {
    if (size < sizeof(Elf32_Ehdr)) {
        LOGE("File too small for ELF header.");
        return false;
    }

    const Elf32_Ehdr* ehdr = reinterpret_cast<const Elf32_Ehdr*>(data);
    if (memcmp(ehdr->e_ident, ELFMAG, SELFMAG) != 0) {
        LOGE("Invalid ELF magic.");
        return false;
    }

    if (ehdr->e_ident[EI_CLASS] != ELFCLASS32) {
        LOGE("Not a 32-bit ELF.");
        return false;
    }

    if (ehdr->e_machine != EM_ARM) {
        LOGE("Not an ARM architecture binary.");
        return false;
    }

    return true;
}

bool ElfLoader::LoadSegments(const uint8_t* data, size_t size) {
    const Elf32_Ehdr* ehdr = reinterpret_cast<const Elf32_Ehdr*>(data);
    const Elf32_Phdr* phdrs = reinterpret_cast<const Elf32_Phdr*>(data + ehdr->e_phoff);

    uint32_t min_vaddr = 0xFFFFFFFF;
    uint32_t max_vaddr = 0;

    for (int i = 0; i < ehdr->e_phnum; ++i) {
        const Elf32_Phdr& phdr = phdrs[i];
        if (phdr.p_type == PT_LOAD) {
            if (phdr.p_vaddr < min_vaddr) min_vaddr = phdr.p_vaddr;
            uint32_t end_vaddr = phdr.p_vaddr + phdr.p_memsz;
            if (end_vaddr > max_vaddr) max_vaddr = end_vaddr;

            ElfSegment seg;
            seg.vaddr = phdr.p_vaddr;
            seg.memsz = phdr.p_memsz;
            seg.filesz = phdr.p_filesz;
            seg.flags = phdr.p_flags;
            seg.offset = phdr.p_offset;
            m_segments.push_back(seg);

            // Copy to guest memory
            uint8_t* dest = m_guest_mem + m_load_base + phdr.p_vaddr;
            if (phdr.p_filesz > 0 && phdr.p_offset + phdr.p_filesz <= size) {
                memcpy(dest, data + phdr.p_offset, phdr.p_filesz);
            }
            if (phdr.p_memsz > phdr.p_filesz) {
                memset(dest + phdr.p_filesz, 0, phdr.p_memsz - phdr.p_filesz);
            }
        }
    }

    m_image_size = (max_vaddr - min_vaddr + 0xFFF) & ~0xFFF;
    LOGI("Loaded %zu segments, image size: 0x%08X", m_segments.size(), m_image_size);
    return true;
}

bool ElfLoader::ParseDynamic(const uint8_t* data, size_t size) {
    const Elf32_Ehdr* ehdr = reinterpret_cast<const Elf32_Ehdr*>(data);
    const Elf32_Phdr* phdrs = reinterpret_cast<const Elf32_Phdr*>(data + ehdr->e_phoff);

    const Elf32_Dyn* dyn = nullptr;
    for (int i = 0; i < ehdr->e_phnum; ++i) {
        if (phdrs[i].p_type == PT_DYNAMIC) {
            dyn = reinterpret_cast<const Elf32_Dyn*>(m_guest_mem + m_load_base + phdrs[i].p_vaddr);
            break;
        }
    }

    if (!dyn) {
        LOGE("No PT_DYNAMIC found in ELF.");
        return false;
    }

    std::vector<uint32_t> needed_str_offsets;
    uint32_t init_array_vaddr = 0;
    uint32_t init_array_size = 0;

    for (const Elf32_Dyn* d = dyn; d->d_tag != DT_NULL; ++d) {
        switch (d->d_tag) {
            case DT_STRTAB:     m_strtab_vaddr = d->d_un.d_ptr; break;
            case DT_SYMTAB:     m_symtab_vaddr = d->d_un.d_ptr; break;
            case DT_REL:        m_rel_vaddr = d->d_un.d_ptr; break;
            case DT_RELSZ:      m_rel_size = d->d_un.d_val; break;
            case DT_JMPREL:     m_plt_rel_vaddr = d->d_un.d_ptr; break;
            case DT_PLTRELSZ:   m_plt_rel_size = d->d_un.d_val; break;
            case DT_PLTGOT:     m_got_vaddr = d->d_un.d_ptr; break;
            case DT_INIT:       m_init_func = d->d_un.d_ptr; break;
            case DT_INIT_ARRAY: init_array_vaddr = d->d_un.d_ptr; break;
            case DT_INIT_ARRAYSZ: init_array_size = d->d_un.d_val; break;
            case DT_NEEDED:     needed_str_offsets.push_back(d->d_un.d_val); break;
            default: break;
        }
    }

    const char* strtab = reinterpret_cast<const char*>(m_guest_mem + m_load_base + m_strtab_vaddr);

    for (uint32_t offset : needed_str_offsets) {
        m_needed_libs.emplace_back(strtab + offset);
    }

    // Parse symbols
    if (m_symtab_vaddr != 0 && m_strtab_vaddr != 0) {
        const Elf32_Sym* symtab = reinterpret_cast<const Elf32_Sym*>(m_guest_mem + m_load_base + m_symtab_vaddr);
        size_t sym_count = (m_strtab_vaddr - m_symtab_vaddr) / sizeof(Elf32_Sym);

        for (size_t i = 0; i < sym_count; ++i) {
            const Elf32_Sym& sym = symtab[i];
            const char* name = strtab + sym.st_name;
            if (strlen(name) == 0) continue;

            ElfSymbol elf_sym;
            elf_sym.name = name;
            elf_sym.value = (sym.st_shndx != SHN_UNDEF) ? (sym.st_value + m_load_base) : 0;
            elf_sym.size = sym.st_size;
            elf_sym.bind = ELF32_ST_BIND(sym.st_info);
            elf_sym.type = ELF32_ST_TYPE(sym.st_info);
            elf_sym.shndx = sym.st_shndx;

            if (sym.st_shndx == SHN_UNDEF) {
                m_undefined_symbols.push_back(name);
            } else {
                m_exported_symbols[name] = elf_sym;
            }
        }
    }

    // Parse init_array
    if (init_array_vaddr != 0 && init_array_size > 0) {
        size_t count = init_array_size / sizeof(uint32_t);
        const uint32_t* init_funcs = reinterpret_cast<const uint32_t*>(m_guest_mem + m_load_base + init_array_vaddr);
        for (size_t i = 0; i < count; ++i) {
            if (init_funcs[i] != 0) {
                m_init_array.push_back(init_funcs[i] + m_load_base);
            }
        }
    }

    LOGI("Parsed %zu exported symbols, %zu undefined symbols, %zu init funcs",
         m_exported_symbols.size(), m_undefined_symbols.size(), m_init_array.size());
    return true;
}

bool ElfLoader::Relocate(const SymbolResolver& resolver) {
    return ProcessRelocations(resolver);
}

bool ElfLoader::ProcessRelocations(const SymbolResolver& resolver) {
    const char* strtab = reinterpret_cast<const char*>(m_guest_mem + m_load_base + m_strtab_vaddr);
    const Elf32_Sym* symtab = reinterpret_cast<const Elf32_Sym*>(m_guest_mem + m_load_base + m_symtab_vaddr);

    auto apply_reloc = [&](const Elf32_Rel& rel) {
        uint32_t* target = reinterpret_cast<uint32_t*>(m_guest_mem + m_load_base + rel.r_offset);
        uint32_t sym_idx = ELF32_R_SYM(rel.r_info);
        uint32_t type = ELF32_R_TYPE(rel.r_info);

        uint32_t sym_val = 0;
        if (sym_idx != 0) {
            const Elf32_Sym& sym = symtab[sym_idx];
            const char* sym_name = strtab + sym.st_name;

            if (sym.st_shndx != SHN_UNDEF) {
                sym_val = sym.st_value + m_load_base;
            } else {
                sym_val = resolver(sym_name);
            }
        }

        switch (type) {
            case R_ARM_NONE:
                break;
            case R_ARM_RELATIVE:
                *target += m_load_base;
                break;
            case R_ARM_GLOB_DAT:
            case R_ARM_JUMP_SLOT:
                *target = sym_val;
                break;
            case R_ARM_ABS32:
                *target += sym_val;
                break;
            case R_ARM_REL32:
                *target += sym_val - (m_load_base + rel.r_offset);
                break;
            default:
                LOGE("Unsupported relocation type: %u", type);
                return false;
        }
        return true;
    };

    // Apply DT_REL
    if (m_rel_vaddr != 0 && m_rel_size > 0) {
        size_t count = m_rel_size / sizeof(Elf32_Rel);
        const Elf32_Rel* rels = reinterpret_cast<const Elf32_Rel*>(m_guest_mem + m_load_base + m_rel_vaddr);
        for (size_t i = 0; i < count; ++i) {
            apply_reloc(rels[i]);
        }
    }

    // Apply DT_JMPREL (PLT relocations)
    if (m_plt_rel_vaddr != 0 && m_plt_rel_size > 0) {
        size_t count = m_plt_rel_size / sizeof(Elf32_Rel);
        const Elf32_Rel* plt_rels = reinterpret_cast<const Elf32_Rel*>(m_guest_mem + m_load_base + m_plt_rel_vaddr);
        for (size_t i = 0; i < count; ++i) {
            apply_reloc(plt_rels[i]);
        }
    }

    LOGI("Relocations completed successfully!");
    return true;
}

uint32_t ElfLoader::GetSymbolAddress(const std::string& name) const {
    auto it = m_exported_symbols.find(name);
    if (it != m_exported_symbols.end()) {
        return it->second.value;
    }
    return 0;
}
