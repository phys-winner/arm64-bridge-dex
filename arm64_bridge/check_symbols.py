# SPDX-License-Identifier: GPL-3.0-or-later
import struct, re, os, sys, glob

def get_symbols(path):
    with open(path, 'rb') as f:
        elf = f.read()
    e_shoff, = struct.unpack_from('<I', elf, 32)
    e_shentsize, = struct.unpack_from('<H', elf, 46)
    e_shnum, = struct.unpack_from('<H', elf, 48)
    sections = []
    for i in range(e_shnum):
        hdr = elf[e_shoff + i * e_shentsize : e_shoff + (i + 1) * e_shentsize]
        name_idx, s_type, s_flags, s_addr, s_offset, s_size, s_link, s_info, s_addralign, s_entsize = struct.unpack_from('<IIIIIIIIII', hdr, 0)
        sections.append({'type': s_type, 'addr': s_addr, 'offset': s_offset, 'size': s_size, 'link': s_link})
    dynsym = [s for s in sections if s['type'] == 11][0]
    dynstr = sections[dynsym['link']]
    imp, exp = set(), set()
    for i in range(0, dynsym['size'], 16):
        st_name, st_value, st_size, st_info, st_other, st_shndx = struct.unpack_from('<IIIBBH', elf, dynsym['offset'] + i)
        name = elf[dynstr['offset'] + st_name:].split(b'\0')[0].decode('latin1')
        if not name: continue
        if st_shndx == 0: imp.add(name)
        else: exp.add(name)
    return exp, imp

# usage: python check_symbols.py <guest.so> [loader_src_dir]
if len(sys.argv) not in (2, 3):
    raise SystemExit('Usage: python check_symbols.py <guest.so> [loader_src_dir]')
guest_so = sys.argv[1]
src_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(__file__), 'src')
name = os.path.basename(guest_so)

guest_exp, guest_imp = get_symbols(guest_so)
print(f'Total imports in {name}: {len(guest_imp)}')

# Read registered thunks from every .cpp in the loader source dir
registered = set()
for fpath in glob.glob(os.path.join(src_dir, '*.cpp')):
    with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
        src = f.read()
    registered.update(re.findall(r'RegisterThunk\s*\(\s*"([^"]+)"', src))

print(f'Total registered thunks in {src_dir}: {len(registered)}')

unhandled = guest_imp - registered
print(f'Unhandled imports in {name}: {len(unhandled)}')
for s in sorted(unhandled):
    print('  [MISSING]', s)

print(f'--- FMOD imports in {name} ---')
for s in sorted(guest_imp):
    if 'FMOD' in s or 'Event' in s:
        print('  [FMOD]', s)
