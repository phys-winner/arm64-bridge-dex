#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Repackage an existing, prepared apktool project; see readme.md.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
if (( $# )); then
    echo "Usage: ./build.sh (configure tools through environment variables)" >&2
    exit 1
fi
: "${APKTOOL_JAR:?Set APKTOOL_JAR}"
: "${APKSIGNER_JAR:?Set APKSIGNER_JAR}"
: "${ZIPALIGN:?Set ZIPALIGN}"
: "${LLVM_STRIP:?Set LLVM_STRIP}"
: "${SIGNING_KEY:?Set SIGNING_KEY to your APK signing key (.pk8)}"
: "${SIGNING_CERT:?Set SIGNING_CERT to your signing certificate (.x509.pem)}"
APP_DIR="${APP_DIR:-$ROOT/app}"
BRIDGE_LIB="${BRIDGE_LIB:-$ROOT/arm64_bridge/build_arm64/libmain.so}"
OUT="$ROOT/out"
FINAL="$OUT/DeusEx_TheFall_arm64.apk"
for file in "$APKTOOL_JAR" "$APKSIGNER_JAR" "$ZIPALIGN" "$LLVM_STRIP" "$SIGNING_KEY" "$SIGNING_CERT" "$BRIDGE_LIB" "$APP_DIR/apktool.yml"; do
    [[ -f "$file" ]] || { echo "Required input missing: $file" >&2; exit 1; }
done
for guest in libmain.so libmono.so libunity.so libAkSoundEnginePlugin.so; do
    [[ -f "$APP_DIR/assets/guestlib/$guest" ]] || { echo "Missing guest library: $guest" >&2; exit 1; }
done
mkdir -p "$OUT" "$APP_DIR/lib/arm64-v8a"
"$LLVM_STRIP" -o "$APP_DIR/lib/arm64-v8a/libmain.so" "$BRIDGE_LIB"
java -jar "$APKTOOL_JAR" b "$APP_DIR" -o "$OUT/dxtf_staged.apk"
python - "$OUT/dxtf_staged.apk" <<'PY'
import sys, zipfile
with zipfile.ZipFile(sys.argv[1]) as apk:
    abis = {n.split('/')[1] for n in apk.namelist() if n.startswith('lib/') and not n.endswith('/')}
    if abis != {'arm64-v8a'}:
        raise SystemExit(f'Unexpected ABI directories: {sorted(abis)}')
    guests = [n for n in apk.namelist() if n.startswith('assets/guestlib/') and n.endswith('.so')]
    if any(apk.getinfo(n).compress_type != zipfile.ZIP_STORED for n in guests):
        raise SystemExit('Guest libraries must be stored uncompressed (apktool doNotCompress: so)')
PY
"$ZIPALIGN" -P 16 -f 4 "$OUT/dxtf_staged.apk" "$OUT/dxtf_aligned.apk"
"$ZIPALIGN" -c -P 16 4 "$OUT/dxtf_aligned.apk"
java -jar "$APKSIGNER_JAR" sign --key "$SIGNING_KEY" --cert "$SIGNING_CERT" \
    --min-sdk-version 26 --out "$FINAL" "$OUT/dxtf_aligned.apk"
java -jar "$APKSIGNER_JAR" verify "$FINAL"
rm -f "$OUT/dxtf_staged.apk" "$OUT/dxtf_aligned.apk"
printf '%s\n' "$FINAL"
