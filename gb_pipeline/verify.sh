#!/bin/bash
# Check that this folder reproduces the GB tables in pyagadir/params.
#   1. the stored intermediates match MANIFEST.sha256;
#   2. the cheap steps (under 2 minutes, run in parallel) re-derive every GB table from them, and
#      each table's data rows (all lines but the '#' comments) are compared with the package file.
# The expensive step (structures and their GB energies) is not re-run; step 1 checks its output. The gbx overrides are
# not derived by a script (README.md, "Parameter overrides"). gbp's tables were retired on 2026-10-22 (README.md).
#
# Usage: gb_pipeline/verify.sh            (from anywhere)
# Env:   BUILD_PY  python of the build environment (env/build.txt); default <repo>/.venv-build/bin/python
#        KEEP=1    keep the re-derived tables (their folder is printed)
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(dirname "$HERE")
BUILD_PY=${BUILD_PY:-$ROOT/.venv-build/bin/python}
OUT=$(mktemp -d)
export OMP_NUM_THREADS=1
fail=0
cd "$HERE"

echo "1. intermediates against MANIFEST.sha256"
if sha256sum --quiet -c MANIFEST.sha256; then echo "   OK"; else echo "   FAILED"; fail=1; fi

echo "2. re-deriving (logs in $OUT)"
run() {  # name, command...
    local name=$1; shift
    local t0=$SECONDS
    if "$@" > "$OUT/$name.log" 2>&1; then echo "   ran   $name ($((SECONDS - t0)) s)"; else echo "   ERROR $name (see $OUT/$name.log)"; fi
}
run dipole "$BUILD_PY" helix_builds/build_gb_dipole_table.py --derive data/gb_raw_v2.tsv.gz \
    --supplement data/gb_raw_v2_strict_supplement.tsv.gz --out "$OUT/sidechain_dipole_gb.tsv" &
run transfer sh -c "cd \"$ROOT\" && uv run python gb_pipeline/transfer/build_gb_transfer.py --out \"$OUT/sidechain_dipole_gb_transfer.tsv\"" &
wait

echo "3. data rows against pyagadir/params"
for t in sidechain_dipole_gb sidechain_dipole_gb_transfer; do
    new="$OUT/$t.tsv"; pkg="$ROOT/pyagadir/params/$t.tsv"
    if [ ! -s "$new" ]; then echo "   MISSING   $t"; fail=1
    elif cmp -s <(grep -v '^#' "$new") <(grep -v '^#' "$pkg"); then echo "   identical $t"
    else echo "   DIFFERENT $t ($(diff <(grep -v '^#' "$new") <(grep -v '^#' "$pkg") | grep -c '^<') rows)"; fail=1; fi
done
[ -n "${KEEP:-}" ] && echo "kept: $OUT" || rm -rf "$OUT"
[ $fail = 0 ] && echo "PASS" || echo "FAIL"
exit $fail
