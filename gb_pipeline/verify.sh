#!/bin/bash
# Check that this folder reproduces the GB tables in pyagadir/params.
#   1. the stored intermediates match MANIFEST.sha256;
#   2. the cheap steps (under 2 minutes, run in parallel) re-derive every GB table from them, and
#      each table's data rows (all lines but the '#' comments) are compared with the package file.
# The expensive steps (structures and their GB energies, MD, GB on MD frames) are not re-run; step 1 checks their
# outputs. Gbx/gbp overrides are not derived by a script (README.md, "Parameter overrides").
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
COIL=$(ls data/n181_coil_cA16_coil{1..8}.npz | paste -sd,)
run dipole "$BUILD_PY" helix_builds/build_gb_dipole_table.py --derive data/gb_raw_v2.tsv.gz \
    --supplement data/gb_raw_v2_strict_supplement.tsv.gz --out "$OUT/sidechain_dipole_gb.tsv" &
run neutral "$BUILD_PY" helix_builds/build_gb_dipole_table.py --derive data/gb_raw_v2.tsv.gz \
    --supplement data/gb_raw_v2_strict_supplement.tsv.gz --neutral --out "$OUT/sidechain_dipole_gb_neutral.tsv" &
run pair "$BUILD_PY" helix_builds/build_gb_pair_table.py --derive-grid data/gb_pair_raw_v3forms.tsv.gz \
    --helix-md data/n181_helix_chi1.npz --coil-md "$COIL" --eps-in 16 --out "$OUT/sidechain_pair_gb.tsv" &
run desolv sh -c "\"$BUILD_PY\" postprocessing/n181_desolv_table.py && cp data/n181_dipole_desolv_avg.tsv \"$OUT/sidechain_dipole_gb_desolv.tsv\"" &
run transfer sh -c "cd \"$ROOT\" && uv run python gb_pipeline/transfer/build_gb_transfer.py --out \"$OUT/sidechain_dipole_gb_transfer.tsv\"" &
run gbp_transfer sh -c "cd \"$ROOT\" && uv run python gb_pipeline/transfer/build_gbp_transfer.py --out \"$OUT/sidechain_gbp_transfer.tsv\"" &
wait

echo "3. data rows against pyagadir/params"
for t in sidechain_dipole_gb sidechain_dipole_gb_neutral sidechain_pair_gb sidechain_dipole_gb_desolv \
         sidechain_dipole_gb_transfer sidechain_gbp_transfer; do
    new="$OUT/$t.tsv"; pkg="$ROOT/pyagadir/params/$t.tsv"
    if [ ! -s "$new" ]; then echo "   MISSING   $t"; fail=1
    elif cmp -s <(grep -v '^#' "$new") <(grep -v '^#' "$pkg"); then echo "   identical $t"
    elif [ $t = sidechain_gbp_transfer ] && (cd "$ROOT" && uv run python - "$new" "$pkg" <<'EOF'
# The gbp transfer is a fixed point solved to |ddG| < 0.0005, |dpsi| < 0.001 and written to 4 decimals: a re-run
# started from the package's values can move a value by one unit in the last digit. Equal within 1e-4 passes.
import sys
import pandas as pd
a, b = (pd.read_csv(f, sep="\t", comment="#") for f in sys.argv[1:])
num = ["dphi", "const"]
same_keys = a.drop(columns=num).equals(b.drop(columns=num))
d = (a[num] - b[num]).abs().to_numpy()
print(f"   within 1e-4 sidechain_gbp_transfer ({int((d > 0).any(axis=1).sum())} rows differ, max {d.max():.4f})")
sys.exit(0 if same_keys and d.max() <= 1.0001e-4 else 1)
EOF
    ); then :
    else echo "   DIFFERENT $t ($(diff <(grep -v '^#' "$new") <(grep -v '^#' "$pkg") | grep -c '^<') rows)"; fail=1; fi
done
[ -n "${KEEP:-}" ] && echo "kept: $OUT" || rm -rf "$OUT"
[ $fail = 0 ] && echo "PASS" || echo "FAIL"
exit $fail
