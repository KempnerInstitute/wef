# Usage:
#   RUN_DIR is the run directory (contains logs/, evals/, figs/).
#   bash scripts/regenerate_latex.sh $RUN_DIR
# $RUN_DIR is the run directory (contains logs/, evals/, figs/).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEX_DIR="$(cd "$SCRIPT_DIR/../manuscript" 2>/dev/null && pwd)" || {
    echo "ERROR: could not resolve manuscript/ directory" >&2; exit 1
}

if [[ $# -lt 1 ]]; then
    echo "Usage: bash regenerate_latex.sh <run_dir>" >&2
    exit 1
fi

RUN_DIR="$(realpath "$1")"
FIGS_DIR="$RUN_DIR/figs"
if [[ ! -d "$FIGS_DIR" ]]; then
    echo "ERROR: figs/ not found in $RUN_DIR" >&2
    exit 1
fi

echo "figs/  : $FIGS_DIR"
echo "manuscript/ : $TEX_DIR"
echo

# Copy the two tex files needed to build — figures stay in place
cp "$TEX_DIR/figseed.tex"   "$FIGS_DIR/figseed.tex"
cp "$TEX_DIR/figmacros.tex" "$FIGS_DIR/figmacros.tex"
cp -rf "$TEX_DIR/manual" "$FIGS_DIR/"
# Always copy supp/ as a real directory; remove symlink first if present
if [[ -L "$FIGS_DIR/supp" ]]; then
    rm "$FIGS_DIR/supp"
    echo "  removed supp/ symlink"
fi
cp -rf "$TEX_DIR/supp" "$FIGS_DIR/"

# Refresh script-generated real-fish plots from real_fish_data/ into figs/manual/
# (works whether figs/manual is a real dir or a symlink)
FISH_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
for f in real_fish_dyad_ipi_histogram.pdf real_fish_dyad_ipi_powerlaw.pdf real_fish_dyad_ipi_gebhardt.pdf; do
    src="$FISH_DIR/real_fish_data/$f"
    [[ -f "$src" ]] && cp "$src" "$FIGS_DIR/manual/$f" && echo "  refreshed manual/$f from real_fish_data/"
done

echo "Copied figseed.tex + figmacros.tex + supp/ → $FIGS_DIR"
echo

echo "Building figseed.pdf ..."
cd "$FIGS_DIR"
pdflatex -interaction=nonstopmode figseed.tex > figseed_compile.log 2>&1 || true
pdflatex -interaction=nonstopmode figseed.tex >> figseed_compile.log 2>&1 || true

echo
echo "Done: $FIGS_DIR/figseed.pdf"

# Sync figure directories into the repo's tex/ so that pdflatex tex/00_main.tex
# picks up the same files that went into figseed.pdf.
REPO_TEX="$(cd "$SCRIPT_DIR/../../../../tex" && pwd)"
echo
echo "Syncing fig dirs → $REPO_TEX ..."
for d in fig3 fig4 fig5 fig6 supp manual; do
    if [[ -d "$FIGS_DIR/$d" ]]; then
        rsync -a "$FIGS_DIR/$d/" "$REPO_TEX/$d/"
        echo "  synced $d/"
    fi
done
echo "Sync done — tex/ is now up to date."
