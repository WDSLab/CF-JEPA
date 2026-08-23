#!/bin/bash
# ============================================================
# Clone all baseline repos into the correct directories.
# Run this once after cloning cf_jepa on a new machine.
# ============================================================

set -e
cd "$(dirname "$0")"

echo "Cloning baseline repos..."

clone_if_missing() {
    local dir=$1
    local url=$2
    if [ -d "$dir" ]; then
        echo "  [SKIP] $dir already exists"
    else
        echo "  [CLONE] $url -> $dir"
        git clone --depth 1 "$url" "$dir"
    fi
}

clone_if_missing baselines/ts2vec/ts2vec          https://github.com/zhihanyue/ts2vec.git
clone_if_missing baselines/trep/T-Rep             https://github.com/Let-it-Care/T-Rep.git
clone_if_missing baselines/autotcl/AutoTCL        https://github.com/AslanDing/AutoTCL.git
clone_if_missing baselines/softclt/softclt        https://github.com/seunghan96/softclt.git
clone_if_missing baselines/timesurl/TimesURL      https://github.com/Alrash/TimesURL.git
clone_if_missing baselines/infots/InfoTS          https://github.com/chengw07/InfoTS.git
clone_if_missing baselines/cost/CoST             https://github.com/salesforce/CoST.git
clone_if_missing baselines/simmtm/SimMTM          https://github.com/thuml/SimMTM.git
clone_if_missing baselines/ts_jepa/TS_JEPA        https://github.com/Sennadir/TS_JEPA.git
clone_if_missing baselines/fei/Frequency-masked-Embedding-Inference \
                                                   https://github.com/USTBInnovationPark/Frequency-masked-Embedding-Inference.git

echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "Done! All baselines ready."
