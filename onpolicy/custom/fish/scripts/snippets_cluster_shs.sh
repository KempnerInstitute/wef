# RUN THESE WHEN ON CLUSTER (LOGIN NODE)

# Not for batch execution; copy-paste to run...
echo "Not for batch execution; copy-paste to run..."
exit 0

# ============================================================
# COMMON VARS
# ============================================================

FISH=~/mfrefactor/onpolicy/custom/fish
DATA_SHS=/n/holylfs06/LABS/krajan_lab/Lab/${USER:-$(whoami)}/marl_fish_storage/results/
DATA_SJY=/n/holylfs06/LABS/krajan_lab/Lab/sjohnsonyu/marl_fish_storage/results/

cd $FISH
ls $DATA_SHS


