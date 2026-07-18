#!/usr/bin/env bash
# Open an interactive GPU session on FASRC.
# Usage:
#   ./fasrc_interactive.sh              # default: 12h, 1 GPU, 12 CPUs, 28G
#   TIME=4:00:00 ./fasrc_interactive.sh # override time

TIME=${TIME:-12:00:00}

if ! ssh -O check fasrc 2>/dev/null; then
    echo "No active FASRC tunnel. Run ./fasrc_connect.sh first."
    exit 1
fi

echo "Connecting to FASRC (interactive session, time=${TIME})..."
ssh -t fasrc "
    echo '--- launching interactive session ---'
    salloc --partition=kempner \
           --account=kempner_krajan_lab \
           --time=${TIME} \
           --mem=28G \
           --gres=gpu:1 \
           --cpus-per-task=12 \
           srun --pty bash -i
"
