#!/usr/bin/env bash
# Establish background SSH tunnel to FASRC (authenticate once, valid 18h).
# Run this first, then use fasrc_interactive.sh or kick off cluster jobs.

SOCKET=$(ssh -G fasrc | awk '/^controlpath/{print $2}')

is_mounted() {
    if command -v mountpoint >/dev/null 2>&1; then
        mountpoint -q "$1"
    else
        mount | awk '{print $3}' | grep -qx "$1"
    fi
}

# Returns 0 if the mount point responds, non-zero if stale (sshfs handle dead).
mount_is_live() {
    ls "$1" >/dev/null 2>&1
}

force_unmount() {
    if command -v diskutil >/dev/null 2>&1; then
        diskutil unmount force "$1" >/dev/null 2>&1
    fi
    umount -f "$1" >/dev/null 2>&1 || fusermount -uz "$1" >/dev/null 2>&1 || true
}

ensure_sshfs_mount() {
    local label="$1" remote="$2" mount_point="$3"
    mkdir -p "$mount_point"
    if is_mounted "$mount_point"; then
        if mount_is_live "$mount_point"; then
            echo "$label already mounted: $mount_point"
            return
        fi
        echo "$label mount is stale, remounting: $mount_point"
        force_unmount "$mount_point"
    fi
    sshfs "$remote" "$mount_point"
    echo "$label mounted: $mount_point"
}

if ssh -O check fasrc 2>/dev/null; then
    echo "FASRC tunnel already active: $SOCKET"
else
    echo "Connecting to FASRC — enter password + 2FA when prompted..."
    ssh -fN fasrc
    echo "Tunnel established: $SOCKET"
fi

ensure_sshfs_mount cluster_lab  fasrc:/n/holylfs06/LABS/krajan_lab/Lab/ "$HOME/cluster_lab"
ensure_sshfs_mount cluster_home fasrc:/n/home02/${USER:-$(whoami)}           "$HOME/cluster_home"

# After fasrc_connect.sh, kick off cluster commands via:                   
# ssh fasrc "squeue -u ${USER:-$(whoami)}"
# ssh fasrc "cd ~/mfrefactor/onpolicy/custom/fish/ && ls -al"
# ssh fasrc "cd ~/mfrefactor/onpolicy/custom/fish/ && git pull"
# ssh fasrc "cd ~/mfrefactor/onpolicy/custom/fish/ && git pull && bash cluster_minimal_shs.sh"