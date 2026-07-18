#!/bin/bash

# Usage function
usage() {
  echo "Usage: $0 <file_or_folder> [more files/folders...]"
  exit 1
}

# Exit if no arguments
if [ "$#" -eq 0 ]; then
  usage
fi

# Function to convert a single .mp4 file to .gif
convert_mp4_to_vid() {
  local MP4="$1"
  local GIF="${MP4%.mp4}.gif"
  local PALETTE="/tmp/palette_$$.png"

  # Skip if GIF already exists
  if [[ -f "$GIF" ]]; then
    echo "$GIF already exists, skipping."
    return
  fi

  echo "Converting $MP4 to $GIF"

  # First pass: generate palette
  ffmpeg -v warning -i "$MP4" -vf "fps=15,scale=640:-1:flags=lanczos,palettegen=stats_mode=full" -frames:v 1 -y "$PALETTE"

  # Second pass: use palette to make high-quality GIF
  ffmpeg -v warning -i "$MP4" -i "$PALETTE" \
    -filter_complex "fps=15,scale=640:-1:flags=lanczos[x];[x][1:v]paletteuse" \
    -y "$GIF"

  rm -f "$PALETTE"
}

# Iterate over each input argument
for INPUT in "$@"; do
  if [[ -f "$INPUT" && "$INPUT" == *.mp4 ]]; then
    convert_mp4_to_vid "$INPUT"
  elif [[ -d "$INPUT" ]]; then
    find "$INPUT" -type f -name "*.mp4" | while read -r MP4; do
      convert_mp4_to_vid "$MP4"
    done
  else
    echo "Skipping unrecognized input: $INPUT"
  fi
done
