#!/bin/bash

# Usage function
usage() {
  echo "Usage: $0 [-rm | -rrm] folder_or_wildcard..."
  echo "  -rm   : Remove .gif after conversion to .mp4"
  echo "  -rrm  : Remove .gif if corresponding .mp4 already exists"
  exit 1
}

# Check for options
REMOVE_GIF_AFTER_CONVERSION=false
REMOVE_EXISTING_GIF=false

while [[ "$1" == -* ]]; do
  case "$1" in
    -rm) REMOVE_GIF_AFTER_CONVERSION=true ;;
    -rrm) REMOVE_EXISTING_GIF=true ;;
    *) usage ;;
  esac
  shift
done

# Check if at least one folder or wildcard is provided
if [ "$#" -eq 0 ]; then
  usage
fi

# Iterate over each provided folder or wildcard pattern
for GIF_FOLDER in "$@"; do
  # Recursively find all .gif files in the folder(s) and subfolders
  find "$GIF_FOLDER" -type f -name "*.gif" | while read -r G; do
    # Define the corresponding .mp4 filename
    MP4="${G%.gif}.mp4"
    
    # If -rrm is set and the .mp4 already exists, remove the .gif
    if [[ "$REMOVE_EXISTING_GIF" == true && -f "$MP4" ]]; then
      echo "Removing existing GIF: $G (MP4 already exists)"
      rm "$G"
      continue
    fi

    # Only convert if the .mp4 file doesn't already exist
    if [[ ! -f "$MP4" ]]; then
      echo "Converting $G to $MP4"
      ffmpeg -i "$G" "$MP4"
      # If -rm is set, remove the .gif after conversion
      if [[ "$REMOVE_GIF_AFTER_CONVERSION" == true ]]; then
        echo "Removing GIF: $G"
        rm "$G"
      fi
    else
      echo "$MP4 already exists, skipping."
    fi
  done
done



# #!/bin/bash

# # Check if at least one folder or wildcard is provided as a command-line argument
# if [ "$#" -eq 0 ]; then
#   echo "Please provide at least one folder or wildcard pattern."
#   exit 1
# fi

# # Iterate over each provided folder or wildcard pattern
# for GIF_FOLDER in "$@"; do
#   # Recursively find all .gif files in the folder(s) and subfolders
#   find $GIF_FOLDER -type f -name "*.gif" | while read -r G; do
#     # Define the corresponding .mp4 filename
#     MP4="${G%.gif}.mp4"
    
#     # Only convert if the .mp4 file doesn't already exist
#     if [[ ! -f "$MP4" ]]; then
#       echo "Converting $G to $MP4"
#       ffmpeg -i "$G" "$MP4"
#     else
#       echo "$MP4 already exists, skipping."
#     fi
#   done
# done
