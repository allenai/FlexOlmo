#!/bin/bash

# Script to diff files between two directories

# Check if two arguments are provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 <directory1> <directory2>"
    exit 1
fi

DIR1="$1"
DIR2="$2"

# Check if both directories exist
if [ ! -d "$DIR1" ]; then
    echo "Error: Directory '$DIR1' does not exist"
    exit 1
fi

if [ ! -d "$DIR2" ]; then
    echo "Error: Directory '$DIR2' does not exist"
    exit 1
fi

# Loop through files in the first directory
for file in "$DIR1"/*; do
    # Get just the filename without the path
    filename=$(basename "$file")
    
    # Check if it's a regular file (not a directory)
    if [ -f "$file" ]; then
        # Check if the corresponding file exists in DIR2
        if [ -f "$DIR2/$filename" ]; then
            echo "========================================"
            echo "Comparing: $filename"
            echo "========================================"
            diff "$file" "$DIR2/$filename"
            
            # Check the exit status of diff
            if [ $? -eq 0 ]; then
                echo "Files are identical"
            fi
            echo ""
        else
            echo "Warning: $filename exists in $DIR1 but not in $DIR2"
            echo ""
        fi
    fi
done