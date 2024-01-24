#!/bin/bash

# Base URL
base_url="https://www.physionet.org/content/santa-fe/1.0.0/"

# File paths
file1="b1.txt"
file2="b2.txt"

# Download the files to the current working directory
echo "Downloading files..."
wget "${base_url}${file1}"
wget "${base_url}${file2}"

# Check if the files have been downloaded successfully
if [[ -f "$file1" && -f "$file2" ]]; then
    echo "Files downloaded successfully. Please check the website for more information: ${base_url}"
else
    echo "Failed to download files."
fi
