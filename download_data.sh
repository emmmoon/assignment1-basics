#!/bin/bash

# Create data directory and enter it
mkdir -p data
cd data

# Download TinyStories dataset
curl -O -L https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
curl -O -L https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

# Go back to previous directory
cd ..