#!/bin/bash

echo "Stopping Jekyll server..."

# Find and kill Jekyll processes
pkill -f "jekyll serve"

if [ $? -eq 0 ]; then
    echo "Jekyll server stopped successfully."
else
    echo "No Jekyll server process found or failed to stop."
fi
