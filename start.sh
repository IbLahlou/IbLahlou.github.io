#!/bin/bash

# Set up gem environment
export GEM_HOME="$HOME/.gem"
export PATH="$HOME/.gem/bin:$PATH"

# Start Jekyll server
echo "Starting Jekyll server..."
bundle exec jekyll serve --host 0.0.0.0

# The server will be available at http://localhost:4000/
