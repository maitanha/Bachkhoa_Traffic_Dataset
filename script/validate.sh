#!/bin/bash

# User input model name
read -p "Enter pretrained model name (empty name will use default model): " model_name

# User input data_path
read -p "Enter data path (empty path will use ./data/full): " data_path

# Start building the command
command="python3 validate.py"

# Add the model name if provided
if [ -n "$model_name" ]; then
  command="$command --model_name $model_name"
fi

# Add the data path if provided
if [ -n "$data_path" ]; then
  command="$command --data_path $data_path"
fi

# Run the final command
echo "Running command: $command"
eval "$command"
