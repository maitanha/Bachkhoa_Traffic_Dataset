#!/bin/bash

read -p "Enter number of epochs (default will be 10): " epochs

read -p "Enter model name (default is resnet_traffic_model.pth):" model_name

read -p "Enter data path (empty path will use ./data/full): " data_path

# Start building the command
command="python3 train.py"

if [ -n "$epochs" ]; then
  command="$command --epochs $epochs"
fi

# Add model name
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
