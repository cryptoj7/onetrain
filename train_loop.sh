#!/usr/bin/env bash

# List of config files (adjust these to your actual 5 configs)
configs=(
  "li4.json"
  "geo-lora.json"
)

# List of input datasets (adjust these to your actual 3 datasets)
datasets=(
  "Ali"
  "Angel"
  "Anna"
  "Ailiyah"
  "Gwen"
  "Fallon"
  "Tatiana"
)

# Where your onetrainer script is located
onetrainer_path="/app/onetrainer"

# Base model
model_name="black-forest-labs/FLUX.1-dev"

generate_random_concept() {
  # Random length between 3 and 6
  local length=$(( (RANDOM % 4) + 3 ))
  # Excluding vowels from the character set:
  local chars='B-DF-HJ-NP-TV-Zb-df-hj-np-tv-z0-9'
  
  # Pull from /dev/urandom, filter to our chars, take 'length' chars
  tr -dc "$chars" < /dev/urandom | head -c "$length"
}

# Loop over each config/dataset pair
for config in "${configs[@]}"; do
    config_basename="$(basename "$config" .json)"  # e.g. "vlad-lora" from "vlad-lora.json"

    for dataset in "${datasets[@]}"; do
        dataset_basename="$(basename "$dataset")"

        # Construct unique filenames/paths
        log_file="/workspace/train-dir/${dataset_basename}_${config_basename}.log"
        output_file="/workspace/data-dir/models/Lora/${dataset_basename}_${config_basename}.safetensors"
        tmp_dir="/workspace/train-dir/tmp_${dataset_basename}_${config_basename}"
        # Generate a random concept (3–6 chars, no vowels)
        concept="$(generate_random_concept)"

        # Decide which concept string to use
        if [[ "$dataset_basename" == "Ali" || "$dataset_basename" == "Angel" ]]; then
            concept_str="n0xyz man"
        else
            concept_str="n0xyz woman"
        fi

        echo "Now training on config: $config, dataset: $dataset"
        echo "Random concept is: $concept"

        # Run training under nohup so it keeps going if you disconnect
        nohup python onetrain.py \
          --onetrainer "$onetrainer_path" \
          --model "$model_name" \
          --input "/workspace/concept-dir/training_datasets/${dataset}" \
          --log "$log_file" \
          --output "$output_file" \
          --tmp "$tmp_dir" \
          --concept "$concept_str" \
          --config "$config" \
          --type flux \
          --sample \
          --debug \
          --caption \
          > "/workspace/train-dir/nohup_${dataset_basename}_${config_basename}.out" 2>&1
        
        # Optional: If you want to run each training job sequentially (one after another),
        # remove the "&" above and uncomment "wait" below.
        #
        wait
    done
done
