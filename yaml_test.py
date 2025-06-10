import yaml
import argparse
import os

def test_load_config(config_path):
    print(f"\n--- Testing config file: {config_path} ---")
    if not os.path.exists(config_path):
        print(f"Error: File not found at {config_path}")
        return

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        print(f"Loaded config dictionary: {config}")

        # Check if 'noise_scale' key exists
        if 'noise_scale' in config:
            retrieved_noise_scale = config['noise_scale']
            print(f"Value for 'noise_scale' found: {retrieved_noise_scale}")
            print(f"Type of 'noise_scale': {type(retrieved_noise_scale)}")
        else:
            print("ERROR: 'noise_scale' key NOT found in the loaded config!")

        # Also test with .get()
        retrieved_noise_scale_get = config.get('noise_scale', 'DEFAULT_VAL_IF_NOT_FOUND')
        print(f"Value for 'noise_scale' (using .get()): {retrieved_noise_scale_get}")

    except yaml.YAMLError as e:
        print(f"ERROR: YAML parsing error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    test_load_config(args.config)