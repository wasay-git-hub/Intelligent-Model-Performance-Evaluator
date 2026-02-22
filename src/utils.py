import yaml
import os

def load_params():

    # Get the directory where THIS file (config.py) is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct the path to params.yaml (assuming it is in the same 'src' folder)
    params_path = os.path.join(current_dir, "params.yaml")

    if not os.path.exists(params_path):
        raise FileNotFoundError(f"Could not find params.yaml at {params_path}")

    with open(params_path, "r") as f:
        config = yaml.safe_load(f)
        
    return config

def get_mappings():

    # This ensures it works even if you run the script from a different folder
    current_dir = os.path.dirname(os.path.abspath(__file__))
    params_path = os.path.join(current_dir, "params.yaml")

    with open(params_path, "r") as f:
        config = yaml.safe_load(f)
    
    return config['mappings']