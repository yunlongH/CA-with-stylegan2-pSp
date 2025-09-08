import os
import json
import matplotlib.pyplot as plt
import torch
from argparse import Namespace
import pandas as pd

def write_log_to_txt(log_msg, results_dir, filename):
    with open(f"{results_dir}/logs/{filename}", 'a') as f:
        f.write(log_msg)

def save_hyparams(args):
    """Save hyperparameters to JSON."""
    args_json_path = os.path.join(args.results_dir, 'hyparams.json')
    with open(args_json_path, 'w') as f:
        json.dump(vars(args), f, indent=4, sort_keys=True)
    print(f"Arguments saved to {args_json_path}")

def load_hyparams_from_json(json_path):
    """Load arguments from a JSON file into an argparse.Namespace."""
    with open(json_path, 'r') as f:
        args_dict = json.load(f)
    return Namespace(**args_dict)



def plot_loss_from_file(file_path, loss_name):
    """
    Reads loss data from a file and plots the specified loss type.
    
    Parameters:
        file_path (str): Path to the loss log file.
        loss_name (str): The name of the loss to plot (e.g., "Total Loss", "Loss BG", "Loss T", "Loss SBG").
    """
    try:
        # Read the file
        with open(file_path, "r") as file:
            lines = file.readlines()

        # Extract loss data
        epochs = []
        loss_values = []
        
        for line in lines:
            parts = line.strip().split("|")
            epoch_info = parts[0].strip().split(" ")
            epoch = int(epoch_info[1].split("/")[0])  # Extract epoch number

            # Extract loss values
            loss_map = {
                "Total Loss": float(parts[1].split(":")[1].strip()),
                "Loss bg": float(parts[2].split(":")[1].strip()),
                "Loss t": float(parts[3].split(":")[1].strip()),
                "Loss sbg": float(parts[4].split(":")[1].strip())
            }
            
            if loss_name not in loss_map:
                print(f"Invalid loss name. Choose from: {list(loss_map.keys())}")
                return
            
            epochs.append(epoch)
            loss_values.append(loss_map[loss_name])

        # Plot the loss curve
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, loss_values, label=loss_name, color='blue')
        plt.xlabel("Epochs")
        plt.ylabel(loss_name)
        plt.title(f"{loss_name} Over Epochs")
        plt.legend()
        plt.grid(True)
        plt.show()

    except FileNotFoundError:
        print(f"File {file_path} not found. Please check the path.")
