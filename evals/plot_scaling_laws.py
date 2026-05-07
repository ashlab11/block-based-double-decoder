"""
Plot scaling laws across tokens, parameters, and FLOPS.

Usage:

    python evals/plot_scaling_laws.py --input_file scaling_laws.json --output_dir plots
"""

import argparse
import json
import os
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot scaling laws across tokens, parameters, and FLOPS.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file containing scaling law data.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the generated plots.")
    args = parser.parse_args()

    # Load the scaling law data from the JSON file
    with open(args.input_file, "r") as f:
        data = json.load(f)

    # Convert the data into a pandas DataFrame for easier plotting
    df = pd.DataFrame(data)
    print(df.head)