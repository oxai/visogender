"""
This file either creates or updates an existing benchmark file with the retrieval bias scores for CLIP-like models. 

Author: @hanwenzhu
"""
import os
import sys
import json

import numpy as np

main_dir = os.getcwd().split("analysis")[0]
model_output_path = os.path.join(main_dir, "results/model_outputs/retrieval_bias_clip_occ_first_ContextOP.json")
benchmark_path = os.path.join(main_dir, "results/benchmark_scores")
sys.path.append(main_dir) 

from src.analysis_utils import load_benchmark_dict
from src.metrics import calculate_retrieval_bias


model_list = ["clip"]

for model_name in model_list:

    # names based on original json as saved by output from models
    file_desc = f"clip_{model_name}" 
    exp_desc = "CLIP"

    with open(model_output_path) as f:
        results = json.load(f)
    bias = calculate_retrieval_bias(results)

    output_file_name = f"benchmark_results_{exp_desc}_{model_name}.json"
    benchmark_dict = load_benchmark_dict(benchmark_path, exp_desc, model_name, output_file_name)

    metrics = ['bias@5', 'bias@10', 'maxskew@5', 'maxskew@10', 'NDKL']
    for metric in metrics:
        stats = [bias[o][metric.lower()] for o in bias]
        mean = np.mean(stats)
        # ddof=1 to match with pandas std dev
        std = np.std(stats, ddof=1)
        benchmark_dict["retrieval_bias"][metric]["mean"] = np.round(mean, 2)
        benchmark_dict["retrieval_bias"][metric]["sigma"] = np.round(std, 2)

    with open(os.path.join(benchmark_path, output_file_name), "w") as f:
        json.dump(benchmark_dict, f, indent=4)
    print(f"Saved under {benchmark_path}/{output_file_name}")
