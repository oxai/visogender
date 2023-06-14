"""
This file runs the preliminary analysis for resolution bias. The outputs of this file can be used for the full analysis (resolution accuracy and gender gap)
as well as run ablation studies on flipped templates and neutral resolution

Authors: @smhall97, @abrantesfg
"""

import os
import sys

main_dir = os.getcwd().split("analysis")[0]
result_dir = os.path.join(main_dir, "results/model_outputs")
saving_path = os.path.join(main_dir, "results/resolution_bias_analysis/preliminary_analysis")
sys.path.append(main_dir) 

from src.data_utils import check_op_and_oo_both_exist
from src.analysis_utils import check_neutral_groundtruth_match


clip_models = ["clip"]
captioning_models = ["blipv2"]

model_list = clip_models + captioning_models

for model_name in model_list:

    # names based on original json as saved by output from models
    if model_name in clip_models:
        file_desc = f"clip_{model_name}" 
        exp_desc = "CLIP" 

    elif model_name in captioning_models:
        file_desc = f"captioning_{model_name}" 
        exp_desc = "CAPTIONING" 

    file_check = check_op_and_oo_both_exist(result_dir, model_name)
    df = check_neutral_groundtruth_match(result_dir, saving_path, file_desc, exp_desc, model_name)


