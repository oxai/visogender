"""
This file provides example usage for loading the VISOGENDER data into a dictionary so that it can be used for model evaluation, independent of the code set up in this repository

Author: @smhall97
"""

import os
import sys

main_dir = os.getcwd().split("data")[0]
sys.path.append(main_dir) 

from src.template_generator_utils import load_metadata_to_dict

file_path_OP = "data/visogender_data/OP/OP_Visogender_15082023.tsv"
file_path_OO = "data/visogender_data/OO/OO_Visogender_15082023.tsv"


op_metadata_dict = load_metadata_to_dict(file_path_OP, "OP")
oo_metadata_dict = load_metadata_to_dict(file_path_OO, "OO")
