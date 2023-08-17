"""
This file provides example usage for loading the VISOGENDER data into a dictionary so that it can be used for model evaluation, independent of the code set up in this repository

Author: @smhall97
"""

import os
import sys

main_dir = os.getcwd().split("data")[0]
sys.path.append(main_dir) 

from src.template_generator_utils import load_metadata_to_dict
from src.definitions import OP_data_filepath, OO_data_filepath

file_path_OP = f"data/visogender_data/{OP_data_filepath}"
file_path_OO = f"data/visogender_data/{OO_data_filepath}"


op_metadata_dict = load_metadata_to_dict(file_path_OP, "OP")
oo_metadata_dict = load_metadata_to_dict(file_path_OO, "OO")
