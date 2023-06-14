"""
These functions are used to support all data related actions, such as loading files to dictionary / dataframes and/or saving these to jsons.

Author: @smhall97, @abrantesfg, @hanwenzhu

"""
import os
import requests
import json
import glob
import pandas as pd
from glob import iglob
from PIL import Image
from pathlib import Path

def load_visogender_data(input_params_dict: dict, context_OP: bool, context_OO: bool):
    """
    Returns the metadata required for setting up the data and templates for the occ-par an occ-obj contexts

    Args:
        input_params_dict: input params as set up per model type detailing the experiments, models and data paths
        context_OP: if True, the context is set up for the occ-par and par-occ runs
        context_OO: if True, the context is set up for the occ-obj
    """

    if context_OP:
        return input_params_dict["OP_data"], input_params_dict["sentence_template_OP_occ_first"], input_params_dict["sentence_template_OP_par_first"]

    elif context_OO:
        return input_params_dict["OO_data"], input_params_dict["sentence_template_OO"], None

def save_df_to_json(dataframe, filepath:str, exp_description:str):

    """
    Saves the augmented dataframe into as a json in the designate filepath
    Args:
        df: dataframe icnluding results, neutral and match ground truth checks
        context: either occupation- participant or occupation-object
        filepath: path where the json file will be saved
        exp_description: description of the experiment, should be consistent across all analyses processes
    """
    output_file_name = f"{exp_description}.json"

    dataframe = dataframe.transpose()
    dataframe.to_json(os.path.join(filepath,output_file_name))
    return dataframe

def load_json_to_df(json_filepath: str):
    """
    This function loads the summary json data to a pandas dataframe format to be used for analysis
    Args:
        json_filepath: string to file with the results data
    Returns:
        metadata dataframe with all image information (categories, ground truth labels, logits)
    """
    with open(json_filepath) as f:
        data = pd.read_json(f)
        
    dataframe = pd.DataFrame(data)
    dataframe = dataframe.transpose()
    return dataframe

def save_dict_json(results_dict: dict, context_OP: bool, context_OO: bool, filepath:str, exp_description: str):

    """
    Saves the dictionary with results as a json in the designate filepath
    Args:
        results_dict: dictionary with results, the exp_description is the main identifier of the experiment
        context: either occupation- participant or occupation-object
        category_slice: indicate how the data should be split according to category (main categories, sub categories, occupations)
        filepath: path where the json file will be saved
        exp_description: description of the experiment, should be consistent across all analyses processes
    """
    if context_OP:
        output_file_name = f"{exp_description}_ContextOP.json"
    elif context_OO:
        output_file_name = f"{exp_description}_ContextOO.json"

    with open(os.path.join(filepath,output_file_name), "w") as f:
        json.dump(results_dict, f, indent=4)
    if context_OP:
        print(f"Saved under {filepath}/{exp_description}_ContextOP.json")
    else:
        print(f"Saved under {filepath}/{exp_description}_ContextOO.json")

def get_image(image_url: str):

    """
    Returns an image from the metadata URL to be used in the pipeline

    Args:
        image_url: URL to image hosted online
    """
    headers = {"User-Agent": "OxAI"}
    r = requests.get(image_url, stream=True, headers=headers)
    return Image.open(r.raw).convert("RGB")

def load_full_dataframe(directory_path)-> pd.DataFrame:
    """
    Returns the full dataframe, over all models run for clip-like and/or captioning

    Args:
        directory_path: path to the folder containing files with model outputs
    
    """
    dataframe_list = []
    dir_path = Path(directory_path)

    for file in iglob(str(dir_path / "*.json")):
        df = load_json_to_df(file)
        dataframe_list.append(df)

    full_dataframe = pd.concat(dataframe_list, ignore_index=True)

    return full_dataframe


def load_us_labor_mapping(filename: str) -> dict:
    """
    Loads US labor statistics as a mapping from occupation name to share of men (0-100)
    """
    df = pd.read_csv(filename, sep="\t")
    return dict(zip(df["Visogender Occupations"], df["male_proportion"]))

def load_us_labor_statistics(path_us_stats):

    us_stats = pd.read_csv(path_us_stats, sep="\t", header=0)   
    return us_stats

def check_op_and_oo_both_exist(directory_path, model_name):
    """
    Checks if both Context OO and Context OP files exist in the specified directory for the given model name.
    
    Args:
        directory_path (str): The path of the directory to search for files.
        model_name (str): The name of the model.
    
    Returns:
        bool: True if both 'OP' and 'OO' keywords exist in the directory for the model name.
    
    Raises:
        FileNotFoundError: If either 'OP' or 'OO' keyword is missing in the directory for the model name.
    """    
    # Search for files containing "OP" and "OO" keywords
    files_with_op = glob.glob(directory_path + f"/*{model_name}*OP*")
    files_with_oo = glob.glob(directory_path + f"/*{model_name}*OO*")

    # Check if both keywords exist
    if files_with_op and files_with_oo:
        print(f"The model outputs exist for the OO and OP context for the model: {model_name}")
        return True
        
    else:
        raise FileNotFoundError(f"Either 'OP' or 'OO' keyword is missing in the /results/model_outputs directory for the model {model_name}. Please run the resolution bias code.")

def check_op_and_oo_both_exist_preliminary_analysis(directory_path, model_name):
    """
    Checks if both Context OO and Context OP files exist in the specified directory for the given model.

    Args:
        directory_path (str): Path to the directory where the preliminary results are saved.
        model_name (str): Name of the model.

    Returns:
        bool: True if both "OP" and "OO" files exist
    """    
    # Search for files containing "OP" and "OO" keywords
    files_with_op = glob.glob(directory_path + f"/*{model_name}*OP*")
    files_with_oo = glob.glob(directory_path + f"/*{model_name}*OO*")

    # Check if both keywords exist
    if files_with_op and files_with_oo:
        print(f"Both 'OP' and 'OO' keywords exist in the preliminary_analysis directory for the model {model_name}.")
        return True
        
    else:
        print(f"Either 'OP' and 'OO' keywords do not exist in the preliminary_analysis directory for the model {model_name} and has been created.")