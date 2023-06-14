"""
Author: @smhall97 
"""

import os
import sys
import unittest
import requests
import glob
import pandas as pd

main_dir_test = os.getcwd().split("tests")[0]
sys.path.append(main_dir_test) 
from src.definitions import gender_idx_dict
from src.data_utils import load_visogender_data, load_json_to_df, get_image, load_us_labor_statistics, load_full_dataframe, check_op_and_oo_both_exist, check_op_and_oo_both_exist_preliminary_analysis


class TestDataUtils(unittest.TestCase):
    def setUp(self):
        self.tests_dir = os.getcwd()
        self.op_filepath = f"{self.tests_dir}/test_data/test_visogender/Test_OP_Visogender_11062023.tsv"
        self.oo_filepath = f"{self.tests_dir}/test_data/test_visogender/Test_OO_Visogender_11062023.tsv"
        self.test_image_valid = "https://images.pexels.com/photos/8617942/pexels-photo-8617942.jpeg?auto=compress&cs=tinysrgb&w=600&lazy=load"
        self.test_image_invalid = "https://"
        self.test_json_path = "tests/test_data/test_mini_json.json"
        self.test_json_path2 = "tests/test_data/test_model_outputs/Test_clip_clip_ContextOP.json"
        self.input_params_dict = {
                        "experiment_name" : "unittests",
                        "OP_data": self.op_filepath,
                        "OO_data": self.oo_filepath,
                        "bias_experiments": ["resolution"], # "retrieval bias if cliplike"
                        "template_type": ["occ_first", "par_first"],
                        "sentence_template_OP_occ_first": "A $OCCUPATION and ",
                        "sentence_template_OP_par_first": "A $PARTICIPANT and ",
                        "sentence_template_OO": "A $OCCUPATION and ",
                        "gender_idx_dict" : gender_idx_dict,
                        "caption_models": ["test_model_1", "test_model_2"]}
        self.test_template_occ_first = "A $OCCUPATION and "
        self.test_template_par_first = "A $PARTICIPANT and "
  
    def test_load_visogender_data(self):
        """Tests that the correct paths and templates are returned depending on the context OP / OO"""
        context_OP = [True, False]
        for op in context_OP:
            if op:
                sentence_path, op_occ_first_sentence, op_par_first_sentence = load_visogender_data(self.input_params_dict, op, False)
                self.assertIsNotNone(op_par_first_sentence)
                self.assertEqual(sentence_path, self.op_filepath)
                self.assertEqual(op_occ_first_sentence, self.test_template_occ_first)
                self.assertEqual(op_par_first_sentence, self.test_template_par_first)
            else:
                oo_sentence_path, oo_occ_first_sentence, oo_none = load_visogender_data(self.input_params_dict, op, True)
                self.assertIsNone(oo_none)
                self.assertEqual(oo_sentence_path, self.oo_filepath)
                self.assertEqual(oo_occ_first_sentence, self.test_template_occ_first)

        context_OO = [True, False]
        for oo in context_OO:
            if oo:
                _, _, op_par_first_sentence = load_visogender_data(self.input_params_dict, False, oo)
                self.assertIsNone(op_par_first_sentence)
                self.assertEqual(oo_sentence_path, self.oo_filepath)
                self.assertEqual(oo_occ_first_sentence, self.test_template_occ_first)
            else:
                _, _, op_par_first_sentence = load_visogender_data(self.input_params_dict, True, oo)
                self.assertIsNotNone(op_par_first_sentence)
                self.assertEqual(sentence_path, self.op_filepath)
                self.assertEqual(op_occ_first_sentence, self.test_template_occ_first)
                self.assertEqual(op_par_first_sentence, self.test_template_par_first)

    def test_load_json_to_df(self):
        """Tests that a pandas dataframe is returned when loading a json"""
        test_df = load_json_to_df(self.test_json_path)
        self.assertIsInstance(test_df, pd.DataFrame)
        
        test_df_2 = load_json_to_df(self.test_json_path2)
        self.assertIsInstance(test_df_2, pd.DataFrame)
        
    def test_get_image(self):
        """Tests that an image is loaded from a URL"""
        #Note: this test relies on an image hosted on Visogender and if it fails, one should check the image still exists
        # The invalid URL does not exist
        valid_img = get_image(self.test_image_valid)
        self.assertIsNotNone(valid_img)

        with self.assertRaises(requests.exceptions.InvalidURL):
            self.assertRaises(get_image(self.test_image_invalid))

    
    def test_load_full_dataframe(self):
        """Tests that the full dataframe is loaded"""
        dataframe = load_full_dataframe("tests/test_data/test_prelim_files")

        self.assertIsInstance(dataframe, pd.DataFrame)

    def test_load_us_labor_statistics(self):
        """Tests that the US Labor Force Statistics are loaded into a dataframe"""
        path_to_us = "tests/test_data/test_us_data/test_US_Visogender_mapping_statistics_11062023.tsv"
        test_us_df = load_us_labor_statistics(path_us_stats=path_to_us)

        self.assertIsInstance(test_us_df, pd.DataFrame)
        self.assertFalse(test_us_df.empty)

        occupation_values = test_us_df["Visogender Occupations"].tolist()
        occupation_list = ["teacher", "physician", "engineer", "clerk", "baker"]

        self.assertEqual(len(occupation_list), len(occupation_values))
        self.assertEqual(occupation_list, occupation_values)

    def test_check_op_and_oo_both_exist(self):
        """Tests that both context OO and OP exist in any directory"""
        directory_path_both = "tests/test_data/test_folder_dir_both/"
        self.assertTrue(check_op_and_oo_both_exist(directory_path_both, "clip"))
        self.assertTrue(check_op_and_oo_both_exist(directory_path_both, "blipv2"))

        directory_path_single = "tests/test_data/test_folder_dir_single/"
        with self.assertRaises(FileNotFoundError):
            self.assertRaises(check_op_and_oo_both_exist(directory_path_single, "clip"))

        with self.assertRaises(FileNotFoundError):
            self.assertRaises(check_op_and_oo_both_exist(directory_path_single, "blipv2"))

    def test_check_op_and_oo_both_exist_preliminary_analysis(self):
        """Tests that the preliminary analysis has been run for both contexts OP / OO"""

        file_path_with_both = "tests/test_data/test_prelim_files/"

        check_both_clip = check_op_and_oo_both_exist_preliminary_analysis(file_path_with_both, "clip")   
        self.assertTrue(check_both_clip)    
        check_both_cap = check_op_and_oo_both_exist_preliminary_analysis(file_path_with_both, "blipv2")   
        self.assertTrue(check_both_cap)  

        file_path_without = "tests/test_data/test_prelim_no_files/"

        check_both_clip = check_op_and_oo_both_exist_preliminary_analysis(file_path_without, "clip")   
        self.assertFalse(check_both_clip)    
        check_both_cap = check_op_and_oo_both_exist_preliminary_analysis(file_path_without, "blipv2")   
        self.assertFalse(check_both_cap)  

        
