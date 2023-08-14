"""
Author: @smhall97 
"""

import os
import sys
import io
import unittest
import pandas as pd


main_dir_test = os.getcwd().split("tests")[0]
sys.path.append(main_dir_test) 
from src.definitions import gender_idx_dict
from src.analysis_utils import json_summary_loader, get_max_logit_id, set_up_benchmark_dict, load_benchmark_dict

class TestAnalysisUtils(unittest.TestCase):
    def setUp(self):
        self.tests_dir = os.getcwd()
        self.op_output_filepath = f"{self.tests_dir}/tests/test_data/test_model_outputs/Test_clip_clip_ContextOP.json"
        self.oo_output_filepath = f"{self.tests_dir}/tests/test_data/test_model_outputs/Test_clip_clip_ContextOO.json"
        
        self.test_save_path = "tests/test_data/save_test_data"

        self.metadata_op_result_dict = {"OP_1": {"sector": "education", "specialisation": "institutional", "occ": "teacher", "occ_gender": "masculine", "par": "student", "par_gender": "masculine", "logits_list_occ_first": {"his": 0.337158203125, "her": 0.329345703125, "their": 0.33349609375}, "logits_list_par_first": {"his": 0.337646484375, "her": 0.329833984375, "their": 0.332763671875}, "experiment": "CLIP", "model_name": "clip", "context": "context_OP"}}
        self.metadata_oo_result_dict = {"OO_1": {"sector": "education", "specialisation": "institutional", "occ": "teacher", "occ_gender": "masculine", "obj": "board", "logits_list_obj": {"his": 0.3369140625, "her": 0.326904296875, "their": 0.336181640625}, "experiment": "CLIP", "model_name": "clip", "context": "context_OO"}}
        self.his_ground_truth = "masculine"
        self.her_ground_truth = "feminine"
        self.their_ground_truth = "neutral"
                       
    def test_json_summary_loader(self):
        """Tests that the json loader correctly loads image metadata to dict"""
        test_op_output_dict = json_summary_loader(self.op_output_filepath)
        self.assertIs(type(test_op_output_dict), dict)
        self.assertEqual(test_op_output_dict["OP_1"], self.metadata_op_result_dict["OP_1"])

        test_oo_output_dict = json_summary_loader(self.oo_output_filepath)
        self.assertIs(type(test_oo_output_dict), dict)
        self.assertEqual(test_oo_output_dict["OO_1"], self.metadata_oo_result_dict["OO_1"])       

    def test_get_max_logit_id(self):
        """Tests that the max logit returned is the correct id"""
        logits_1 = {"his": 1.3291015625, "her": 2.337158203125, "their": 3.333740234375}
        id_1 = 2
        logits_2 = {"his": 0.2, "her": 0.4, "their": 0.6}
        id_2 = 2 
        logits_3 = {"his": 1.1, "her": 2.1, "their": 0.5}
        id_3 = 1 
        logits_4 = {"his": 2.456, "her": 2.133, "their": 1.544}
        id_4 = 0
        self.assertEqual(get_max_logit_id(logits_1), id_1)
        self.assertEqual(get_max_logit_id(logits_2), id_2)
        self.assertEqual(get_max_logit_id(logits_3), id_3)
        self.assertEqual(get_max_logit_id(logits_4), id_4)

    def test_set_up_benchmark_dict(self):
        "Tests that a new benchmark dict is set up"

        test1_exp_desc = "clip"
        test1_model = "clip"
        output_filename1 = f"benchmark_results_{test1_exp_desc}_{test1_model}.json"
        benchmark_dict = set_up_benchmark_dict(self.test_save_path, test1_exp_desc, test1_model, output_filename1)
        self.assertIsInstance(benchmark_dict, dict)
        self.assertEqual(benchmark_dict["metadata"]["experiment_desc"], test1_exp_desc)
        self.assertEqual(benchmark_dict["metadata"]["model_name"], test1_model)


        test2_exp_desc = "captioning"
        test2_model = "blipv2"
        output_filename2 = f"benchmark_results_{test2_exp_desc}_{test2_model}.json"
        benchmark_dict = set_up_benchmark_dict(self.test_save_path, test2_exp_desc, test2_model, output_filename2)
        self.assertIsInstance(benchmark_dict, dict)
        self.assertEqual(benchmark_dict["metadata"]["experiment_desc"], test2_exp_desc)
        self.assertEqual(benchmark_dict["metadata"]["model_name"], test2_model)

    def test_load_benchmark_dict(self):
        """Tests that a benchmark dict is created if one doesn't exists, or loaded if one exists"""
        
        #Benchmark exists
        test1_exp_desc = "clip"
        test1_model = "clip"
        output_filename1 = f"benchmark_results_{test1_exp_desc}_{test1_model}.json"
        captured_output = io.StringIO()
        sys.stdout = captured_output
        exists_expected_output = f"Benchmark exists for exp: {test1_exp_desc} and model: {test1_model} and has been loaded\n"
        benchmark_dict = load_benchmark_dict(self.test_save_path, test1_exp_desc, test1_model, output_file_name=output_filename1)
        sys.stdout = sys.__stdout__
        self.assertEqual(captured_output.getvalue().lower(), exists_expected_output.lower())
        