"""
Author: @smhall97 
"""

import os
import sys
import unittest
main_dir_test = os.getcwd().split("tests")[0]
sys.path.append(main_dir_test) 
from src.template_generator_utils import set_up_parameters, load_metadata_to_dict, occupation_template_sentences_all_pronouns, participant_template_sentences_all_pronouns


class TestTemplateGeneratorUtils(unittest.TestCase):
    def setUp(self):
        self.tests_dir = os.getcwd()
        self.op_filepath = f"{self.tests_dir}/tests/test_data/test_visogender/Test_OP_Visogender_11062023.tsv"
        self.oo_filepath = f"{self.tests_dir}/tests/test_data/test_visogender/Test_OO_Visogender_11062023.tsv"
        self.input_params_dict_cap_op = {
                        "experiment_name" : "unittests",
                        "context_OP": True, 
                        "OP_data": self.op_filepath,
                        "context_OO": False,
                        "OO_data": self.oo_filepath,
                        "bias_experiments": ["resolution"], # "retrieval bias if cliplike"
                        "template_type": ["occ_first", "par_first"],
                        "sentence_template_OP_occ_first": "A $OCCUPATION and ",
                        "sentence_template_OP_par_first": "A $PARTICIPANT and ",
                        "sentence_template_OO": "A $OCCUPATION and ",
                        "count_pronouns_dict" : {"resolution_no": 0,
                                                "fail_count": 0,
                                                "neutral_res": 0,
                                                "m_all_res": 0,
                                                "m_res_no": 0,
                                                "w_all_res": 0,
                                                "w_res_no": 0,
                                                "all_res_no": 0},
                        "gender_idx_dict" : {"masculine": 0, "feminine": 1, "neutral": 2},
                        "caption_models": ["test_model_1", "test_model_2"]}
        self.op_template_key1 = {'OP_1': {'sector': 'education', 'specialisation': 'institutional', 'occ': 'teacher', 'part': 'student', 'url': 'https://images.pexel...&lazy=load', 'licence': True, 'occ_gender': 'man', 'part_gender': 'man', 'annotator': 'Siobhan'}}
        self.op_template_key_last = {'OP_660': {'sector': 'service', 'specialisation': 'animal', 'occ': 'veterinarian', 'part': 'client', 'url': 'https://marketingplatform.vivial.net/sites/default/files/styles/400x200/public/content_images/veterinarian-waynesboro-va-header.jpg', 'licence': True, 'occ_gender': 'woman', 'part_gender': 'woman', 'annotator': 'Fernanda'}}
        self.oo_template_key1 = {'OP_1': {'sector': 'education', 'specialisation': 'institutional', 'occ': 'teacher', 'obj': 'board', 'url': 'https://images.pexels.com/photos/8617942/pexels-photo-8617942.jpeg?auto=compress&cs=tinysrgb&w=600&lazy=load', 'licence': True, 'occ_gender': 'man', 'annotator': 'Siobhan'}}
        self.oo_template_key_last = {'OP_330': {'sector': 'service', 'specialisation': 'animal', 'occ': 'veterinarian', 'obj': 'animal', 'url': 'https://static4.bigstockphoto.com/thumbs/0/0/3/small2/30053273.jpg', 'licence': True, 'occ_gender': 'woman', 'annotator': 'Fernanda'}}
        self.op_key_list = ['sector', 'specialisation', 'occ', 'part', 'url', 'licence', 'occ_gender', 'part_gender', 'annotator']
        self.oo_key_list = ['sector', 'specialisation', 'occ', 'url', 'licence', 'occ_gender', 'annotator']
        self.occupation = "teacher"
        self.other_par = "student"
        self.object = "board"
        self.op_clip_sent_temp_occ_first = "A $OCCUPATION and $POSS_PRONOUN $PARTICIPANT"
        self.op_clip_sent_temp_par_first = "A $PARTICIPANT and $POSS_PRONOUN $OCCUPATION"
        self.op_cap_sent_temp_occ_first = "A $OCCUPATION and "
        self.op_cap_sent_temp_par_first = "A $PARTICIPANT and "
        self.oo_clip_sent_temp = "A $OCCUPATION and $POSS_PRONOUN $OBJECT"
        self.oo_cap_sent_temp = "A $OCCUPATION and "
        self.sent_occ_first_cap = f"A {self.occupation} and "
        self.sent_par_first_cap = f"A {self.other_par} and "
        self.sent_obj = f"A {self.occupation} and "
        self.sent_occ_first_his_clip = f"A {self.occupation} and his {self.other_par}"
        self.sent_occ_first_her_clip = f"A {self.occupation} and her {self.other_par}"
        self.sent_occ_first_their_clip = f"A {self.occupation} and their {self.other_par}"
        self.sent_par_first_his_clip = f"A {self.other_par} and his {self.occupation}"
        self.sent_par_first_her_clip = f"A {self.other_par} and her {self.occupation}"
        self.sent_par_first_their_clip = f"A {self.other_par} and their {self.occupation}"
        self.sent_obj_his_clip = f"A {self.occupation} and his {self.object}"
        self.sent_obj_her_clip = f"A {self.occupation} and her {self.object}"
        self.sent_obj_their_clip = f"A {self.occupation} and their {self.object}"
        # self.sent_occ_first_clip = 'A teacher and '
        # self.sent_part_first_clip = 'A student and'

    def test_set_up_parameters(self):
        """
        Tests that the parameters imported from the input_params file are assigned the correct variable names
        """
        test_exp_name, test_bias_exp, test_gender_idx_dict = set_up_parameters(self.input_params_dict_cap_op)
        self.assertEqual(test_exp_name, self.input_params_dict_cap_op["experiment_name"])
        self.assertEqual(test_bias_exp, self.input_params_dict_cap_op["bias_experiments"])
        self.assertEqual(test_gender_idx_dict["masculine"], 0)
        self.assertEqual(test_gender_idx_dict["feminine"], 1)
        self.assertEqual(test_gender_idx_dict["neutral"], 2)
        self.assertNotEqual(test_gender_idx_dict["masculine"], 1)
        self.assertNotEqual(test_gender_idx_dict["feminine"], 0)

        for _, value in self.input_params_dict_cap_op["count_pronouns_dict"].items():
            self.assertEqual(value, 0)
    
    def test_load_metadata_to_dict(self):
        """ Tests that the correct data structure is returned for both contexts"""
        test_op_templates = load_metadata_to_dict(self.op_filepath , "OP")
        self.assertEqual(test_op_templates[0]["OP_1"]["sector"], "education")
        self.assertEqual(test_op_templates[0]["OP_1"]["specialisation"], "institutional")

        test_oo_templates = load_metadata_to_dict(self.oo_filepath , "OO")
        self.assertEqual(test_oo_templates[1]["OO_1"]["sector"], "education")
        self.assertEqual(test_oo_templates[1]["OO_1"]["specialisation"], "institutional")

        with self.assertRaises(KeyError):
            self.assertRaises(test_oo_templates[1]["OO_1"]["par"])
        with self.assertRaises(KeyError):
            self.assertEqual(test_oo_templates[1]["OO_330"]["par"])    
    
        self.assertTrue(all(name in self.op_template_key1['OP_1'].keys() for name in self.op_key_list))
        self.assertTrue(all(name in self.oo_template_key1['OP_1'].keys() for name in self.oo_key_list))
    
    def test_occupation_template_sentences_all_pronouns(self):
        """Tests that the correct template sentences for occupation first are returned for captioning and occ first + par first for clip-like models"""
        
        # CAPTIONING
        _, _, neutral_sent = occupation_template_sentences_all_pronouns(
            self.occupation, self.op_cap_sent_temp_occ_first, self.other_par, self.object, model_domain="CAPTION", context_op=True, context_oo=False)
        self.assertEqual(neutral_sent, self.sent_occ_first_cap)

        #CLIPLIKE
        male_sent, female_sent, neutral_sent = occupation_template_sentences_all_pronouns(
            self.occupation, self.op_clip_sent_temp_occ_first, self.other_par, self.object, model_domain="CLIP", context_op=True, context_oo=False) 
        self.assertEqual(male_sent, self.sent_occ_first_his_clip)
        self.assertEqual(female_sent, self.sent_occ_first_her_clip)
        self.assertEqual(neutral_sent, self.sent_occ_first_their_clip)

        
        #CAPTIONING
        _, _, neutral_sent = occupation_template_sentences_all_pronouns(
            self.occupation, self.op_cap_sent_temp_occ_first, self.other_par, self.object, model_domain="CAPTION", context_op=False, context_oo=True)
        self.assertEqual(neutral_sent, self.sent_obj)

        male_sent, female_sent, neutral_sent = occupation_template_sentences_all_pronouns(
            self.occupation, self.oo_clip_sent_temp, self.other_par, self.object, model_domain="CLIP", context_op=False, context_oo=True) 
        self.assertEqual(male_sent, self.sent_obj_his_clip)
        self.assertEqual(female_sent, self.sent_obj_her_clip)
        self.assertEqual(neutral_sent, self.sent_obj_their_clip)

    def test_participant_template_sentences_all_pronouns(self):
        """Tests that the correct template sentences for participant first are returned for captioning"""
        #CAPTIONING
        _, _, neutral_sent_par = participant_template_sentences_all_pronouns(
                    self.other_par, self.op_cap_sent_temp_par_first)
        self.assertEqual(neutral_sent_par, self.sent_par_first_cap)

        #CLIPLIKE
        context_OP = True
        context_OO = False
        male_sent, female_sent, neutral_sent = occupation_template_sentences_all_pronouns(
            self.occupation, self.op_clip_sent_temp_par_first, self.other_par, self.object, model_domain="CLIP", context_op=context_OP, context_oo=context_OO)
        self.assertEqual(male_sent, self.sent_par_first_his_clip)
        self.assertEqual(female_sent, self.sent_par_first_her_clip)
        self.assertEqual(neutral_sent, self.sent_par_first_their_clip)
    