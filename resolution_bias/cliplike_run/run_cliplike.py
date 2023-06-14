"""
This file runs the evaluation on the CLIP-like models for resolution bias. The models are input as a list in /resolution_bias/cliplike_run/clip_input_params.py. 
This outputs the raw logits from the VISOGENDER data and outputs the json according to this naming convention:
output_file_name = f"{experiment_name}_{model_name}"
These are saved in /results/model_outputs/

Author: @abrantesfg 
"""
import sys
import PIL
from tqdm import tqdm

from clip_input_params import clip_input_params, main_dir

sys.path.append(main_dir) 

from src.template_generator_utils import set_up_parameters, load_metadata_to_dict, occupation_template_sentences_all_pronouns
from src.data_utils import load_visogender_data, save_dict_json
from src.clip_set_up import clip_set_up_model_processor, clip_model

experiment_name, bias_experiments, gender_idx_dict = set_up_parameters(clip_input_params)

# Do Occupation-Participant (OP) and Occupation-Object (OO) experiments
for context_args in [(True, False), (False, True)]:
    context_OP, context_OO = context_args

    for model_name in clip_input_params["clip_models"]:

        print(f"Experiment name: {experiment_name}, Bias experiment: {bias_experiments}, Model name: {model_name}, Context OP: {context_OP}, Context OO: {context_OO}")

        if model_name == "clip":
            model, processor = clip_set_up_model_processor()

        if context_OP:
            context = "context_OP"
            sentence_path, template_occ_first, template_par_first = load_visogender_data(clip_input_params, context_OP, context_OO)
            metadata_dict = load_metadata_to_dict(sentence_path, "OP")
            template_type_list = clip_input_params["template_type"]
        elif context_OO:
            context = "context_OO"
            sentence_path, template_sentence_obj, _ = load_visogender_data(clip_input_params, context_OP, context_OO)
            metadata_dict = load_metadata_to_dict(sentence_path, "OO")
            template_type_list = [clip_input_params["template_type"][0]]

        other_obj = None
        other_participant = None
        results_dict = {}


        for template_type in template_type_list:

            print(f"Template type: {template_type}")

            for IDX_dict in metadata_dict:
                for metadata_key in tqdm(IDX_dict):

                    occupation = IDX_dict[metadata_key]["occ"]
                    url = IDX_dict[metadata_key]["url"]
                    if url is None or url == "" or url == "NA":
                        continue
                    licence = IDX_dict[metadata_key]["licence"]
                    occ_gender = IDX_dict[metadata_key]["occ_gender"]

                    if occ_gender == "neutral":
                        print(metadata_key)
                        break

                    if context_OP:
                        other_participant = IDX_dict[metadata_key]["par"]
                        par_gender = IDX_dict[metadata_key]["par_gender"]
                        if template_type == "occ_first":
                            sentence_template = template_occ_first
                            male_sent, female_sent, neutral_sent = occupation_template_sentences_all_pronouns(
                                occupation, sentence_template, other_participant, other_obj, model_domain="CLIP", context_op=context_OP, context_oo=context_OO)
                        
                        elif template_type == "par_first":
                            sentence_template = template_par_first
                            male_sent, female_sent, neutral_sent = occupation_template_sentences_all_pronouns(
                                occupation, sentence_template, other_participant, other_obj, model_domain="CLIP", context_op=context_OP, context_oo=context_OO)
                    
                    elif context_OO:
                        other_obj = IDX_dict[metadata_key]["obj"]
                        if template_type == "occ_first":
                            sentence_template = template_sentence_obj
                            male_sent, female_sent, neutral_sent = occupation_template_sentences_all_pronouns(
                                occupation, sentence_template, other_participant, other_obj, model_domain="CLIP", context_op=context_OP, context_oo=context_OO)
                            
                        elif template_type == "par_first":
                            continue

                    try:
                        logits_list = clip_model([male_sent, female_sent, neutral_sent], [url], model, processor)
                        logits_dict = {"his" : logits_list[0], "her": logits_list[1], "their": logits_list[2]}

                    except PIL.UnidentifiedImageError:
                        print(f"Image failed to load {IDX_dict[metadata_key]['occ']}")
                        continue

                    if context_OP:
                        
                        if template_type == "occ_first":
                            logits_list_occ_first = logits_dict
                            results_dict[f"{metadata_key}"] = {"sector": IDX_dict[metadata_key]["sector"],
                                                            "specialisation": IDX_dict[metadata_key]["specialisation"],
                                                            "occ": occupation,
                                                            "occ_gender": occ_gender,
                                                            "par": other_participant,
                                                            "par_gender": par_gender}


                            results_dict[f"{metadata_key}"]["logits_list_occ_first"] = logits_list_occ_first


                        elif template_type == "par_first":
                            logits_list_par_first = logits_dict
                            results_dict[f"{metadata_key}"]["logits_list_par_first"] = logits_list_par_first
                            results_dict[f"{metadata_key}"]["experiment"] = "CLIP"
                            results_dict[f"{metadata_key}"]["model_name"] = model_name
                            results_dict[f"{metadata_key}"]["context"] = context
                        
                    elif context_OO:
                        logits_list_obj = logits_dict
                        results_dict[f"{metadata_key}"] = {"sector": IDX_dict[metadata_key]["sector"],
                                                        "specialisation": IDX_dict[metadata_key]["specialisation"],
                                                        "occ": occupation,
                                                        "occ_gender": occ_gender,
                                                        "obj": other_obj,
                                                        "logits_list_obj" : logits_list_obj,
                                                        "experiment": "CLIP", 
                                                        "model_name": f"{model_name}",
                                                        "context": f"{context}"}

        if context_OP:
            output_file_name = f"{experiment_name}_{model_name}"
        else:
            output_file_name = f"{experiment_name}_{model_name}"
        save_dict_json(results_dict, context_OP, context_OO, filepath=clip_input_params["result_savepath"], exp_description=output_file_name)