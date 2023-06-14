"""
This file runs the evaluation on the captioning models for the resolution bias. The models are input as a list in /resolution_bias/captioning_run/caption_input_params.py. 
This outputs the raw logits from the VISOGENDER data and outputs the json according to this naming convention:
output_file_name = f"{experiment_name}_{model_name}"
These are saved in /results/model_outputs/

Author: @smhall97 
"""
import sys
import PIL
from tqdm import tqdm

from caption_input_params import caption_input_params, main_dir

sys.path.append(main_dir) 

from src.template_generator_utils import set_up_parameters, load_metadata_to_dict, occupation_template_sentences_all_pronouns, participant_template_sentences_all_pronouns
from src.data_utils import load_visogender_data, save_dict_json
from src.captioning_set_up import blip_get_probabilities_his_her_their, blip_setup_model_processor, blipv2_set_up_model_processor

experiment_name, bias_experiments, gender_idx_dict = set_up_parameters(caption_input_params)

# Do Occupation-Participant (OP) and Occupation-Object (OO) experiments
for context_args in [(True, False), (False, True)]:
    context_OP, context_OO = context_args

    for model_name in caption_input_params["caption_models"]:

        print(f"Experiment name: {experiment_name}, Bias experiment: {bias_experiments}, Model name: {model_name}, Context OP: {context_OP}, Context OO: {context_OO}")
        
        if model_name == "blip":
            model, processor = blip_setup_model_processor()

        elif model_name == "blipv2":
            model, processor = blipv2_set_up_model_processor()
            model = model.float()
            model.float()

        if context_OP:
            context = "context_OP"
            sentence_path, template_occ_first, template_par_first = load_visogender_data(caption_input_params, context_OP, context_OO)
            metadata_dict = load_metadata_to_dict(sentence_path, "OP")
            template_type_list = caption_input_params["template_type"]
        elif context_OO:
            context = "context_OO"
            sentence_path, template_sentence_obj, _ = load_visogender_data(caption_input_params, context_OP, context_OO)
            metadata_dict = load_metadata_to_dict(sentence_path, "OO")
            template_type_list = [caption_input_params["template_type"][0]]


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
                    license = IDX_dict[metadata_key]["licence"]
                    occ_gender = IDX_dict[metadata_key]["occ_gender"]

                    if occ_gender == "neutral":
                        print(metadata_key)
                        break
                    
                    if context_OP:
                        other_participant = IDX_dict[metadata_key]["par"]
                        par_gender = IDX_dict[metadata_key]["par_gender"]
                        if template_type == "occ_first":
                            sentence_template = template_occ_first
                            _, _, neutral_sent = occupation_template_sentences_all_pronouns(
                                occupation, sentence_template, other_participant, other_obj, model_domain="CAPTIONING", context_op=context_OP, context_oo=context_OO)


                        elif template_type == "par_first":
                            sentence_template = template_par_first
                            _, _, neutral_sent = participant_template_sentences_all_pronouns(
                            other_participant, sentence_template)
                        
                    elif context_OO:
                        other_obj = IDX_dict[metadata_key]["obj"]
                        if template_type == "occ_first":
                            sentence_template = template_sentence_obj
                            _, _, neutral_sent = occupation_template_sentences_all_pronouns(
                                occupation, sentence_template, other_participant, other_obj, model_domain="CAPTIONING", context_op=context_OP, context_oo=context_OO)

                        elif template_type == "par_first":
                            continue


                    try:
                        logits_list = blip_get_probabilities_his_her_their(url, neutral_sent, model, processor)
                    
                    except PIL.UnidentifiedImageError:
                        print(f"Image failed to load {IDX_dict[metadata_key]['occ']}")
                        continue

                    if context_OP:

                        if template_type == "occ_first":
                            logits_list_occ_first = logits_list
                            results_dict[f"{metadata_key}"] = {"sector": IDX_dict[metadata_key]["sector"],
                                                            "specialisation": IDX_dict[metadata_key]["specialisation"],
                                                            "occ": occupation,
                                                            "occ_gender": occ_gender,
                                                            "par": other_participant,

                                                            "par_gender": par_gender}

                            
                            results_dict[f"{metadata_key}"]["logits_list_occ_first"] = logits_list_occ_first

                        elif template_type == "par_first":
                            logits_list_par_first = logits_list
                            results_dict[f"{metadata_key}"]["logits_list_par_first"] = logits_list_par_first
                            results_dict[f"{metadata_key}"]["experiment"] = "CAPTIONING"
                            results_dict[f"{metadata_key}"]["model_name"] = model_name
                            results_dict[f"{metadata_key}"]["context"] = context

                    elif context_OO:
                        logits_list_obj = logits_list

                        results_dict[f"{metadata_key}"] = {"sector": IDX_dict[metadata_key]["sector"],
                                                        "specialisation": IDX_dict[metadata_key]["specialisation"],
                                                        "occ": occupation,
                                                        "occ_gender": occ_gender,
                                                        "obj": other_obj,
                                                        "logits_list_obj" : logits_list_obj, 
                                                        "experiment": "CAPTIONING", 
                                                        "model_name": f"{model_name}",
                                                        "context": f"{context}"}
                        
        if context_OP:
            output_file_name = f"{experiment_name}_{model_name}"
        else:
            output_file_name = f"{experiment_name}_{model_name}"
        save_dict_json(results_dict, context_OP, context_OO, filepath=caption_input_params["result_savepath"], exp_description=output_file_name)
    

            
