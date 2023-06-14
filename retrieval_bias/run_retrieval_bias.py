"""
This file runs the evaluation on the CLIP-like models for retrieval bias. The models are input as a list in /retrieval_bias/retrieval_bias_input_params.py. 
This outputs the raw logits from the VISOGENDER data and outputs the json according to this naming convention:
output_file_name = f"{experiment_name}_{model_name}"

Author: @hanwenzhu

"""
import sys
import PIL
from tqdm import tqdm

from retrieval_bias_input_params import retrieval_input_params, main_dir

sys.path.append(main_dir)

from src.clip_set_up import clip_set_up_model_processor, clip_model
from src.data_utils import load_visogender_data, save_dict_json
from src.template_generator_utils import set_up_parameters, load_metadata_to_dict, occupation_template_sentences_all_pronouns


experiment_name, bias_experiments, gender_idx_dict = set_up_parameters(retrieval_input_params)

# Do Occupation-Participant (OP) and Occupation-Object (OO) experiments
for context_args in [(True, False), (False, True)]:
    context_OP, context_OO = context_args

    if context_OP:
        sentence_path, template_occ_first, template_par_first = load_visogender_data(
            retrieval_input_params, context_OP, context_OO)
        templates = load_metadata_to_dict(sentence_path, "OP")
        template_type_list = retrieval_input_params["template_type"]
    elif context_OO:
        sentence_path, template_sentence_obj, _ = load_visogender_data(
            retrieval_input_params, context_OP, context_OO)
        templates = load_metadata_to_dict(sentence_path, "OO")
        template_type_list = [retrieval_input_params["template_type"][0]]

    for model_name in retrieval_input_params['clip_models']:

        print(f"Experiment name: {experiment_name}, Bias experiment: {bias_experiments}, Model name: {model_name}, Context OP: {context_OP}, Context OO: {context_OO}")

        if model_name == 'clip':
            model, processor = clip_set_up_model_processor()
        else:
            raise NotImplementedError

        other_obj = None
        other_participant = None
        for template_type in template_type_list:

            print(f"Template type: {template_type}")

            results_dict = {}

            for IDX_dict in templates:

                occupations = {IDX_dict[key]['occ'] for key in IDX_dict}

                for occupation in tqdm(occupations):

                    occupation_keys = [
                        key for key, data in IDX_dict.items()
                        if data['occ'] == occupation and data['url'] != '' and data['url'] != 'NA']
                    occupation_urls = [IDX_dict[key]['url']
                                    for key in occupation_keys]
                    occ_genders = [IDX_dict[key]['occ_gender']
                                for key in occupation_keys]

                    if not occupation_keys:
                        continue
                    sample = IDX_dict[occupation_keys[0]]

                    if context_OP:
                        other_participant = sample['par']
                        part_genders = [IDX_dict[key]['par_gender']
                                        for key in occupation_keys]
                        genders = list(zip(occ_genders, part_genders))
                        if template_type == "occ_first":
                            sentence_template = template_occ_first
                        elif template_type == "par_first":
                            sentence_template = template_par_first
                        _, _, neutral_sent = occupation_template_sentences_all_pronouns(
                            occupation, sentence_template,
                            other_participant=other_participant, model_domain="CLIP",
                            context_op=context_OP, context_oo=context_OO)
                    elif context_OO:
                        other_obj = sample['obj']
                        genders = occ_genders
                        if template_type == "occ_first":
                            sentence_template = template_sentence_obj
                        elif template_type == "par_first":
                            continue
                        _, _, neutral_sent = occupation_template_sentences_all_pronouns(
                            occupation, sentence_template,
                            other_object=other_obj, model_domain="CLIP",
                            context_op=context_OP, context_oo=context_OO)


                    results_dict[occupation] = {}
                    results_dict[occupation]["sector"] = sample['sector']
                    results_dict[occupation]["specialisation"] = sample['specialisation']
                    results_dict[occupation]["occ"] = occupation
                    # we measure bias only for the occupation genders, not participant genders
                    results_dict[occupation]["occ_genders"] = occ_genders
                    if context_OP:
                        results_dict[occupation]["par"] = other_participant
                    else:
                        results_dict[occupation]["obj"] = other_obj

                    try:
                        logits_list = clip_model(
                            [neutral_sent], occupation_urls, model, processor)
                    except PIL.UnidentifiedImageError:
                        results_dict[occupation]["error"] = True
                        continue

                    results_dict[occupation]["error"] = False
                    results_dict[occupation]["logits_list"] = logits_list

            output_file_name = f"{experiment_name}_{model_name}_{template_type}"
            save_dict_json(results_dict, context_OP=context_OP, context_OO=context_OO,
                        filepath=retrieval_input_params['result_savepath'],
                        exp_description=output_file_name)
