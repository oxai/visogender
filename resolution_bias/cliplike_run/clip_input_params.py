import os
import sys

main_dir = os.getcwd().split("resolution_bias")[0]
data_dir = os.path.join(main_dir, "data/visogender_data")
result_dir = os.path.join(main_dir, "results/model_outputs")
sys.path.append(main_dir) 

from src.definitions import gender_idx_dict


clip_input_params = {
                        "experiment_name" : "clip",
                        "result_savepath": result_dir,
                        "OP_data": os.path.join(data_dir, "OP/OP_Visogender_15082023.tsv"),
                        "OO_data": os.path.join(data_dir, "OO/OO_Visogender_15082023.tsv"),
                        "bias_experiments": "resolution",
                        "template_type": ["occ_first", "par_first"],
                        "sentence_template_OP_occ_first": "The $OCCUPATION and $POSS_PRONOUN $PARTICIPANT",
                        "sentence_template_OP_par_first": "The $PARTICIPANT and $POSS_PRONOUN $OCCUPATION",
                        "sentence_template_OO": "The $OCCUPATION and $POSS_PRONOUN $OBJECT",
                        "gender_idx_dict" : gender_idx_dict,
                        "clip_models": ["clip"]}
    
