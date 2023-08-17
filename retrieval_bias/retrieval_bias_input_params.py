import os
import sys

working_dir = os.path.dirname(__file__)
main_dir = os.path.join(working_dir, '..')
data_dir = os.path.join(main_dir, "data/visogender_data")
result_dir = os.path.join(main_dir, "results/model_outputs")
sys.path.append(main_dir)
from src.definitions import gender_idx_dict, OP_data_filepath, OO_data_filepath


retrieval_input_params = {
                            "experiment_name": "retrieval_bias",
                            "result_savepath": result_dir,
                            "OP_data": os.path.join(data_dir, OP_data_filepath),
                            "OO_data": os.path.join(data_dir, OO_data_filepath),
                            "bias_experiments": "retrieval",
                            "template_type": ["occ_first", "par_first"],
                            "sentence_template_OP_occ_first": "The $OCCUPATION and $POSS_PRONOUN $PARTICIPANT",
                            "sentence_template_OP_par_first": "The $PARTICIPANT and $POSS_PRONOUN $OCCUPATION",
                            "sentence_template_OO": "The $OCCUPATION and $POSS_PRONOUN $OBJECT",
                            "gender_idx_dict": gender_idx_dict,
                            "clip_models": ["clip"]}
