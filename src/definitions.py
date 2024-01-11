"""
The gender mapping dictionary is saved in a single location for easier refactoring, and adjusting of the tyoes of pronouns used in the VISOGENDER task
"""

gender_idx_dict = {"masculine": 0, "feminine": 1, "neutral": 2}


"""
VISOGENDER DATA PATHS
"""

OP_data_filepath = "/data/visogender_data/OP/OP_Visogender_11012024.tsv"
OO_data_filepath = "/data/visogender_data/OO/OO_Visogender_02102023.tsv"
