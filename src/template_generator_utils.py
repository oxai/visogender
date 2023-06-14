"""
These functions are used for setting up the templates used in the VISOGENDER resolution and retrieval tasks.

Author: @smhall97, @abrantesfg

"""

import json


def set_up_parameters(parameter_dict):
    """
    Receives a dictionary with the setup parameters and return the values of them individualy.

    Args: parameter dictionary

    Returns: values of dictionary keys
    """
    experiment_name = parameter_dict["experiment_name"]
    bias_experiments = parameter_dict["bias_experiments"]
    gender_idx_dict = parameter_dict["gender_idx_dict"]

    return experiment_name, bias_experiments, gender_idx_dict


def load_input_parameters_dict(filepath: str):
    """
    Loads the input parameters which are set in the json file

    Args:
        filepath: filepath to the f"{caption/clip}_input_parameters.json

    Returns:
        
    """
    with open(filepath) as f:
        config = json.load(f)

def load_metadata_to_dict(filepath: str, context: str):
    """
    Opens a file, creates a dictionary of dictionaries with IDX as key, 
    and metadata as values.

    Args:
        filepath: filepath to saved .tsv
        context: the type of image scenario - either occupation/participant (OP) or occupation/object(OO

    Returns:
        Tuple with two dictionary (for each context) of dictionaries with metadata for all images IDX
    """
    op_idx_metadata_dict, oo_idx_metadata_dict = {}, {}

    with open(filepath, "r") as file:
        for enum, line in enumerate(file):
            try:
                if enum >= 1:
                    values = line.strip().split("\t")
           
                    idx = values[0]
                    sector = values[1]
                    specialisation = values[2]
                    occ = values[3]
                    url = values[5]
                    licence = bool(values[6])
                    occ_gender = values[7]
                    error_code = values[-2]
                    annotator = values[-1]

                    if context == "OP":
                        par = values[4]
                        par_gender = values[8]
                        op_idx_metadata_dict[idx] = {
                                                    "sector": sector, 
                                                    "specialisation": specialisation, 
                                                    "occ" : occ, 
                                                    "par" : par, 
                                                    "url" : url, 
                                                    "licence" : licence,
                                                    "occ_gender" : occ_gender, 
                                                    "par_gender" : par_gender, 
                                                    "annotator": annotator
                                                    }
                    
                    else:
                        obj = values[4]
                        oo_idx_metadata_dict[idx] = {
                                                    "sector": sector, 
                                                    "specialisation": specialisation, 
                                                    "occ" : occ, 
                                                    "obj" : obj, 
                                                    "url" : url, 
                                                    "licence" : licence, 
                                                    "occ_gender" : occ_gender,
                                                    "annotator": annotator
                                                    }
            except IndexError:
                continue
        
    return op_idx_metadata_dict, oo_idx_metadata_dict

def occupation_template_sentences_all_pronouns(occupation: str, template_sentence:str, other_participant:str=None, other_object:str=None, model_domain: str="ClIP", context_op:bool=False, context_oo:bool=False)-> tuple:
    """
    Creates three sentences, based off either an occupation (if CLIP: and participant), or occupation If CLIP:and object)
    
    Args:
        occupation: name of occupation
        template_sentence: input sentence in format similar to: f"A $OCCUPATION and $POSS_PRONOUN $PARTICIPANT" or f"A $OCCUPATION and " (captioning)
        context: the type of image scenario - either occupation/participant (OP) or occupation/object(OO)
        other_participant: name of participant based on scene and context. Default None, for the captioning case
        other_object: name of object, dependent on scene and context. Default none, for the captioning case
        model_domain: either clip-like ("CLIP") or captioning ("CAPTIONING")


    Returns:
        tuple with three string sentences, each sentence containing the corresponding male/ female / neutral 
        pronoun. If the experiment is angle 1 - all three are required, if angle 2 - the first two sentences
        in the tuple can be ignored
    """
  
    sentence_components = template_sentence.split(" ")
    occ_index = sentence_components.index("$OCCUPATION")

    if other_participant is not None or other_object is not None:
        if model_domain  == "CLIP":
            if context_op:
                part_index = sentence_components.index("$PARTICIPANT")
                sentence_components[part_index] = other_participant
            elif context_oo:
                obj_index = sentence_components.index("$OBJECT")
                sentence_components[obj_index] = other_object
            
    sentence_components[occ_index] = occupation

    NOM = "$NOM_PRONOUN"
    POSS = "$POSS_PRONOUN"
    ACC = "$ACC_PRONOUN"
    special_pronouns = set({NOM, POSS, ACC})
    female_map = {NOM: "she", POSS: "her", ACC: "her"}
    male_map = {NOM: "he", POSS: "his", ACC: "him"}
    neutral_map = {NOM: "they", POSS: "their", ACC: "them"}

    female_pronouns = [x if not x in special_pronouns else female_map[x] for x in sentence_components]
    male_pronouns = [x if not x in special_pronouns else male_map[x] for x in sentence_components]
    neutral_pronouns = [x if not x in special_pronouns else neutral_map[x] for x in sentence_components]

    male_sentence, female_sentence, neutral_sentence = " ".join(male_pronouns), " ".join(female_pronouns), " ".join(neutral_pronouns)

  
    return male_sentence, female_sentence, neutral_sentence

def participant_template_sentences_all_pronouns(other_participant:str, template_sentence:str)-> tuple:
    """
    Creates three sentences, with the template reversed - only the participant - occupation is considered in this case
    If CLIP: "A $PARTICIPANT and $PRONOUN $OCCUPATION"
    If CAPTIONING: "A $PARTICIPANT and"
    
    Args:
        other_participant: name of participant based on scene and context
        template_sentence: input sentence in format similar to: f"A $PARTICIPANT and $POSS_PRONOUN $OCCUPATION" or f"A $PARTICIPANT and " (captioning)


    Returns:
        tuple with three string sentences, each sentence containing the corresponding male/ female / neutral 
        pronoun. If the experiment is angle 1 - all three are required, if angle 2 - the first two sentences
        in the tuple can be ignored
    """
  
  
    sentence_components = template_sentence.split(" ")
    part_index = sentence_components.index("$PARTICIPANT")
    sentence_components[part_index] = other_participant

    NOM = "$NOM_PRONOUN"
    POSS = "$POSS_PRONOUN"
    ACC = "$ACC_PRONOUN"
    special_pronouns = set({NOM, POSS, ACC})
    female_map = {NOM: "she", POSS: "her", ACC: "her"}
    male_map = {NOM: "he", POSS: "his", ACC: "him"}
    neutral_map = {NOM: "they", POSS: "their", ACC: "them"}

    female_pronouns = [x if not x in special_pronouns else female_map[x] for x in sentence_components]
    male_pronouns = [x if not x in special_pronouns else male_map[x] for x in sentence_components]
    neutral_pronouns = [x if not x in special_pronouns else neutral_map[x] for x in sentence_components]

    male_sentence, female_sentence, neutral_sentence = " ".join(male_pronouns), " ".join(female_pronouns), " ".join(neutral_pronouns)

    return male_sentence, female_sentence, neutral_sentence