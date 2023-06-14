"""
This file is used to set up any captioning models that will be evaluated

Author: @smhall97

"""
import torch 
torch.cuda.empty_cache()
from typing import List
from transformers import BlipProcessor, BlipForConditionalGeneration, Blip2Processor, Blip2Model

from src.data_utils import get_image


def blip_setup_model_processor():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda")

    return blip_model, processor

def blipv2_set_up_model_processor():
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    blipv2_model =  Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b").to("cuda")

    return blipv2_model, processor


def blip_get_probabilities_his_her_their(image_url: str, text_input: str, model, processor)->List:
    """"
    Returns the logits of the key phrases for "his", "her" and "their". The order of the list is preserved, and no reordering is done

    Args:
        image_url: url to the image to be captioned
        text_input: start sentence which includes the profession
        model: captioning model
        processor: captioning model processor

    Returns:
        list(tensors): logits for "his", "her" and "their". The order is used in subsequent code as well. 
    """
    raw_image = get_image(image_url)
    token_for_his = processor(raw_image, "his", return_tensors="pt").to("cuda")["input_ids"][0][1] 
    token_for_her = processor(raw_image, "her", return_tensors="pt").to("cuda")["input_ids"][0][1]
    token_for_their = processor(raw_image, "their", return_tensors="pt").to("cuda")["input_ids"][0][1] 

    inputs = processor(raw_image, text_input, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model(**inputs)
        if "logits" in outputs.keys():
            logits = outputs["logits"][0,-1,:]
        else:
            logits = outputs["decoder_logits"][0,-1,:]

    logits_for_his = logits[token_for_his] 
    logits_for_her = logits[token_for_her] 
    logits_for_their = logits[token_for_their] 

    return {"his": logits_for_his.item(), "her": logits_for_her.item(), "their": logits_for_their.item()}


def blip_conditional_image_captioning(image_url: str, text_input: str, model, processor):
    """
    Returns the caption as output by the BLIP model. The output is based on an input-starting sentence.
    The function prints the caption, and returns the encoded tokens

    Args:
        image_url: url to the image to be captioned
        text_input: start sentence which includes the profession

    Returns:
        tensor (not decoded) with tokens of the caption
    
    """
    raw_image = get_image(image_url)

    inputs = processor(raw_image, text_input, return_tensors="pt").to("cuda")

    output = model.generate(**inputs)
    # print("CONDITIONAL", processor.decode(output[0], skip_special_tokens=True))
    print("CONDITIONAL", processor.decode(output[0], skip_special_tokens=True))

    return output

