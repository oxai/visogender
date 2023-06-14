"""
This file is used to set up any CLIP-like models that will be evaluated

Author: @abrantesfg. @hanwenzhu

"""

import clip
import torch 
torch.cuda.empty_cache()

from src.data_utils import get_image

device = "cuda" if torch.cuda.is_available() else "cpu"

def clip_set_up_model_processor():
    model, processor = clip.load("ViT-B/32", device)
    return model, processor

def clip_model(phrase_list, url_list, model, processor):
    """"
    Returns the logits of the key phrases for "his", "her" and "their". The order of the list is preserved, and no reordering is done

    Args:
        phrase_list: list of sentences which include both occupation and participant. The list is ordered as [male_sentence, female_sentence, neutral_sentence]
        image_url: url to the image to be captioned
        model: clip model
        processor: clip model processor

    Returns:
        list(tensors): logits for "his", "her" and "their". The order is used in subsequent code as well. 
    """
    
    image_inputs = [processor(get_image(url)).unsqueeze(0).to(device) for url in url_list]
    text_inputs = torch.cat([clip.tokenize(f"{c}") for c in phrase_list]).to(device)

    # Calculate features
    with torch.no_grad():
        image_features = torch.cat([model.encode_image(image_input) for image_input in image_inputs], dim=0)
        text_features = model.encode_text(text_inputs)

    # Rank results according to most similar labels for the image
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (image_features @ text_features.T).flatten().softmax(-1)
    similarity_list = similarity.tolist()

    return similarity_list 




