# VISOGENDER: A dataset for benchmarking gender bias in image-text pronoun resolution

Authors: [Siobhan Mackenzie Hall](https://github.com/smhall97), [Fernanda Gonçalves Abrantes](https://github.com/abrantesfg), [Hanwen Zhu](https://github.com/hanwenzhu), [Grace Sodunke](https://github.com/grace-sodunke), [Aleksandar Shtedritski](https://github.com/suny-sht) and [Hannah Rose Kirk](https://github.com/HannahKirk)

![Visogender splash figure](/visogender_splash.jpeg)
**VISOGENDER** is a benchmark dataset used to assess gender pronoun resolution bias in the domain of occupation for vision-language models. **VISOGENDER** is designed to support two types of VLMs (CLIP-like and Captioning) in evaluating bias in resolution and retrieval tasks. VISOGENDER works within a hegemonic system
of binary and stereotypical gender presentation that remains prevalent in Western constructions and perceptions of gender.

## Paper
Our paper can be found in the [link here](https://arxiv.org/abs/2306.12424).

## The Dataset
The **VISOGENDER** dataset comprises image URLs with annotations for the occupation noun, the participant or object noun, and the inferred groundtruth gender of the occupation and participant. These annotations can be used to reconstruct the templated captions. Data collection was carried out by the authors of the paper from March to May 2023 on a variety of image databases and search providers, such as Pexels and Google Image Search.

The data is available here: `data/visogender_data` and is in a .tsv format.

There are two sets of image datasets:

- Single person: 
  - 230 images, divided into 23 occupations 
  - [Link to dataset named OO_Visogender_15082023.tsv](data/visogender_data/OO/OO_Visogender_15082023.tsv) 
- Two-persons: 
  - 460 images, divided into 23 occupations. Same pairs: M-M ; F-F | Diff. pairs: M-F ; F-M
  - [Link to dataset named OP_Visogender_15082023.tsv](data/visogender_data/OP/OO_Visogender_15082023.tsv)

The dataset is made up of the following metadata headings: Sector, Specialisation and Occupation tags; URL; confirmation of licence; labels assigned to `occupation` as well as the `object` (single person) or the `participant` (two-person). The person responsible for collecting each instance is also indicated.

*Date collected:* this data was collected between March and May 2023. 

*Data collection:* The data was collected trhough a variety of image databases and search providers, such as Pexels and Google Image Search

### Opting out, removing images and/or adjusting labels
We are continually looking to improve VISOGENDER. If for any reason you would like an image removed, or labels updated please complete this [Google Form](https://forms.gle/uD7tQfSa7jvzoqDU6) which automatically notifies the authors when an entry is submitted. Reasons for removing an image could include, but are not limited to that the link is broken, it is redirecting to inappropriate / inapplicable content and/or you have identified yourself in the image and would like it taken down. Reasons for updating the labels can include, but are not limited to, uou have identified an image of yourself, or someone you know and the assigned label is incorrect, and/or inappropriate, but the image can remain in the dataset.

**[Link to Google Form](https://forms.gle/uD7tQfSa7jvzoqDU6)**

### Checking the data integrity

If you would like to check that all URLs in the dataset are working, and/or check that there are no duplicates, please run the following code:

```sh
python3 data/visogender_url_integrity_check.py
```

For the full maintanance plan, please review the [LICENCE](/LICENCE)

## The VISOGENDER setup

The **VISOGENDER** setup has the flexibility to measure VLM (CLIP-like and Captioning models) bias in two ways:

1. *Resolution bias*: The resolution task considers a single image with perceived gender presentation and matches it to multiple candidate captions containing different gender pronouns. For example, we start with an image containing a female doctor, and specify the set of candidate captions as “the doctor and her/his patient. We define a resolution accuracy and gender gap score in the paper.

2. *Retrieval bias*:  The retrieval task considers a single gender neutral caption for a given occupation and matches it to multiple images containing subjects with different perceived gender presentations from the same occupation. For example, we start with the caption “the doctor and their patient” and define the set of candidate images as containing 50% images of doctors who are men and 50% who are women. Given there is no groundtruth for a “correct” ranking of images for a gender-neutral caption, we cannot define a retrieval accuracy metric. For defining retrieval bias, we use 3 commonly used metrics – Bias@K, Skew@K and NDKL (see the paper for details)

The code base is set up to run the benchmark, finegrained analysis and the comparison to the US Labor Force Statistics. Details to run these analyses are given below:

=======

To run this data, make sure you have python3 installed. 
You can install dependencies using: 

```sh
pip install -r requirements.txt
```
=========
## The VISOGENDER Benchmark
There following code can be run to return the benchmark scores for resolution and retrieval bias.

### Computing raw results
#### Resolution bias
For CLIP-like models:
```sh
python3 resolution_bias/cliplike_run/run_cliplike.py 
```
For captioning models:
```sh
python3 resolution_bias/captioning_run/run_captioning.py 
```
This runs for the models set up, and saves the raw results to `/results/model_outputs/` in a raw output JSON `<model-output>.json`. Results for both occupation-participant (OP) and occupation-object (OO) are saved.

#### Retrieval bias

```sh
python3 retrieval_bias/run_retrieval_bias.py 
```
This runs for the models set up, and saves the raw results to `/results/model_outputs/` in a raw output JSON `<model-output>.json`. Results for both occupation-participant (OP) and occupation-object (OO) are saved.

### Computing bias measures
To get benchmark scores, please run both `return_benchmark.py` files for resolution and retrieval bias:

Resolution bias:
```sh
cd analysis/resolution_bias; python3 return_benchmark.py 
```
Retrieval bias:
```sh
cd analysis/retrieval_bias; python3 return_benchmark.py 
```
This runs the benchmark analsysis and outputs to `/results/benchmark_scores/` in a raw output JSON `<benchmark>.json` for both resolution and retrieval bias.


#### What does a good result look like?

To perform well on **VISOGENDER**, the scores should be optimised as follows:

Resolution accuracy: this should be as close to 1.0 (100%) as possible which indicates a high capability in performing perceived gender coreference resolution 
Gender Gap: this should be as close to 0 as possible (i.e. the model performs equally well for both perceived presentations of genders, and isn't biased towards either perceived gender presentation)
Retrieval metrics: these should be as close to 0 as possible (i.e. perceived gender presentations in retrieval results are balanced to demonstrate a model is not biased towards either perceived gender presentation)

## Adding your own models
**VISOGENDER** supports two types of VLMs: CLIP-like models and Captioning models. You can evaluate your own models as follows:
- Add model set up to `src/cliplike_set_up.py` or `src/captioning_set_up.py`. You should provide a model and a processor object. 
- Pair each model with a string identifier and add this as a string to:
  - Resolution bias: 
  `resolution_bias/cliplike_run/cliplike_input_params.py`
  `resolution_bias/captioning_run/captioning_input_params.py`
  - Retrieval bias:
  `retrieval_bias/cliplike_input_params.py`
- For models with different implementation details to CLIP / BLIP, some custom changes might be needed: 
  - For CLIP-like models, the bechmark relies on similarity scores between an image and several text prompts. An example implemnetation of that is provided in  `src/clip_set_up/clip_model` that returns a list `[s1, s2, s3]` with similarities for each caption. If the implementation of your model and the computation of similarity scores differs significantly, a similar function will need to be implemented. 
  - For captioning models, the benchmark relies on the logits for the pronouns for "his" and "her" during next token prediction. An example implementation of this is provided in `src/captioning_set_up/blip_get_probabilities_his_her_their`. Due to the differences in implementatuion details, a similar function needs to be implmeneted for different captioning models. 

## Additional Analysis

The data can be used once loaded into a dictionary. The dictionary can be looped over to access each individual instance, and its metadata as needed. There are two sets of data:single person and two-person. The following script shows an example of how these can be loaded into dictionaries using tailored functions:
```
data/dataloader.py
```
### Resolution Bias Analysis
This analysis returns a dataframe with basic analysis. This analysis adds two columns to the dataframe with boolean values to indicate the following:
1. If the model's hightest probabilty matches the ground truth
2. If the model's highest probability returns the neutral pronoun

```sh
python3 analysis/resolution_bias/run_preliminary_analysis.py 
```
converts the raw output into analysis stats `<output-analysis>.json` for subsequent scripts and is saved in `results/resolution_bias_analysis/preliminary_analysis`:


### Retrieval Bias Analysis

#### Analysis script
```sh
python3 retrieval_analysis.py <model-output>.json <output-analysis>.json
```
converts the raw output into analysis stats `<output-analysis>.json` for subsequent scripts:

#### Summary stats
```sh
python3 summary_stats.py <output-analysis>.json
```
prints the headline statistics from the analysis (Bias@5, Bias@10, MaxSkew@5, MaxSkew@10, NDKL).

## Running the comparison to US Labor Force Statistics

We share the mapping we used to compare the resolution and retrieval bias scores with the US Labor Force Statistics. The data can be accessed in the follow file:

### Accessing the data:
```
cd data/US_Labor_Force_Statistics/US_Visogender_mapping_statistics_11062023.tsv
```

# Licence
Please see the associated licence in the [LICENCE file](/LICENCE). 

## Citation

```
@misc{hall2023visogender,
      title={VisoGender: A dataset for benchmarking gender bias in image-text pronoun resolution}, 
      author={Siobhan Mackenzie Hall and Fernanda Gonçalves Abrantes and Hanwen Zhu and Grace Sodunke and Aleksandar Shtedritski and Hannah Rose Kirk},
      year={2023},
      eprint={2306.12424},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
