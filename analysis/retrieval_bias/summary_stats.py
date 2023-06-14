"""
This file runs analysis for the retrieval bias metric scores for both single person and two-person instances (context OO and OP)

Author: @hanwenzhu
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

main_dir = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, main_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'analysis_json', type=argparse.FileType('r'), help='JSON analysis results from retrieval_analysis.py')

    args = parser.parse_args()
    bias = json.load(args.analysis_json)

    metrics = ['bias@5', 'bias@10', 'maxskew@5', 'maxskew@10', 'ndkl', 'bias_count@10']
    means = []
    stds = []
    for metric in metrics:
        stats = [bias[o][metric] for o in bias]
        mean = np.mean(stats)
        # ddof=1 to match with pandas std dev
        std = np.std(stats, ddof=1)
        means.append(mean)
        stds.append(std)
    df = pd.DataFrame([means, stds], columns=metrics, index=['Mean', 'SD'])
    print(df)

if __name__ == '__main__':
    main()
