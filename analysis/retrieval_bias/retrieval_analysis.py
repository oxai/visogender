"""
This file runs analysis for the retrieval bias metric scores 

Author: @hanwenzhu
"""

import argparse
import json
import os
import sys

main_dir = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, main_dir)

from src.metrics import calculate_retrieval_bias


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'results_json', type=argparse.FileType('r'),
        help='raw JSON results file from run_retrieval_bias.py')
    parser.add_argument(
        'analysis_json', nargs='?', type=argparse.FileType('w'), default=sys.stdout,
        help='location for output analysis in JSON (default stdout)')
    parser.add_argument(
        '--diff_gender', action='store_true',
        help='analyse diff gender (man-man woman-woman vs man-woman woman-man) instead of man vs woman')

    args = parser.parse_args()
    results = json.load(args.results_json)

    bias = calculate_retrieval_bias(results, diff_gender=args.diff_gender)
    json.dump(bias, args.analysis_json)


if __name__ == '__main__':
    main()
