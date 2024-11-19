import codecs
import re
import json
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import yaml

def merge_jsonl_files(input_files, output_file):
    with open(output_file, 'w') as outfile:
        for jsonl_file in input_files:
            with open(jsonl_file, 'r') as infile:
                for line in infile:
                    outfile.write(line)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input_file')
    parser.add_argument('--output_folder')
    parser.add_argument('--dev_samples', type=int)

    args = parser.parse_args()

    input_file = args.input_file
    output_folder = Path(args.output_folder)
    dev_samples = args.dev_samples

    samples_df = pd.read_json(input_file, orient='records', lines=True)
    processed_samples_df = []
    for idx, sample in samples_df.iterrows():
        sentence = sample["sentence"]
        if '''sorry''' in sentence.lower():
            continue
        if isinstance(sentence, float):
            continue

        query = sample["query"]
        query = yaml.safe_load(query)
        query_string = yaml.dump(query, allow_unicode=True)
        processed_samples_df.append({'query': query_string, 'sentence': sentence})


    processed_samples_df = pd.DataFrame(processed_samples_df)
    development_set = processed_samples_df.sample(dev_samples)
    training_set = processed_samples_df[~processed_samples_df['query'].isin(development_set['query'].tolist())]

    print(f"Number of training set: {len(training_set)}")
    print(f"Number of validated samples: {len(development_set)}")

    training_set.to_csv(output_folder / 'train.tsv', sep="\t", index=False)
    development_set.to_csv(output_folder / 'dev.tsv', sep="\t", index=False)

