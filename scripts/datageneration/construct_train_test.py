import codecs
import re
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import yaml


def decode_unicode(text):
    def unicode_replacer(match):
        return codecs.decode(match.group(0), 'unicode_escape')

    pattern = re.compile(r'\\u[0-9a-fA-F]{4}')
    decoded_text = pattern.sub(unicode_replacer, text)
    decoded_text = decoded_text.replace('\\', '')
    return decoded_text


def convert_yaml_output(data):
    minimized_data = {}
    minimized_data["area"] = data["a"]

    if isinstance(minimized_data["area"], list):
        print("That is a list!!!!!!")

    else:
        minimized_data["area"]["type"] = minimized_data["area"].pop("t")
        minimized_data["area"]["name"] = minimized_data["area"].pop("v")
        minimized_data["area"]["name"] = decode_unicode(minimized_data["area"]["name"])

    # if data["area"]["value"] == "":
    #     del data["a"]["v"]

    if len(data["es"]) > 0:
        minimized_data["relations"] = data["es"]

        for idx, relation in enumerate(minimized_data["relations"]):
            minimized_data["relations"][idx]["source"] = relation.pop("src")
            minimized_data["relations"][idx]["target"] = relation.pop("tgt")
            minimized_data["relations"][idx]["name"] = relation.pop("t")
            minimized_data["relations"][idx]["value"] = relation.pop("dist")

    if len(data["ns"]) > 0:
        minimized_data["entities"] = []

    nodes = []
    for idx, node in enumerate(data["ns"]):
        minimized_node = {}
        minimized_node["id"] = node["id"]
        minimized_node["name"] = node["flts"][0]["n"]

        flts = []
        if len(node["flts"]) > 1:
            for flt in node["flts"][1:]:
                if not flt["k"]:
                    continue
                if ":" in flt["n"]:
                    flt["n"] = flt["n"].split(":")[-1]
                flts.append({"name": flt["n"], "operator": flt["op"], "value": flt["v"]})

        if len(flts) > 0:
            minimized_node["filters"] = flts

        nodes.append(minimized_node)

    minimized_data["entities"] = nodes
    minimized_data = yaml.dump(minimized_data, allow_unicode=True)

    return minimized_data


def merge_jsonl_files(input_files, output_file):
    with open(output_file, 'w') as outfile:
        for jsonl_file in input_files:
            with open(jsonl_file, 'r') as infile:
                for line in infile:
                    outfile.write(line)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input_folder')
    parser.add_argument('--output_folder')
    parser.add_argument('--dev_samples', type=int)

    args = parser.parse_args()

    input_folder = args.input_folder
    output_folder = Path(args.output_folder)
    dev_samples = args.dev_samples

    samples_as_df = []
    for fname in Path(input_folder).rglob('*.jsonl'):
        sample_df = pd.read_json(fname, orient='records', lines=True)
        sample_df["query"] = sample_df["query"].apply(lambda x: convert_yaml_output(x))
        sample_df.rename(columns={"text": "sentence"}, inplace=True)

        sample_df["sentence"] = sample_df["sentence"].apply(lambda x: x.replace("\"", "").replace("\n", " "))
        sample_df = sample_df[~sample_df['sentence'].str.contains('''sorry''', flags=re.IGNORECASE, regex=True)]
        sample_df = sample_df[["sentence", "query"]]

        samples_as_df.append(sample_df)

    samples_as_df = pd.concat(samples_as_df)

    development_set = samples_as_df.sample(dev_samples)
    training_set = samples_as_df[~samples_as_df['query'].isin(development_set['query'].tolist())]

    print(f"Number of training set: {len(training_set)}")
    print(f"Number of validated samples: {len(development_set)}")

    development_set.to_csv(output_folder / 'dev.tsv', sep="\t", index=False)
    training_set.to_csv(output_folder / 'train.tsv', sep="\t", index=False)
