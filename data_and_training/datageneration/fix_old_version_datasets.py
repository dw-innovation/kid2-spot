import yaml
import pandas as pd
from argparse import ArgumentParser

def preprocessing(yaml_str):
    query = yaml.safe_load(yaml_str)
    area = query["area"]

    if area['type'] == 'bbox':
        area.pop('value', None)


    query_string = yaml.dump(query, allow_unicode=True)
    return query_string

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--input_file")
    parser.add_argument("--output_file")
    args = parser.parse_args()

    input_file = args.input_file
    output_file = args.output_file

    data = pd.read_csv(input_file, sep='\t')
    data['query'] = data['query'].apply(lambda x: preprocessing(x))

    data.to_csv(output_file, sep='\t', index=False)
