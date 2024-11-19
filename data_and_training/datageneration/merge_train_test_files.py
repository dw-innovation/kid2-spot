import pandas as pd
from pathlib import Path
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input_files', type=lambda s:[i for i in s.split(',')])
    parser.add_argument('--output_folder')

    args = parser.parse_args()

    input_files = args.input_files
    output_folder = Path(args.output_folder)

    train = []
    dev = []
    for input_file in input_files:
        data = pd.read_csv(input_file, sep='\t')

        if 'dev' in input_file:
            dev.append(data)
        elif 'train' in input_file:
            train.append(data)


    train = pd.concat(train)
    dev = pd.concat(dev)

    print(f'Number of merged training data: {len(train)}')
    print(f'Number of merged dev data: {len(dev)}')

    train.to_csv(output_folder/'train.tsv', sep='\t', index=False)
    dev.to_csv(output_folder/'dev.tsv', sep='\t', index=False)

