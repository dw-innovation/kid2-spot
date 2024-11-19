import pandas as pd
import json



if __name__ == '__main__':
    with open('datageneration/data/color_encycolorpedia.json', 'r') as file:
        all_color_hex = json.load(file)
        all_color_hex = {key.lower(): value for key, value in all_color_hex.items()}

    color_tags = ['roof:colour=***example***', 'building:colour=***example***']
    color_file = 'datageneration/data/prop_examples_v17.jsonl'
    prop_data = pd.read_json(color_file, lines=True)

    colors = set(sum(prop_data[prop_data['key'].isin(color_tags)]['examples'].tolist(),[]))

    print(f'Number of colors: {len(colors)}')

    hex_to_color_mapping = {}
    none_values = 0
    for color in colors:
        if color.startswith('#'):
            print(f'Checking {color}')
            lowerized_color = color.lower()
            nl_definition = all_color_hex.get(lowerized_color, None)
            print(nl_definition)
            if not nl_definition:
                none_values += 1
                print(lowerized_color)

    print(none_values)