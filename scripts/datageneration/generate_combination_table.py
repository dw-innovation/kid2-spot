import json
import random
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def extract_variables(input_string):
    try:
        desc = input_string.split("#")[0]
        tag = input_string.split("#")[1]
    except IndexError as e:
        print(f"Index error while extracting variables {e} for {input_string}")
        return None, None, None, None
    operators = ['=', '>', '<', '~']  # '>=', '<=',
    for op in operators:
        if op in input_string:
            key, rest = tag.split(op, 1)
            # if op == "~":
            #     value = re.search(r'"([^"]*)"', rest).group(1)
            # else:
            value = rest
            return key.strip(), value.strip(), op, desc.strip()

    return None, None, None, None


def translate_to_new_format(json_dict):
    new_format = {
        "a": {},
        "ns": [],
        "es": []
    }

    node_mapping = {}  # To keep track of node IDs in the new format

    # new_format["a"]["t"] = "bbox" if json_dict["nodes"][0]["name"] == "bbox" else "polygon"
    new_format["a"]["t"] = json_dict["nodes"][0]["type"]
    new_format["a"]["v"] = json_dict["nodes"][0]["val"]
    for node in json_dict["nodes"][1]:
        nwr = {
            "id": len(new_format["ns"]),
            # "n": node["name"],
            # "osm_tag": node.get("props", [])[0].split("#")[1],
            "flts": [],
            "t": node["type"]
        }

        filters = node.get("props", [])
        for pid, prop in enumerate(filters):
            # if pid == 0:
            #     continue
            k, v, op, d = extract_variables(prop)
            filter_op = op

            nwr["flts"].append({
                "k": k,
                "v": v,
                "op": filter_op,
                "n": d
            })

        new_format["ns"].append(nwr)
        node_mapping[node["name"]] = nwr["id"]

    for relation in json_dict["relations"]:
        edge = {
            "src": relation["from"],
            "tgt": relation["to"]
        }

        if "weight" in relation:
            edge["t"] = "dist"
            edge["dist"] = relation["weight"]
        else:
            edge["t"] = "in"

        new_format["es"].append(edge)

    return new_format


def get_random_decimal_with_metric(range):
    h_ = np.random.choice(np.arange(range), 1)[0]
    if np.random.choice([True, False], 1)[0]:
        h_ = h_ / np.random.choice([10, 100], 1)[0]

    h_ = str(h_) + " " + np.random.choice(["mm", "cm", "m", "km", "in", "ft", "yd", "mi", "le"], 1)[0]  # "cm",

    return h_


def isNaN(string):
    '''
    Checks if a string is "NaN".

    :param str string: The string to be checked
    '''
    return string != string


class NpEncoder(json.JSONEncoder):
    '''
    Custom encoder for the json.dumps function that can handle numpy datastructures.

    :param JSONEncoder json.JSONEncoder: Extensible JSON encoder for python datastructures
    '''

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


# ipek - what does it do?
def pick_tag(tag_list_string):
    tag_list = tag_list_string.split(",")
    tag_list = [tag.strip() for tag in tag_list]

    tag = np.random.choice(tag_list)

    if " AND " in tag:
        tag = np.random.choice(tag.split("AND")).strip()

    return tag


class QueryCombinationGenerator(object):
    def __init__(self, geolocations_file_path, tag_list_path, arbitrary_value_list_path):
        countries, states, cities = self.fetch_countries_states_cities(geolocations_file_path)

        self.countries = countries
        self.states = states
        self.cities = cities
        self.arbitrary_value_df = pd.read_csv(arbitrary_value_list_path, dtype={'index': int})

        # Select type="core" as these are base categories (e.g. "house"), unlike attributes (e.g. "height")

        # the following data structure produce unodered list, so the drawn index don't map the correct ones
        # self.descriptor_list = self.tag_df.loc[self.tag_df['type'] == 'core'][
        #     'descriptors'].tolist()  # descriptor is the generic but unique name of this category
        # ipek - changed it as follows

        # todo: find a better name
        # self.descriptor_list = {}
        # self.tag_lists = {}

        # for idx, row in self.tag_df[self.tag_df['type'] == 'core'].iterrows():
        #     self.descriptor_list[row['index']] = row['descriptors']
        #     self.tag_lists[row['index']] = row['tags']

        tag_df = pd.read_csv(tag_list_path)
        tag_df = tag_df[tag_df.select_dtypes(float).notna().any(axis=1)]

        descriptors_to_idx = {}
        all_tags = {}
        core_tags = {}
        for tag in tag_df.to_dict(orient='records'):
            all_tags[int(tag['index'])] = tag
            descriptors = tag['descriptors'].split('|')

            if tag['type'] == 'core':
                core_tags[int(tag['index'])] = tag

            for descriptor in descriptors:
                descriptors_to_idx[descriptor.strip()] = int(tag['index'])

        self.core_tags = core_tags
        self.desriptors_to_idx = descriptors_to_idx
        self.all_tags = all_tags
        self.numeric_list = [num.split("=")[0] + "=" for num in tag_df['tags'].tolist() if "***numeric***" in num]
        # self.tag_lists = self.tag_df.loc[self.tag_df['type'] == 'core']['tags'].tolist()

    # is this for determining features, this mostly gives one combination, I think use_combs should not be in the function
    # ipek - check why it is recursive???
    def get_combs(self, drawn_idx, max_number_combs):
        '''
        Takes a tag (key/value pair) and adds a variable number of random tags.
        In reality co-occurring combinations were determined earlier in this info stored in the
        tag_df dataframe.

        :param str drawn_tag: The current key/value pair
        :param pd.dataframe tag_df: Dataframe of all tags plus additional info such as valid tag combinations -- get from self.tag_df
        :param float comb_chance: The chance whether combinations should be added at all
        :param int max_number_combs: The maximum number of combinations that can be added to a tag
        '''
        tag_combs = self.all_tags[int(drawn_idx)]['combinations']
        if not isNaN(tag_combs):
            tag_combs = tag_combs.split("|")
            comb_weights = [1 / (idx + 1) for idx in range(len(tag_combs))]
            comb_weights = [w / sum(comb_weights) for w in comb_weights]
            num_combs = np.random.choice(np.arange(len(tag_combs)), 1, p=comb_weights)[0] + 1
            num_combs = min(num_combs, max_number_combs)

            drawn_combs = np.random.choice(tag_combs, num_combs, replace=False)
            if len(drawn_combs) > 0:
                self.get_combs(drawn_combs, max_number_combs)

            for comb in drawn_combs:
                yield comb

    def index_to_descriptors(self, index):
        return self.all_tags[int(index)]['descriptors']

    def generate_random_tag_combinations(self, num_queries):
        '''
        This method randomly selects a different number of tags (key/value pairs) and adds a variable number of
        additional tags and info. It includes variations such as different comparison operators and different
        positions of substrings in the name (as people might search for e.g. "name beginning with "Gluc").
        The resulting information serves as the basis of overpass queries, covering most OSM tag database info.

        :param str tag_list_path: Path to the CSV file containing all tags + a lot of meta info
        :param str arbitrary_value_list_path: Path to CSV file containing samples for arbitrary and categorical values
        :param str num_queries: Defines whether "train", "dev", or "test" set is currently generated
        '''

        # ipek - I removed version and changed it to num_queries, because version was used as another variable in the code.

        # ipek - why didn't you make them parametric?
        max_number_tags_per_query = 4  # The maximum number of objects that will be included in one query

        # ipek - i would add it as max_number of features
        max_number_combs = 4  # The maximum number of additional tags that will be added to one objects
        comb_chance = 0.7  # Chance that additional tags will be added to a tag
        count_chance = 0.0  # Chance that the %count% tag will be added (indicating search for multiple objects)

        # weights = [1 / (idx + 1) for idx in range(max_number_tags_per_query)]
        # weights[0] = 1/8 # Ensure that single object searches are not disproportionally often represented
        # weights = [w / sum(weights) for w in weights] # A weight array that decreases in probability in higher indices

        # Ipek how come we got these probs?
        weights = [0.04, 0.24, 0.38, 0.34]

        # key_list = tag_df.loc[tag_df['type'] == 'core']['key'].tolist()
        # value_list = tag_df.loc[tag_df['type'] == 'core']['value'].tolist()
        # tag_list = np.asarray(["{}={}".format(a, b) for a, b in zip(key_list, value_list)]).tolist()

        # ipek all_descriptors are not used by any func
        # all_descriptors = tag_df['descriptors'].tolist()
        # tag_desc_list = np.asarray(["{}:{}={}".format(a, b, c) for a, b, c in zip(descriptor_list, key_list, value_list)]).tolist()
        # numeric_list = ["{}=".format(a) for a in tag_df.loc[tag_df['value'] == '***numeric***']['key'].tolist()]

        # highway_list = ["highway=road", "highway=motorway", "highway=primary", "highway=trunk_link", "highway=path",
        #                 "highway=motorway_link", "highway=residential"]
        # "amenity=restaurant"
        # comment out
        # highway_list = ["444", "445", "450", "452", "459", "460", "2450"]

        for query in range(num_queries):
            num_tags = np.random.choice(np.arange(max_number_tags_per_query), 1, p=weights)[
                           0] + 1  # Select number of tags of current query based on weights

            # ipek - I changed from len(self.tag_lists) to list of indices
            # draft_indices = np.random.choice(np.arange(len(self.tag_lists)), num_tags, replace=False)
            draft_indices = np.random.choice(np.asarray(list(self.core_tags.keys())), num_tags, replace=False)

            # drawn_descriptors = np.asarray(self.descriptor_list)[draft_idx].tolist()
            drawn_descriptors = [self.core_tags[draft_index]['descriptors'] for draft_index in draft_indices]
            # drawn_tags = np.asarray(self.tag_lists)[draft_indices].tolist()

            drawn_tags = [pick_tag(self.core_tags[draft_index]['tags']) for draft_index in draft_indices]

            # drawn_tags = np.random.choice(tag_list, num_tags, replace=False).tolist()

            # ipek - unclear for me -- commented out
            # if num_tags < max_number_tags_per_query:
            #     increased_rate_chance = 0.15
            #     if np.random.choice([True, False], p=[increased_rate_chance, 1 - increased_rate_chance]):
            #         # why do we pick 34?
            #         additional_idx = np.random.choice(["34", np.random.choice(highway_list)])
            #         additional_tag = pick_tag(self.tag_df.loc[int(additional_idx)]['tags'])
            #         # additional_key = additional_tag.split("=")[0]
            #         # additional_value = additional_tag.split("=")[1]
            #         additional_descriptor = self.tag_df.loc[int(additional_idx)]['descriptors']
            #         num_tags += 1
            #         drawn_tags.append(additional_tag)
            #         drawn_descriptors.append(additional_descriptor)

            # ipek what is obj_dicts?
            obj_dicts = []

            for di_id, drawn_idx in enumerate(draft_indices):
                drawn_tag = drawn_tags[di_id]

                obj_dict = dict()
                if "|" in drawn_descriptors[
                    di_id]:  # In case an object has multiple descriptors, use both (this create more variance in generated natural sentences)
                    obj_dict["name"] = np.random.choice(drawn_descriptors[di_id].split("|"), 1)[0]
                else:
                    obj_dict["name"] = drawn_descriptors[di_id]
                obj_dict["type"] = "nwr"

                drawn_tags[di_id] = [obj_dict["name"] + "#" + drawn_tag]  # {obj_dict["name"]: drawn_tag}

                # use_count = np.random.choice([True, False], p=[count_chance, 1 - count_chance])
                # if use_count:
                #     count = np.random.choice(np.arange(2, 20), 1)[0]
                #     drawn_tags[dt_id].extend(["%count%=" + str(count)])

                # ipek - I commented out the following code snippet
                # lane_list = ["474", "483", "9007", "9008", "9009"]
                #
                # combs = []
                # if drawn_idx in highway_list:
                #     lane_comb = []
                #     lane_chance = 0.3
                #     if np.random.choice([True, False], p=[lane_chance, 1 - lane_chance]):
                #         lane_comb = [np.random.choice(lane_list)]
                #     lane_comb.extend(get_combs(drawn_idx, self.tag_df, comb_chance, max_number_combs - 1))
                #     combs = list(set(lane_comb))
                # elif drawn_idx == "34":
                #     restaurant_comb = []
                #     cuisine_chance = 0.7
                #     if np.random.choice([True, False], p=[cuisine_chance, 1 - cuisine_chance]):
                #         restaurant_comb = ["9010"]
                #     restaurant_comb.extend(get_combs(drawn_idx, self.tag_df, comb_chance, max_number_combs - 1))
                #     combs = list(set(restaurant_comb))
                # elif drawn_tag.startswith("building="):
                #     building_combs = []
                #     material_chance = 0.2
                #     if np.random.choice([True, False], p=[material_chance, 1 - material_chance]):
                #         building_combs.append("303")
                #     level_chance = 0.2
                #     if np.random.choice([True, False], p=[level_chance, 1 - level_chance]):
                #         building_combs.append("304")
                #     building_combs.extend(
                #         get_combs(drawn_idx, self.tag_df, comb_chance, max_number_combs - len(building_combs)))
                #     combs = list(set(building_combs))
                # else:
                #     combs = list(set(get_combs(drawn_idx, self.tag_df, comb_chance, max_number_combs)))

                # ipek - i commented out the previous code snippet, and uncommented the others
                use_combs = np.random.choice([True, False], p=[comb_chance, 1 - comb_chance])

                if use_combs:
                    combs = list(set(self.get_combs(drawn_idx=drawn_idx, max_number_combs=max_number_combs)))
                    if len(combs) > 0:
                        for comb_id, comb in enumerate(combs):
                            try:
                                row = self.all_tags[int(comb)]
                            except ValueError as e:
                                print(f"comb {comb} has value error: {e}")
                                continue
                            row_tag = pick_tag(row['tags'])
                            row_key = row_tag.split("=")[0]

                            try:
                                row_val = row_tag.split("=")[1]
                            except IndexError as e:
                                continue
                                print(f"Index error on {row_tag}: {e}")

                            curr_desc = np.random.choice(row['descriptors'].split("|"), 1)[0]
                            if "***any***" in row_val or "***numeric***" in row_val:
                                comb_tag = row_key + "="
                            else:
                                if "|" in row_val:
                                    comb_tag = row_key + "=" + np.random.choice(row_val.split("|"), 1)[0]
                                else:
                                    comb_tag = row_key + "=" + row_val

                            if comb_tag in self.numeric_list:
                                comb_tag = comb_tag[:-1] + np.random.choice([">", "=", "<"], 1)[
                                    0]  # ">=", "<=",  For numeric values, randomly use one of these comparison operators
                                if "height" in comb_tag:
                                    combs[comb_id] = comb_tag + get_random_decimal_with_metric(2000)
                                else:
                                    combs[comb_id] = comb_tag + str(np.random.choice(np.arange(50), 1)[0])

                            elif len(comb_tag.split("=")[1]) == 0:
                                try:
                                    arb_vals = \
                                        self.arbitrary_value_df.loc[
                                            self.arbitrary_value_df['key'] == comb_tag.split("=")[0]][
                                            "value_list"].iloc[0].split("|")
                                except IndexError as e:
                                    print(f'{comb_tag} has an indexing error {e}')
                                drawn_val = np.random.choice(arb_vals, 1)[0]
                                if comb_tag.split("=")[0] in ["name", "addr:street"]:
                                    if len(drawn_val) <= 1:
                                        version = "equals"
                                    else:
                                        # ipek - what is the logic behind this?
                                        # version = np.random.choice(["begins", "ends", "contains", "equals"], 1)[0]
                                        version = np.random.choice(["contains", "equals"], 1)[
                                            0]  # Randomly select one of these variants and format the string + regex accordingly
                                    # if version == "begins":
                                    #     cutoff = np.random.choice(np.arange(1, len(drawn_val)))
                                    #     combs[comb_id] = comb_tag[:-1] + "~\"^" + drawn_val[:cutoff] + "\""
                                    # elif version == "ends":
                                    #     cutoff = np.random.choice(np.arange(1, len(drawn_val)))
                                    #     combs[comb_id] = comb_tag[:-1] + "~\"" + drawn_val[cutoff:] + "$\""
                                    if version == "contains":
                                        len_substring = np.random.choice(np.arange(1, len(drawn_val)))
                                        idx = random.randrange(0, len(drawn_val) - len_substring + 1)
                                        combs[comb_id] = comb_tag[:-1] + "~" + drawn_val[idx: (idx + len_substring)]
                                    else:
                                        combs[comb_id] = comb_tag + drawn_val
                                else:
                                    combs[comb_id] = comb_tag + drawn_val
                            else:
                                combs[comb_id] = comb_tag
                                # if "|" in row['value']:
                                #     combs[comb_id] = row['key'] + "=" + np.random.choice(row['value'].split("|"), 1)[0]
                                # else:
                                #     combs[comb_id] = row['key'] + "=" + row['value']

                            combs[comb_id] = curr_desc + "#" + combs[comb_id]

                        drawn_tags[di_id].extend(combs)

                obj_dict["props"] = drawn_tags[di_id]
                obj_dicts.append(obj_dict)

            yield obj_dicts

    def fetch_countries_states_cities(self, geolocations_file_path):
        with open(geolocations_file_path, "r") as jsonfile:
            csc_dict = json.load(jsonfile)

        countries = []
        states = []
        cities = []

        # ipek: do we need this i, it is not used.
        i = 0
        for country in csc_dict:
            countries.append(country["name"])
            for state in country["states"]:
                states.append(state["name"])
                for city in state["cities"]:
                    cities.append(city["name"])

        return countries, states, cities

    def run(self, area_chance, num_queries):
        '''
        A method that generates random query combinations and optionally saves them to a JSON file.
        It gets a list of random tag combinations and adds additional information that is required to generate
        full queries, including area names, and different search tasks.
        The current search tasks are: (1) individual distances: a random specific distance is defined between all objects,
        (2) within radius: a single radius within which all objects are located, (3) in area: general search for all objects
        within given area.

        :param float area_chance: probability for picking up real name
        :param str tag_list_path: Path to the CSV file containing all tags + a lot of meta info
        :param str arbitrary_value_list_path: Path to CSV file containing samples for arbitrary and categorical values
        :param str output_filename: Name under which the resulting output file should be stored (minus version specification)
        :param str version: Defines whether "train", "dev", or "test" set is currently generated
        :param bool save_json: Boolean that determines whether a JSON file should be saved or not
        '''
        # ipek - node types are not used
        node_types = ["nwr", "cluster", "group"]
        tasks = ["within_radius", "in_area", "individual_distances"]

        # Ipek what is rationale to choose these probs
        task_chances = [0.2, 0.1, 0.7]

        json_list = []

        # ipek- we don't need this anymore
        # table_row = 0

        for obj_dicts in tqdm(self.generate_random_tag_combinations(num_queries), total=num_queries):
            tag_combinations = [x["props"] for x in obj_dicts]
            # ipek - i capsulate the area generation as function
            area_item = self.generate_area(area_chance)

            edges = []

            task = np.random.choice(tasks, p=task_chances)

            if task == "individual_distances" and len(
                    tag_combinations) > 2:  # Pick random distance between all individual objects
                action = "individual_distances"

                for t_no, t in enumerate(tag_combinations):
                    if t_no != len(tag_combinations) - 1:
                        edge_dict = {"from": t_no, "to": t_no + 1, "weight": get_random_decimal_with_metric(2000)}
                        edges.append(edge_dict)
            elif task == "in_area" or len(
                    tag_combinations) == 1:  # Just search for all given objects in area, no distance required
                action = "in_area"

                for od in obj_dicts:
                    od["props"] = [s for s in od["props"] if "%count%" not in s]

                # edge_dict = {"from": -1, "to": -1, "weight": -1}
                # edges.append(edge_dict)
            else:  # Search for all places where all objects are within certain radius
                action = "within_radius"
                dist_ = get_random_decimal_with_metric(2000)

                for t_no, t in enumerate(tag_combinations):
                    if t_no != len(tag_combinations) - 1:
                        edge_dict = {"from": 0, "to": t_no + 1, "weight": dist_}
                        edges.append(edge_dict)

                # edge_dict = {"from": -1, "to": -1, "weight": get_random_decimal_with_metric(2000)}
                # edges.append(edge_dict)

            nodes = [area_item, obj_dicts]

            json_dict = {"nodes": nodes, "relations": edges, "action": action}

            # ipek- we don't need this anymore
            # table_row += 1

            nf = translate_to_new_format(json_dict)
            json_list.append(nf)

            # print(json.dumps(json_dict, indent=4, cls=NpEncoder))
        # ipek - I commented the following text, and move into the main to avoid multiple params in the function
        # if save_json:
        #     with open(output_filename + "_" + version + ".json", "w") as jsonfile:
        #         json.dump(json_list, jsonfile, cls=NpEncoder)
        #     print("Saved file to output path!")
        #
        #     with open(output_filename + "_" + version + ".csv", 'w', newline='') as csvfile:
        #         csv_writer = csv.writer(csvfile)
        #         for item in json_list:
        #             csv_writer.writerow([item])

        return json_list

    def generate_area(self, area_chance):
        use_area = np.random.choice([True, False], p=[area_chance, 1 - area_chance])

        if use_area:  # Pick random area from list, or default to "bbox"
            drawn_area = "area"
            # area_val = np.random.choice(areas, 1)[0]
            area_type = np.random.choice(np.asarray([self.countries, self.states, self.cities], dtype=object),
                                         p=[0.05, 0.1, 0.85])
            area_val = np.random.choice(area_type)
        else:
            drawn_area = "bbox"
            area_val = ""
            # drawn_area = "bbox"

        return dict(val=area_val, type=drawn_area)


def write_output(generated_combs, output_file):
    # ipek - output should be jsonl

    with open(output_file, "w") as out_file:
        for generated_comb in generated_combs:
            json.dump(generated_comb, out_file)
            out_file.write('\n')

        # if save_json:
        #     with open(output_filename + "_" + version + ".json", "w") as jsonfile:
        #         json.dump(json_list, jsonfile, cls=NpEncoder)
        #     print("Saved file to output path!")
        #
        #     with open(output_filename + "_" + version + ".csv", 'w', newline='') as csvfile:
        #         csv_writer = csv.writer(csvfile)
        #         for item in json_list:
        #             csv_writer.writerow([item])
        #


if __name__ == '__main__':
    '''
    Define paths and run all desired functions.
    '''
    parser = ArgumentParser()
    parser.add_argument('--geolocations_file_path', help='Path to a file containing cities, countries, etc.')
    parser.add_argument('--tag_list_path', help='tag list file generated via retrieve_combinations')
    parser.add_argument('--arbitrary_value_list_path', help='Arbitrary value list generated via combinations')
    parser.add_argument('--output_folder', help='Folder to save the output')
    parser.add_argument('--write_output', action='store_true')
    parser.add_argument('--area_chance', help='Add to probability of picking real area', type=float)
    parser.add_argument('--training_samples', help='number of samples to generate as a training set', type=int)
    parser.add_argument('--development_samples', help='number of samples to generate as a development set', type=int)
    parser.add_argument('--testing_samples', help='number of samples to generate as a test set', type=int)

    args = parser.parse_args()

    tag_list_path = args.tag_list_path
    arbitrary_value_list_path = args.arbitrary_value_list_path
    output_folder = args.output_filename = args.output_folder
    geolocations_file_path = args.geolocations_file_path

    query_comb_generator = QueryCombinationGenerator(geolocations_file_path=geolocations_file_path,
                                                     tag_list_path=tag_list_path,
                                                     arbitrary_value_list_path=arbitrary_value_list_path)
    area_chance = args.area_chance

    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    dataset_splits = {
        'train': args.training_samples,
        'development': args.development_samples,
        'test': args.testing_samples
    }

    for ds_split, num_samples in dataset_splits.items():

        generated_combs = query_comb_generator.run(area_chance=area_chance,
                                                   num_queries=num_samples)

        if args.write_output:
            write_output(generated_combs, output_file=output_folder / f'{ds_split}.jsonl')

    # # Example input dictionary
    # json_dict = {
    #     "nodes": [
    #         {"name": "bbox", "type": "area"},
    #         [
    #             {"name": "apartment", "type": "object", "props": ["building=apartments"]},
    #             {"name": "bar", "type": "object", "props": ["amenity=bar", "height>240"]},
    #             {"name": "school", "type": "object",
    #              "props": ["amenity=school", "building:material=cement_block", 'name~"T"']}
    #         ]
    #     ],
    #     "relations": [
    #         {"from": 0, "to": 1, "weight": 619},
    #         {"from": 1, "to": 2, "weight": 431}
    #     ],
    #     "action": "individual_distances"
    # }
    #
    # new_format_data = translate_to_new_format(json_dict)
    # print(new_format_data)
