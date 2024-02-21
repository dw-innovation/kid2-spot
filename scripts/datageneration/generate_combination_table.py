import sys
import numpy as np
import pandas as pd
import json
import random
import re
import csv


def extract_variables(input_string):
    desc = input_string.split("#")[0]
    tag = input_string.split("#")[1]
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


def get_combs(drawn_idx, tag_df, comb_chance, max_number_combs):
    '''
    Takes a tag (key/value pair) and adds a variable number of random tags.
    In reality co-occurring combinations were determined earlier in this info stored in the
    tag_df dataframe.

    :param str drawn_tag: The current key/value pair
    :param pd.dataframe tag_df: Dataframe of all tags plus additional info such as valid tag combinations
    :param float comb_chance: The chance whether combinations should be added at all
    :param int max_number_combs: The maximum number of combinations that can be added to a tag
    '''
    use_combs = np.random.choice([True, False], p=[comb_chance, 1 - comb_chance])
    if use_combs:
        tag_combs = tag_df.iloc[int(drawn_idx)]['combinations']
        if not isNaN(tag_combs):
            tag_combs = tag_combs.split("|")
            comb_weights = [1 / (idx + 1) for idx in range(len(tag_combs))]
            comb_weights = [w / sum(comb_weights) for w in comb_weights]
            num_combs = np.random.choice(np.arange(len(tag_combs)), 1, p=comb_weights)[0] + 1
            num_combs = min(num_combs, max_number_combs)

            drawn_combs = np.random.choice(tag_combs, num_combs, replace=False)
            if len(drawn_combs) > 0:
                get_combs(drawn_combs, tag_df, comb_chance, max_number_combs)

            for comb in drawn_combs:
                yield comb

def pick_tag(tag_list_string):
    tag_list = tag_list_string.split(",")
    tag_list = [tag.strip() for tag in tag_list]

    tag = np.random.choice(tag_list)

    if " AND " in tag:
        tag = np.random.choice(tag.split("AND")).strip()

    return tag


def generate_random_tag_combinations(tag_list_path, arbitrary_value_list_path, version="train"):
    '''
    This method randomly selects a different number of tags (key/value pairs) and adds a variable number of
    additional tags and info. It includes variations such as different comparison operators and different
    positions of substrings in the name (as people might search for e.g. "name beginning with "Gluc").
    The resulting information serves as the basis of overpass queries, covering most OSM tag database info.

    :param str tag_list_path: Path to the CSV file containing all tags + a lot of meta info
    :param str arbitrary_value_list_path: Path to CSV file containing samples for arbitrary and categorical values
    :param str version: Defines whether "train", "dev", or "test" set is currently generated
    '''
    if version == "train":
        num_queries = 10000  # The number of queries (tag-combinations) that will be generated
    elif version == "dev":
        num_queries = 1000
    elif version == "test":
        num_queries = 1000
    max_number_tags_per_query = 4  # The maximum number of objects that will be included in one query
    max_number_combs = 2  # The maximum number of additional tags that will be added to one objects
    comb_chance = 0.4  # Chance that additional tags will be added to a tag
    count_chance = 0.0  # Chance that the %count% tag will be added (indicating search for multiple objects)

    tag_df = pd.read_csv(tag_list_path, header=0, index_col=0,
                         names=['index', 'descriptors', 'type', 'tags', 'combinations'])
    arbitrary_value_df = pd.read_csv(arbitrary_value_list_path, header=0, index_col=0,
                                     names=['index', 'key', 'value_list'])

    # weights = [1 / (idx + 1) for idx in range(max_number_tags_per_query)]
    # weights[0] = 1/8 # Ensure that single object searches are not disproportionally often represented
    # weights = [w / sum(weights) for w in weights] # A weight array that decreases in probability in higher indices
    weights = [0.04, 0.24, 0.38, 0.34]

    # Select type="core" as these are base categories (e.g. "house"), unlike attributes (e.g. "height")
    descriptor_list = tag_df.loc[tag_df['type'] == 'core'][
        'descriptors'].tolist()  # descriptor is the generic but unique name of this category
    tag_lists = tag_df.loc[tag_df['type'] == 'core']['tags'].tolist()
    # key_list = tag_df.loc[tag_df['type'] == 'core']['key'].tolist()
    # value_list = tag_df.loc[tag_df['type'] == 'core']['value'].tolist()
    # tag_list = np.asarray(["{}={}".format(a, b) for a, b in zip(key_list, value_list)]).tolist()
    all_descriptors = tag_df['descriptors'].tolist()
    # tag_desc_list = np.asarray(["{}:{}={}".format(a, b, c) for a, b, c in zip(descriptor_list, key_list, value_list)]).tolist()
    # numeric_list = ["{}=".format(a) for a in tag_df.loc[tag_df['value'] == '***numeric***']['key'].tolist()]
    numeric_list = [num.split("=")[0] + "=" for num in tag_df['tags'].tolist() if "***numeric***" in num]

    # highway_list = ["highway=road", "highway=motorway", "highway=primary", "highway=trunk_link", "highway=path",
    #                 "highway=motorway_link", "highway=residential"]
    # "amenity=restaurant"
    highway_list = ["444", "445", "450", "452", "459", "460", "2450"]

    for query in range(num_queries):
        num_tags = np.random.choice(np.arange(max_number_tags_per_query), 1, p=weights)[
                       0] + 1  # Select number of tags of current query based on weights
        draft_idx = np.random.choice(np.arange(len(tag_lists)), num_tags, replace=False)
        drawn_descriptors = np.asarray(descriptor_list)[draft_idx].tolist()
        drawn_tags = np.asarray(tag_lists)[draft_idx].tolist()
        drawn_tags = [pick_tag(tags) for tags in drawn_tags]
        # drawn_tags = np.random.choice(tag_list, num_tags, replace=False).tolist()

        if num_tags < max_number_tags_per_query:
            increased_rate_chance = 0.15
            if np.random.choice([True, False], p=[increased_rate_chance, 1 - increased_rate_chance]):
                additional_idx = np.random.choice(["34", np.random.choice(highway_list)])
                additional_tag = pick_tag(tag_df.loc[int(additional_idx)]['tags'])
                # additional_key = additional_tag.split("=")[0]
                # additional_value = additional_tag.split("=")[1]
                additional_descriptor = tag_df.loc[int(additional_idx)]['descriptors']
                num_tags += 1
                drawn_tags.append(additional_tag)
                drawn_descriptors.append(additional_descriptor)

        obj_dicts = []
        for di_id, drawn_idx in enumerate(draft_idx):
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
            lane_list = ["474", "483", "9007", "9008", "9009"]

            combs = []
            if drawn_idx in highway_list:
                lane_comb = []
                lane_chance = 0.3
                if np.random.choice([True, False], p=[lane_chance, 1 - lane_chance]):
                    lane_comb = [np.random.choice(lane_list)]
                lane_comb.extend(get_combs(drawn_idx, tag_df, comb_chance, max_number_combs - 1))
                combs = list(set(lane_comb))
            elif drawn_idx == "34":
                restaurant_comb = []
                cuisine_chance = 0.7
                if np.random.choice([True, False], p=[cuisine_chance, 1 - cuisine_chance]):
                    restaurant_comb = ["9010"]
                restaurant_comb.extend(get_combs(drawn_idx, tag_df, comb_chance, max_number_combs - 1))
                combs = list(set(restaurant_comb))
            elif drawn_tag.startswith("building="):
                building_combs = []
                material_chance = 0.2
                if np.random.choice([True, False], p=[material_chance, 1 - material_chance]):
                    building_combs.append("303")
                level_chance = 0.2
                if np.random.choice([True, False], p=[level_chance, 1 - level_chance]):
                    building_combs.append("304")
                building_combs.extend(get_combs(drawn_idx, tag_df, comb_chance, max_number_combs - len(building_combs)))
                combs = list(set(building_combs))
            else:
                combs = list(set(get_combs(drawn_idx, tag_df, comb_chance, max_number_combs)))
            if len(combs) > 0:
                for comb_id, comb in enumerate(combs):
                    row = tag_df.loc[int(comb)]
                    row_tag = pick_tag(row['tags'])
                    row_key = row_tag.split("=")[0]
                    row_val = row_tag.split("=")[1]
                    curr_desc = np.random.choice(row['descriptors'].split("|"), 1)[0]
                    if "***any***" in row_val or "***numeric***"  in row_val:
                        comb_tag = row_key + "="
                    else:
                        if "|" in row_val:
                            comb_tag = row_key + "=" + np.random.choice(row_val.split("|"), 1)[0]
                        else:
                            comb_tag = row_key + "=" + row_val

                    if comb_tag in numeric_list:
                        comb_tag = comb_tag[:-1] + np.random.choice([">", "=", "<"], 1)[
                            0]  # ">=", "<=",  For numeric values, randomly use one of these comparison operators
                        if "height" in comb_tag:
                            combs[comb_id] = comb_tag + get_random_decimal_with_metric(2000)
                        else:
                            combs[comb_id] = comb_tag + str(np.random.choice(np.arange(50), 1)[0])

                    elif len(comb_tag.split("=")[1]) == 0:
                        arb_vals = \
                            arbitrary_value_df.loc[arbitrary_value_df['key'] == comb_tag.split("=")[0]]["value_list"].iloc[0].split("|")
                        drawn_val = np.random.choice(arb_vals, 1)[0]
                        if comb_tag.split("=")[0] in ["name", "addr:street"]:
                            if len(drawn_val) <= 1:
                                version = "equals"
                            else:
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


def generate_query_combinations(tag_list_path, arbitrary_value_list_path, output_filename, version, save_json=False):
    '''
    A method that generates random query combinations and optionally saves them to a JSON file.
    It gets a list of random tag combinations and adds additional information that is required to generate
    full queries, including area names, and different search tasks.
    The current search tasks are: (1) individual distances: a random specific distance is defined between all objects,
    (2) within radius: a single radius within which all objects are located, (3) in area: general search for all objects
    within given area.


    :param str tag_list_path: Path to the CSV file containing all tags + a lot of meta info
    :param str arbitrary_value_list_path: Path to CSV file containing samples for arbitrary and categorical values
    :param str output_filename: Name under which the resulting output file should be stored (minus version specification)
    :param str version: Defines whether "train", "dev", or "test" set is currently generated
    :param bool save_json: Boolean that determines whether a JSON file should be saved or not
    '''
    area_chance = 0.9  # The chance that a specific area will be added to the query

    csc_json = "data/countries+states+cities.json"

    with open(csc_json, "r") as jsonfile:
        csc_dict = json.load(jsonfile)

    # print(csc_dict)
    countries = []
    states = []
    cities = []
    i = 0
    for country in csc_dict:
        countries.append(country["name"])
        for state in country["states"]:
            states.append(state["name"])
            for city in state["cities"]:
                cities.append(city["name"])
    # areas = ["Berlin", "Cologne", "Koblenz"]


    node_types = ["nwr", "cluster", "group"]
    tasks = ["within_radius", "in_area", "individual_distances"]
    task_chances = [0.2, 0.1, 0.7]

    json_list = []

    table_row = 0

    for obj_dicts in generate_random_tag_combinations(tag_list_path, arbitrary_value_list_path, version):
        tag_combinations = [x["props"] for x in obj_dicts]
        # query_items = dict()

        use_area = np.random.choice([True, False], p=[area_chance, 1 - area_chance])
        area_dict = dict()
        if use_area:  # Pick random area from list, or default to "bbox"
            drawn_area = "area"
            # area_val = np.random.choice(areas, 1)[0]
            area_type = np.random.choice(np.asarray([countries, states, cities], dtype=object), p=[0.05, 0.1, 0.85])
            area_val = np.random.choice(area_type)
        else:
            drawn_area = "bbox"
            area_val = ""
            # drawn_area = "bbox"
        # query_items["area"] = drawn_area
        # query_items["area_val"] = area_val
        area_dict["val"] = area_val
        area_dict["type"] = drawn_area

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

        nodes = [area_dict, obj_dicts]

        json_dict = {"nodes": nodes, "relations": edges, "action": action}
        table_row += 1

        nf = translate_to_new_format(json_dict)
        json_list.append(nf)
        print(nf)

        # print(json.dumps(json_dict, indent=4, cls=NpEncoder))

    if save_json:
        with open(output_filename + "_" + version + ".json", "w") as jsonfile:
            json.dump(json_list, jsonfile, cls=NpEncoder)
        print("Saved file to output path!")

        with open(output_filename + "_" + version + ".csv", 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            for item in json_list:
                csv_writer.writerow([item])

    return json_list


if __name__ == '__main__':
    '''
    Define paths and run all desired functions.
    '''
    tag_list_path = "data/Tag_List_v9.csv"
    arbitrary_value_list_path = "data/Arbitrary_Value_List_v9.csv"
    output_filename = "results/IMR_Dataset_v10"
    
    for version in ["train", "dev", "test"]:
        comb_df = generate_query_combinations(tag_list_path, arbitrary_value_list_path, output_filename, version, True)

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
