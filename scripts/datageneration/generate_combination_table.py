import json
from argparse import ArgumentParser
from typing import List

import numpy as np
from datageneration.data_model import Entity, Area, Relation, Property, LocPoint, TagCombination, TagAttributeExample, \
    TagAttribute
from datageneration.property_generator import PropertyGenerator
from datageneration.relation_generator import RelationGenerator
from tqdm import tqdm


# ipek - my suggestion is to generate 100 samples for each tags and then split this dataset into training/test/validation

# def extract_variables(input_string): LocPoint
#     try:
#         desc = input_string.split("#")[0]
#         tag = input_string.split("#")[1]
#     except IndexError as e:
#         print(f"Index error while extracting variables {e} for {input_string}")
#         return None, None, None, None
#     operators = ['=', '>', '<', '~']  # '>=', '<=',
#     for op in operators:
#         if op in input_string:
#             key, rest = tag.split(op, 1)
#             # if op == "~":
#             #     value = re.search(r'"([^"]*)"', rest).group(1)
#             # else:
#             value = rest
#             return key.strip(), value.strip(), op, desc.strip()
#
#     return None, None, None, None


# def translate_to_new_format(json_dict):
#     print("json dictionary")
#     print(json_dict)
#     new_format = {
#         "a": {},
#         "ns": [],
#         "es": []
#     }
#
#     node_mapping = {}  # To keep track of node IDs in the new format
#
#     # new_format["a"]["t"] = "bbox" if json_dict["nodes"][0]["name"] == "bbox" else "polygon"
#     new_format["a"]["t"] = json_dict["nodes"][0]["type"]
#     new_format["a"]["v"] = json_dict["nodes"][0]["val"]
#     for node in json_dict["nodes"][1]:
#         nwr = {
#             "id": len(new_format["ns"]),
#             # "n": node["name"],
#             # "osm_tag": node.get("props", [])[0].split("#")[1],
#             "flts": [],
#             "t": node["type"]
#         }
#
#         filters = node.get("props", [])
#         for pid, prop in enumerate(filters):
#             # if pid == 0:
#             #     continue
#             k, v, op, d = extract_variables(prop)
#             filter_op = op
#
#             nwr["flts"].append({
#                 "k": k,
#                 "v": v,
#                 "op": filter_op,
#                 "n": d
#             })
#
#         new_format["ns"].append(nwr)
#         node_mapping[node["name"]] = nwr["id"]
#
#     for relation in json_dict["relations"]:
#         edge = {
#             "src": relation["from"],
#             "tgt": relation["to"]
#         }
#
#         if "weight" in relation:
#             edge["t"] = "dist"
#             edge["dist"] = relation["weight"]
#         else:
#             edge["t"] = "in"
#
#         new_format["es"].append(edge)
#
#     return new_format


def get_random_decimal_with_metric(range):
    h_ = np.random.choice(np.arange(range), 1)[0]
    if np.random.choice([True, False], 1)[0]:
        h_ = h_ / np.random.choice([10, 100], 1)[0]

    h_ = str(h_) + " " + np.random.choice(["m", "km", "in", "ft", "yd", "mi", "le"], 1)[0]  # "cm",

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
    def __init__(self, geolocations, tag_combinations: List[TagCombination],
                 attribute_examples: List[TagAttributeExample]):
        # countries, states, cities = self.fetch_countries_states_cities(geolocations_file_path)
        # print(countries, states, cities)
        # self.countries = countries
        # self.states = states
        # self.cities = cities

        # self.tag_combinations = tag_combinations
        self.entity_tag_combinations = list(filter(lambda x: 'core' in x['comb_type'], tag_combinations))
        # self.attribute_examples = attribute_examples
        self.property_generator = PropertyGenerator(attribute_examples)
        self.relation_generator = RelationGenerator(max_distance=2000)

        # id2_descriptors = {}
        # descriptor2_ids = {}
        # all_tags = {}
        # osm_tags = {}
        # osm_properties = {}
        #
        # for tag in tag_df.to_dict(orient='records'):
        #     all_tags[int(tag['index'])] = tag
        #     descriptors = tag['descriptors'].split('|')
        #     descriptors = list(map(lambda x: x.strip().lower(), descriptors))
        #
        #     tag_id = int(tag['index'])
        #     if tag_id not in id2_descriptors:
        #         id2_descriptors[tag_id] = descriptors
        #
        #     if tag['type'] == 'core':
        #         osm_tags[int(tag['index'])] = tag
        #     else:
        #         osm_properties[int(tag['index'])] = tag
        #
        #     for descriptor in descriptors:
        #         descriptor2_ids[descriptor] = int(tag['index'])
        #
        # self.osm_tags = osm_tags
        # self.osm_tag_ids = list(osm_tags.keys())
        # self.osm_properties = osm_properties
        # self.id2_descriptors = id2_descriptors
        # self.descriptor2_ids = descriptor2_ids
        #
        # self.all_tags = all_tags
        # self.numeric_list = [num.split("=")[0] + "=" for num in tag_df['tags'].tolist() if "***numeric***" in num]
        #
        # # settings for relation generation
        # self.MAX_DISTANCE = 2000
        # self.task_chances = {
        #     "within_radius": 0.2,
        #     "in_area": 0.1,
        #     "individual_distances": 0.7
        # }

    def index_to_descriptors(self, index):
        return self.all_tags[int(index)]['descriptors']

    def generate_prompt_entities(self, number_of_entities_in_prompt: int, max_number_of_props_in_entity: int) -> List[
        Entity]:
        """
        Generates a list of entities with associated properties based on random selection of descriptors.

        Args:
            number_of_entities_in_prompt (int): Number of entities to generate.
            max_number_of_props_in_entity (int): Maximum number of properties each entity can have.

        Returns:
            List[Entity]: A list of generated entities with associated properties.

        Note:
            - The function selects a random subset of descriptors from the available combinations of entity tag combinations.
            - Each entity is assigned a random name chosen from the selected descriptors.
            - If `max_number_of_props_in_entity` is greater than or equal to 1, properties are generated for each entity.
              Otherwise, entities are generated without properties.
        """
        selected_entities = []
        while len(selected_entities) < number_of_entities_in_prompt:
            selected_idx_for_combinations = np.random.randint(0, len(self.entity_tag_combinations))
            selected_tag_comb = self.entity_tag_combinations[selected_idx_for_combinations]
            associated_descriptors = selected_tag_comb['descriptors']
            entity_name = np.random.choice(associated_descriptors)

            if max_number_of_props_in_entity >= 1:
                candidate_attributes = selected_tag_comb['tag_attributes']
                if len(candidate_attributes) == 0:
                    continue
                max_number_of_props_in_entity = min(len(candidate_attributes), max_number_of_props_in_entity)
                if max_number_of_props_in_entity > 1:
                    selected_num_of_props = np.random.randint(1, max_number_of_props_in_entity)
                else:
                    selected_num_of_props = max_number_of_props_in_entity
                properties = self.generate_properties(candidate_attributes=candidate_attributes,
                                                      num_of_props=selected_num_of_props)
                selected_entities.append(Entity(id=len(selected_entities), name=entity_name, properties=properties))
            else:
                selected_entities.append(Entity(id=len(selected_entities), name=entity_name, properties=[]))

        return selected_entities

    def generate_properties(self, candidate_attributes: List[TagAttribute], num_of_props: int) -> List[Property]:
        candidate_indices = np.arange(len(candidate_attributes))
        np.random.shuffle(candidate_indices)
        selected_indices = candidate_indices[:num_of_props]

        tag_properties = []
        for idx in selected_indices:
            tag_attribute = TagAttribute(**candidate_attributes[idx])
            tag_property = self.property_generator.run(tag_attribute)
            tag_properties.append(tag_property)

        return tag_properties

    # todo make it independent from entities
    def generate_relations(self, num_entities) -> List[Relation]:
        relations = self.relation_generator.run(num_entities=num_entities)
        return relations

    def run(self, area_chance, num_queries) -> List[LocPoint]:
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
        loc_points = []
        for _ in tqdm(range(num_queries), total=num_queries):
            entities = self.generate_entities(max_number_tags_per_query=4, max_number_of_props=4)
            area = self.generate_area(area_chance)
            relations = self.generate_relations(num_entities=len(entities))
            loc_points.append(LocPoint(area=area, entities=entities, relations=relations).dict())
        return loc_points

    def generate_area(self, area_chance) -> Area:
        # todo: change this
        use_area = np.random.choice([True, False], p=[area_chance, 1 - area_chance])

        if use_area:  # Pick random area from list, or default to "bbox"
            area_type = "area"
            selected_areas = np.random.choice(np.asarray([self.countries, self.states, self.cities], dtype=object),
                                              p=[0.05, 0.1, 0.85])
            area_val = np.random.choice(selected_areas)
        else:
            area_type = "bbox"
            area_val = ""
        return Area(type=area_type, value=area_val)


def write_output(generated_combs, output_file):
    with open(output_file, "w") as out_file:
        for generated_comb in generated_combs:
            json.dump(generated_comb, out_file)
            out_file.write('\n')


if __name__ == '__main__':
    '''
    Define paths and run all desired functions.
    '''
    parser = ArgumentParser()
    parser.add_argument('--geolocations_file_path', help='Path to a file containing cities, countries, etc.')
    parser.add_argument('--tag_list_path', help='tag list file generated via retrieve_combinations')
    parser.add_argument('--arbitrary_value_list_path', help='Arbitrary value list generated via combinations')
    parser.add_argument('--output_file', help='File to save the output')
    parser.add_argument('--write_output', action='store_true')
    parser.add_argument('--area_chance', help='Add to probability of picking real area', type=float)
    parser.add_argument('--samples', help='Number of the samples to generate', type=int)
    parser.add_argument('--max_number_tags_per_query', type=int)

    args = parser.parse_args()

    tag_list_path = args.tag_list_path
    arbitrary_value_list_path = args.arbitrary_value_list_path
    geolocations_file_path = args.geolocations_file_path
    area_chance = args.area_chance
    num_samples = args.samples
    output_file = args.output_file
    max_number_tags_per_query = args.max_number_tags_per_query

    query_comb_generator = QueryCombinationGenerator(geolocations_file_path=geolocations_file_path,
                                                     tag_list_path=tag_list_path,
                                                     arbitrary_value_list_path=arbitrary_value_list_path)

    generated_combs = query_comb_generator.run(area_chance=area_chance,
                                               num_queries=num_samples)
    if args.write_output:
        write_output(generated_combs, output_file=output_file)
