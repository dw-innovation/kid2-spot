import json
from argparse import ArgumentParser
from typing import List

import numpy as np
import pandas as pd
from datageneration.area_generator import AreaGenerator, NamedAreaData, load_named_area_data
from datageneration.data_model import TagAttributeExample, TagAttribute, Property, TagCombination, Entity, Relations, \
    LocPoint, Area
from datageneration.property_generator import PropertyGenerator
from datageneration.relation_generator import RelationGenerator
from tqdm import tqdm
from pathlib import Path


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
    def __init__(self, geolocations: List[NamedAreaData], tag_combinations: List[TagCombination],
                 attribute_examples: List[TagAttributeExample], max_distance: int):
        self.entity_tag_combinations = list(filter(lambda x: 'core' in x['comb_type'], tag_combinations))
        self.area_generator = AreaGenerator(geolocations)
        self.property_generator = PropertyGenerator(attribute_examples)
        self.relation_generator = RelationGenerator(max_distance=max_distance)

    def index_to_descriptors(self, index):
        return self.all_tags[int(index)]['descriptors']

    def generate_entities(self, number_of_entities_in_prompt: int, max_number_of_props_in_entity: int) -> List[
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
        selected_entity_numbers = []
        while len(selected_entities) < number_of_entities_in_prompt:
            selected_idx_for_combinations = np.random.randint(0, len(self.entity_tag_combinations))
            if selected_idx_for_combinations in selected_entity_numbers:
                print(selected_entity_numbers, ">", selected_idx_for_combinations)
                continue
            selected_entity_numbers.append(selected_idx_for_combinations)
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
    def generate_relations(self, num_entities) -> Relations:
        relations = self.relation_generator.run(num_entities=num_entities)
        return relations

    def run(self, num_queries, number_of_entities_in_prompt, max_number_of_props_in_entity) -> List[LocPoint]:
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
            new_loc_points = []
            entities = self.generate_entities(number_of_entities_in_prompt=number_of_entities_in_prompt,
                                              max_number_of_props_in_entity=max_number_of_props_in_entity)
            area = self.generate_area()
            relations = self.generate_relations(num_entities=len(entities))
            new_loc_points.append(LocPoint(area=area, entities=entities, relations=relations).dict())

            # Clean up the output and remove all "None" added by the data model (the optional fields)
            for loc_point in new_loc_points:
                for entity in loc_point["entities"]:
                    for property in entity["properties"]:
                        if property["operator"] is None:
                            property.pop('operator', None)
                        if property["value"] is None:
                            property.pop('value', None)

                if loc_point["relations"]["relations"] is None:
                    loc_point["relations"].pop('relations', None)


            loc_points.extend(new_loc_points)

        return loc_points

    def generate_area(self) -> Area:
        return self.area_generator.run()


def write_output(generated_combs, output_file):
    output_Path = Path(output_file)
    output_Path.parent.mkdir(exist_ok=True, parents=True)
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
    parser.add_argument('--tag_combination_path', help='tag list file generated via retrieve_combinations')
    parser.add_argument('--tag_attribute_examples_path', help='Examples of tag attributes')
    parser.add_argument('--output_file', help='File to save the output')
    parser.add_argument('--max_distance', help='Define max distance', type=int)
    parser.add_argument('--write_output', action='store_true')
    parser.add_argument('--samples', help='Number of the samples to generate', type=int)
    parser.add_argument('--max_number_of_props_in_entity', type=int, default=4)
    parser.add_argument('--number_of_entities_in_prompt', type=int, default=4)

    args = parser.parse_args()

    tag_combination_path = args.tag_combination_path
    tag_attribute_examples_path = args.tag_attribute_examples_path
    geolocations_file_path = args.geolocations_file_path
    max_distance = args.max_distance
    num_samples = args.samples
    output_file = args.output_file
    max_number_of_props_in_entity = args.max_number_of_props_in_entity
    number_of_entities_in_prompt = args.number_of_entities_in_prompt

    tag_combinations = pd.read_json(tag_combination_path, lines=True).to_dict('records')
    attribute_examples = pd.read_json(tag_attribute_examples_path, lines=True).to_dict('records')
    geolocations = load_named_area_data(geolocations_file_path)

    query_comb_generator = QueryCombinationGenerator(geolocations=geolocations,
                                                     tag_combinations=tag_combinations,
                                                     attribute_examples=attribute_examples,
                                                     max_distance=args.max_distance)

    generated_combs = query_comb_generator.run(num_queries=num_samples, number_of_entities_in_prompt=4,
                                               max_number_of_props_in_entity=max_number_of_props_in_entity)
    if args.write_output:
        write_output(generated_combs, output_file=output_file)