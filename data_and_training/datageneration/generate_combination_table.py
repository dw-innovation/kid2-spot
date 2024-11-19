import copy
import json
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict

from datageneration.area_generator import AreaGenerator, NamedAreaData, load_named_area_data
from datageneration.data_model import TagPropertyExample, TagProperty, Property, TagCombination, Entity, Relations, \
    LocPoint, Area
from datageneration.property_generator import PropertyGenerator, fetch_color_bundle
from datageneration.relation_generator import RelationGenerator
from datageneration.utils import get_random_decimal_with_metric, write_output


class QueryCombinationGenerator(object):
    def __init__(self, geolocation_file: str,
                 non_roman_vocab_file: str,
                 tag_combinations: List[TagCombination],
                 property_examples: List[TagPropertyExample],
                 max_distance_digits: int,
                 prob_of_two_word_areas: float,
                 prob_generating_contain_rel: float,
                 prob_adding_brand_names_as_entity: float,
                 prob_of_numerical_properties: float,
                 prob_of_color_properties: float,
                 prob_of_popular_non_numerical_properties: float,
                 prob_of_other_non_numerical_properties: float,
                 prob_of_rare_non_numerical_properties: float,
                 prob_of_non_roman_areas: float,
                 color_bundle_path: str,
                 prob_of_cluster_entities: float
                 ):

        color_bundles = fetch_color_bundle(property_examples=property_examples,bundle_path=color_bundle_path)
        self.property_generator = PropertyGenerator(property_examples, color_bundles=color_bundles)
        self.entity_tag_combinations = self.categorize_entities_based_on_their_props(list(filter(lambda x: 'core' in x.comb_type.value, tag_combinations)))

        self.area_generator = AreaGenerator(geolocation_file=geolocation_file, non_roman_vocab_file=non_roman_vocab_file, prob_of_two_word_areas=prob_of_two_word_areas, prob_of_non_roman_areas=prob_of_non_roman_areas)
        self.prob_adding_brand_names_as_entity = prob_adding_brand_names_as_entity
        self.relation_generator = RelationGenerator(max_distance_digits=max_distance_digits,
                                                    prob_generating_contain_rel=prob_generating_contain_rel)
        self.prob_of_numerical_properties = prob_of_numerical_properties
        self.prob_of_color_properties = prob_of_color_properties
        self.prob_of_popular_non_numerical_properties = prob_of_popular_non_numerical_properties
        self.prob_of_other_non_numerical_properties = prob_of_other_non_numerical_properties
        self.prob_of_rare_non_numerical_properties = prob_of_rare_non_numerical_properties
        self.prob_of_cluster_entities = prob_of_cluster_entities
        self.all_properties_with_probs = {
            "numerical": self.prob_of_numerical_properties,
            "colour": self.prob_of_color_properties,
            'rare_non_numerical': self.prob_of_rare_non_numerical_properties,
            "popular_non_numerical": self.prob_of_popular_non_numerical_properties,
            "other_non_numerical": self.prob_of_other_non_numerical_properties,
        }

    def categorize_entities_based_on_their_props(self, tag_combinations: List[TagCombination]) -> Dict:
        categorized_entities = {
            'numerical': [],
            'colour': [],
            'rare_non_numerical': [],
            'popular_non_numerical': [],
            'other_non_numerical': [],
            'default': [] # add every type of entities here
        }
        for tag_combination in tag_combinations:
            tag_properties = tag_combination.tag_properties
            categorized_entities['default'].append(tag_combination)
            prop_categories = self.property_generator.categorize_properties(tag_properties=tag_properties)
            for prop_key in prop_categories.keys():
                categorized_entities[prop_key].append(tag_combination)
        return categorized_entities

    def get_number_of_entities(self, max_number_of_entities_in_prompt: int) -> int:
        """
        This method of selecting the number of entities uses an exponential decay method that returns
        a probability distribution that has a peak probability value, from which the probabilities decrease
        towards both sides. The decay rate can be customised per side to allow for the selection of a higher
        decay rate towards the left to minimise one and two entity samples. This is due to the fact that these
        queries don't have sufficient entities to assign all three query types and should hence occur less in the
        training data.

        Example probability distribution with 4 ents, peak 3, decay left 0.3, decay right 0.4:
            [0.0993, 0.1999, 0.4026, 0.2982]

        :param max_number_of_entities_in_prompt: The maximum allowed number of entities per query
        :return: The selected number of entities
        """
        peak_value = 3  # Number of entity with the highest probability
        decay_rate_right = 0.7
        decay_rate_left = 0.5 #0.3
        entity_nums = np.arange(1, max_number_of_entities_in_prompt + 1)
        probabilities = np.zeros(max_number_of_entities_in_prompt)
        probabilities[peak_value - 1] = 1
        probabilities[peak_value:] = np.exp(-decay_rate_right * (entity_nums[peak_value:] - peak_value))
        probabilities[:peak_value] = np.exp(-decay_rate_left * (peak_value - entity_nums[:peak_value]))
        probabilities /= np.sum(probabilities)
        number_of_entities_in_prompt = np.random.choice(entity_nums, p=probabilities)

        return number_of_entities_in_prompt

    def get_number_of_props(self, max_number_of_props_in_entity: int):
        """
        This method of selecting the number of properties uses an exponential decay method that returns
        a probability distribution that assigns higher probabilities to lower values, as many entities
        with multiple properties will result in convoluted sentence

        Example probability distribution with 4 props & decay of 0.3: [0.3709, 0.2748, 0.2036, 0.1508]

        :param max_number_of_props_in_entity: The maximum allowed number of properties per entity
        :return: The selected number of properties
        """
        decay_rate = 0.3
        prop_nums = np.arange(1, max_number_of_props_in_entity + 1)
        probabilities = np.exp(-decay_rate * prop_nums)
        probabilities /= np.sum(probabilities)
        selected_num_of_props = np.random.choice(prop_nums, p=probabilities)

        return selected_num_of_props

    def add_cluster_entities(self, selected_entities):
        for id, entity in enumerate(selected_entities):
            add_cluster = np.random.choice([True, False], p=[self.prob_of_cluster_entities,
                                                                     1 - self.prob_of_cluster_entities])
            if add_cluster:
                minPoints = np.random.choice(np.arange(20))
                maxDistance = get_random_decimal_with_metric(5)
                selected_entities[id] = Entity(id=selected_entities[id].id, is_area=selected_entities[id].is_area,
                                                name=selected_entities[id].name, type='cluster',
                                                minPoints=minPoints, maxDistance=maxDistance, properties=[])

        return selected_entities

    def generate_entities(self, max_number_of_entities_in_prompt: int, max_number_of_props_in_entity: int,
                          prob_of_entities_with_props: float) -> List[Entity]:
        """
        Generates a list of entities with associated properties based on random selection of descriptors.

        Args:
            max_number_of_entities_in_prompt (int): Number of entities to generate.
            max_number_of_props_in_entity (int): Maximum number of properties each entity can have.
            prob_of_entities_with_props (float): Ratio of entities that have a non-zero number of properties

        Returns:
            List[Entity]: A list of generated entities with associated properties.

        Note:
            - The function selects a random subset of descriptors from the available combinations of entity tag combinations.
            - Each entity is assigned a random name chosen from the selected descriptors.
            - If `max_number_of_props_in_entity` is greater than or equal to 1, properties are generated for each entity.
              Otherwise, entities are generated without properties.
        """
        number_of_entities_in_prompt = self.get_number_of_entities(max_number_of_entities_in_prompt)
        selected_entities = []
        selected_tag_combs = []
        while len(selected_entities) < number_of_entities_in_prompt:
            selected_brand_name = np.random.choice([True, False], p=[self.prob_adding_brand_names_as_entity,
                                                                     1 - self.prob_adding_brand_names_as_entity])
            if not selected_brand_name:
                add_properties = np.random.choice([True, False], p=[prob_of_entities_with_props,
                                                                1 - prob_of_entities_with_props])
                if add_properties and max_number_of_props_in_entity >= 1:
                    selected_property_category = np.random.choice(list(self.all_properties_with_probs.keys()),
                                                                  p=list(self.all_properties_with_probs.values()))

                    selected_idx_for_combinations = np.random.randint(0, len(self.entity_tag_combinations[selected_property_category]))
                    selected_tag_comb = self.entity_tag_combinations[selected_property_category][selected_idx_for_combinations]
                    is_area = selected_tag_comb.is_area

                    if selected_tag_comb in selected_tag_combs:
                        continue

                    selected_tag_combs.append(selected_tag_comb)
                    associated_descriptors = selected_tag_comb.descriptors

                    entity_name = np.random.choice(associated_descriptors)
                    candidate_properties = selected_tag_comb.tag_properties

                    if len(candidate_properties) == 0:
                        continue

                    current_max_number_of_props = min(len(candidate_properties), max_number_of_props_in_entity)
                    if current_max_number_of_props > 1:
                        # selected_num_of_props = np.random.randint(1, max_number_of_props_in_entity)
                        selected_num_of_props = self.get_number_of_props(current_max_number_of_props)
                    else:
                        selected_num_of_props = current_max_number_of_props

                    # print('selected tag prompt')
                    # print(selected_tag_comb)
                    # print('candidate properties')
                    # print(candidate_properties)
                    properties = self.generate_properties(candidate_properties=candidate_properties,
                                                          num_of_props=selected_num_of_props)
                    selected_entities.append(
                        Entity(id=len(selected_entities), is_area=is_area, name=entity_name, properties=properties))
                else:
                    selected_idx_for_combinations = np.random.randint(0, len(self.entity_tag_combinations['default']))
                    selected_tag_comb = self.entity_tag_combinations['default'][selected_idx_for_combinations]
                    associated_descriptors = selected_tag_comb.descriptors
                    entity_name = np.random.choice(associated_descriptors)

                    is_area = selected_tag_comb.is_area
                    if selected_tag_comb in selected_tag_combs:
                        continue
                    selected_tag_combs.append(selected_tag_comb)
                    selected_entities.append(
                        Entity(id=len(selected_entities), is_area=is_area, name=entity_name, properties=[]))
            else:
                brand_examples = self.property_generator.select_named_property_example("brand~***example***")
                entity_name = f"brand:{np.random.choice(brand_examples)}"
                is_area = False
                selected_entities.append(
                    Entity(id=len(selected_entities), is_area=is_area, name=entity_name, properties=[]))

        selected_entities = self.add_cluster_entities(selected_entities)

        return selected_entities

    def generate_properties(self, candidate_properties: List[TagProperty], num_of_props: int, trial_err_count=100) -> List[Property]:
        categorized_properties = self.property_generator.categorize_properties(candidate_properties)
        all_property_categories = list(self.all_properties_with_probs.keys())
        all_properties_with_probs = self.all_properties_with_probs

        new_all_property_categories = [
            category for category in all_property_categories
            if all_properties_with_probs.get(category) != 0.0 and category in categorized_properties
        ]
        new_all_property_category_probs = {
            category: prob for category, prob in all_properties_with_probs.items()
            if prob != 0.0 and category in categorized_properties
        }

        all_property_categories = new_all_property_categories
        all_property_category_probs = new_all_property_category_probs

        all_property_category_probs_values = list(all_property_category_probs.values())
        tag_properties = []
        tag_properties_keys = []

        trial_err = 0
        while(len(tag_properties)<num_of_props):
            if trial_err == trial_err_count:
                return tag_properties
            trial_err += 1
            if sum(all_property_category_probs_values) != 1:
                remaining_prob = (1- sum(all_property_category_probs_values)) / len(all_property_category_probs_values)
                all_property_category_probs_values = list(map(lambda x: x+remaining_prob, all_property_category_probs_values))


            selected_property_category = np.random.choice(all_property_categories, p=all_property_category_probs_values)
            selected_category_properties = categorized_properties[selected_property_category]
            candidate_indices = np.arange(len(selected_category_properties))
            np.random.shuffle(candidate_indices)
            selected_index = candidate_indices[0]
            tag_property = selected_category_properties[selected_index]
            tag_props_key = ' '.join(tag_property.descriptors)
            # if tag_props_key in tag_properties_keys and tag_props_key not in ['cuisine', 'sport']:
                # we keep cuisine, sport because facilities can serve multiple cuisine, and offer different sport activities
                # continue
            tag_properties_keys.append(tag_props_key)
            tag_property = self.property_generator.run(tag_property)
            tag_properties.append(tag_property)

        return tag_properties

    # todo make it independent from entities
    def generate_relations(self, entities: List[Entity]) -> Relations:
        relations = self.relation_generator.run(entities=entities)
        return relations

    def sort_entities(self, entities: List[Entity], relations: Relations) -> (List[Entity], Relations):
        """
        In the process of selecting areas and points that are in a "contains" relations with another, the IDs in
        the IMR can become fairly messy, as the random entity selection does not select based on area or point entities.
        This sorting step is performed to generate a uniform output (contains relations before distance relations,
        always first the area and then all the contained points). It puts the entities in the correct order and
        adjusts the IDs.

        :param entities: The entities of the query
        :param relations: The relations of the query
        :return: The sorted entities and relations
        """
        sorted_relations = copy.deepcopy(relations)
        sorted_relations.relations = sorted(relations.relations, key=lambda r: (min(r.source, r.target), max(r.source, r.target)))

        return entities, sorted_relations

        # sorted_entities = []
        # sorted_relations = copy.deepcopy(relations)
        # lookup_table = dict()
        # id = 0
        # # Loop over all relations, which must be in the order that "contains" relations come first.
        # for relation in relations.relations:
        #     # If the "source" (area) is not yet known, add that first
        #     if entities[relation.source] not in sorted_entities:
        #         sorted_entities.append(entities[relation.source])
        #         sorted_entities[-1].id = id
        #         lookup_table[relation.source] = id
        #         id += 1
        #     # After the "source" (area) was added, add all their "targets" (points contained within)
        #     if entities[relation.target] not in sorted_entities:
        #         sorted_entities.append(entities[relation.target])
        #         sorted_entities[-1].id = id
        #         lookup_table[relation.target] = id
        #         id += 1
        #
        # # Update the relations based on lookup table to match with the new entity IDs
        # for sorted_relation in sorted_relations.relations:
        #     sorted_relation.source = lookup_table[sorted_relation.source]
        #     sorted_relation.target = lookup_table[sorted_relation.target]
        #
        # return sorted_entities, sorted_relations

    def run(self, num_queries: int, max_number_of_entities_in_prompt: int, max_number_of_props_in_entity: int,
            prob_of_entities_with_props: float) -> List[LocPoint]:
        '''
        A method that generates random query combinations and optionally saves them to a JSON file.
        It gets a list of random tag combinations and adds additional information that is required to generate
        full queries, including area names, and different search tasks.
        The current search tasks are: (1) individual distances: a random specific distance is defined between all objects,
        (2) within radius: a single radius within which all objects are located, (3) in area: general search for all objects
        within given area.

        TODO: Write/update all docstrings, maybe use this text somewhere else

        :param num_queries: (int) TODO
        :param max_number_of_entities_in_prompt: (int) TODO
        :param max_number_of_props_in_entity: (int) TODO
        :param prob_of_entities_with_props: (int) TODO
        :param percentage_of_entities_with_props: (float) TODO
        :return: loc_points (List[LocPoint])
        '''
        loc_points = []
        for _ in tqdm(range(num_queries), total=num_queries):
            area = self.generate_area()
            entities = self.generate_entities(max_number_of_entities_in_prompt=max_number_of_entities_in_prompt,
                                              max_number_of_props_in_entity=max_number_of_props_in_entity,
                                              prob_of_entities_with_props=prob_of_entities_with_props)
            relations = self.generate_relations(entities=entities)

            if relations.type in ["individual_distances_with_contains", "contains_relation"]:
                sorted_entities, sorted_relations = self.sort_entities(entities, relations)
                loc_points.append(LocPoint(area=area, entities=sorted_entities, relations=sorted_relations))
            else:
                loc_points.append(LocPoint(area=area, entities=entities, relations=relations))

        return loc_points

    def generate_area(self) -> Area:
        return self.area_generator.run()


if __name__ == '__main__':
    '''
    Define paths and run all desired functions.
    '''
    parser = ArgumentParser()
    parser.add_argument('--geolocations_file_path', help='Path to a file containing cities, countries, etc.')
    parser.add_argument('--non_roman_vocab_file_path', help='Path to a file containing a vocabulary of areas with non-roman alphabets')
    parser.add_argument('--tag_combination_path', help='tag list file generated via retrieve_combinations')
    parser.add_argument('--tag_prop_examples_path', help='Examples of tag properties')
    parser.add_argument('--color_bundle_path', help='Path to color bundles')
    parser.add_argument('--output_file', help='File to save the output')
    parser.add_argument('--max_distance_digits', help='Define max distance', type=int)
    parser.add_argument('--write_output', action='store_true')
    parser.add_argument('--samples', help='Number of the samples to generate', type=int)
    parser.add_argument('--max_number_of_entities_in_prompt', type=int, default=4)
    parser.add_argument('--max_number_of_props_in_entity', type=int, default=4)
    parser.add_argument('--prob_of_entities_with_props', type=float, default=0.3)
    parser.add_argument('--prob_of_non_roman_areas', type=float, default=0.2)
    parser.add_argument('--prob_of_two_word_areas', type=float, default=0.5)
    parser.add_argument('--prob_adding_brand_names_as_entity', type=float, default=0.5)
    parser.add_argument('--prob_generating_contain_rel', type=float, default=0.3)
    parser.add_argument('--prob_of_rare_non_numerical_properties', type=float, default=0.2)
    parser.add_argument('--prob_of_numerical_properties', type=float, default=0.3)
    parser.add_argument('--prob_of_color_properties', type=float, default=0.0)
    parser.add_argument('--prob_of_popular_non_numerical_properties', type=float, default=0.2)
    parser.add_argument('--prob_of_other_non_numerical_properties', type=float, default=0.5)
    parser.add_argument('--prob_of_cluster_entities', type=float, default=0.3)

    args = parser.parse_args()

    tag_combination_path = args.tag_combination_path
    tag_prop_examples_path = args.tag_prop_examples_path
    geolocations_file_path = args.geolocations_file_path
    color_bundle_path = args.color_bundle_path
    non_roman_vocab_file_path = args.non_roman_vocab_file_path
    max_distance_digits = args.max_distance_digits
    num_samples = args.samples
    output_file = args.output_file
    max_number_of_entities_in_prompt = args.max_number_of_entities_in_prompt
    max_number_of_props_in_entity = args.max_number_of_props_in_entity
    prob_of_entities_with_props = args.prob_of_entities_with_props
    prob_of_two_word_areas = args.prob_of_two_word_areas
    prob_generating_contain_rel = args.prob_generating_contain_rel
    prob_of_non_roman_areas = args.prob_of_non_roman_areas
    prob_adding_brand_names_as_entity = args.prob_adding_brand_names_as_entity
    prob_of_numerical_properties = args.prob_of_numerical_properties
    prob_of_color_properties = args.prob_of_color_properties
    prob_of_other_non_numerical_properties = args.prob_of_other_non_numerical_properties
    prob_of_popular_non_numerical_properties = args.prob_of_popular_non_numerical_properties
    prob_of_rare_non_numerical_properties = args.prob_of_rare_non_numerical_properties
    prob_of_cluster_entities = args.prob_of_cluster_entities

    tag_combinations = pd.read_json(tag_combination_path, lines=True).to_dict('records')
    tag_combinations = [TagCombination(**tag_comb) for tag_comb in tag_combinations]
    property_examples = pd.read_json(tag_prop_examples_path, lines=True).to_dict('records')

    query_comb_generator = QueryCombinationGenerator(geolocation_file=geolocations_file_path,
                                                     non_roman_vocab_file=non_roman_vocab_file_path,
                                                     color_bundle_path=color_bundle_path,
                                                     tag_combinations=tag_combinations,
                                                     property_examples=property_examples,
                                                     max_distance_digits=args.max_distance_digits,
                                                     prob_of_two_word_areas=prob_of_two_word_areas,
                                                     prob_of_non_roman_areas=prob_of_non_roman_areas,
                                                     prob_generating_contain_rel=prob_generating_contain_rel,
                                                     prob_adding_brand_names_as_entity=prob_adding_brand_names_as_entity,
                                                     prob_of_numerical_properties=prob_of_numerical_properties,
                                                     prob_of_color_properties=prob_of_color_properties,
                                                     prob_of_popular_non_numerical_properties=prob_of_popular_non_numerical_properties,
                                                     prob_of_other_non_numerical_properties= prob_of_other_non_numerical_properties,
                                                     prob_of_rare_non_numerical_properties=prob_of_rare_non_numerical_properties,
                                                     prob_of_cluster_entities=prob_of_cluster_entities)

    generated_combs = query_comb_generator.run(num_queries=num_samples,
                                               max_number_of_entities_in_prompt=max_number_of_entities_in_prompt,
                                               max_number_of_props_in_entity=max_number_of_props_in_entity,
                                               prob_of_entities_with_props=prob_of_entities_with_props)

    if args.write_output:
        write_output(generated_combs, output_file=output_file)
