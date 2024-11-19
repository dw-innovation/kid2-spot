import numpy as np
import pandas as pd
import random
import json
from itertools import combinations
from collections import defaultdict, Counter
from argparse import ArgumentParser

from benchmarking.format_text import format_text
from datageneration.area_generator import AreaGenerator

def generate_instructions(entities, areas, types, styles, typos, grammar_mistakes, relative_spatial_terms,
                          written_numbers, brand, multiple_of_one, names_or_areas_in_non_roman_alphabet,
                          min_items, max_items):
    """
    Function that generates the lists of instructions based on the given parameters.
    Each instruction set includes a few required pieces of information (e.g. number of entities), and zero or multiple
    optional ones.

    :param entities: The options for number of entities.
    :param areas: The options for types of areas (e.g. "City, Country").
    :param types: The options for types of queries (e.g. "within radius").
    :param styles: The options for style instructions (e.g. "Simple language, multiple sentences.").
    :param typos: Options & occurrence count for amount of typos.
    :param grammar_mistakes: Options & occurrence count for amount of grammar mistakes.
    :param relative_spatial_terms: Option & occurrence count for use of relative spatial terms.
    :param written_numbers: Option & occurrence count for use of written numbers.
    :param brand: Options & occurrence count for the use of the name/brand tag bundle.
    :param multiple_of_one: Option & occurrence count for the use of clusters (multiple of one object).
    :param names_or_areas_in_non_roman_alphabet: Options & occurrence count for the use of non-roman alphabets.
    :param min_items: The minimum number of items drawn for each query (in case of zero optional).
    :param max_items: The maximum number of items drawn for each query.
    :return: final_instructions - The final list of written instructions.
    """
    # Dict to keep track of the amount of times each optional instruction was drawn already
    optional_items_counts = {
        'typos_few': typos['few'],
        'typos_many': typos['many'],
        'grammar_few': grammar_mistakes['few'],
        'grammar_many': grammar_mistakes['many'],
        'spatial_yes': relative_spatial_terms['yes'],
        'number_yes': written_numbers['yes'],
        'brand_alone': brand['brand_alone'],
        'brand_type': brand['brand+type'],
        'multiple_of_one': multiple_of_one['yes'],
        'non_roman_area': names_or_areas_in_non_roman_alphabet['non_roman_area'],
        'non_roman_brand': names_or_areas_in_non_roman_alphabet['non_roman_brand']
    }

    def generate_optional_combinations(nonzero_optional_num, zero_optional_num):
        """
        Helper function that generates a list of combinations of required and optional instructions. It checks to make
        sure no combinations with contradicting information is included.

        :param nonzero_optional_num: The no. of times a combination with at least one optional instructions is included.
        :param zero_optional_num: The no. of times a combination with no optional instructions is included.
        :return: The list of combinations.
        """
        buckets = [0, 0, 0, 0]
        all_combinations = []
        # First, draw one of each combination to ensure diversity
        for r in range(min_items - 4, max_items - 4 + 1):
            for comb in combinations(optional_items_counts.keys(), r):
                # Skip combinations with contradicting information
                if (("grammar_few" in comb and "grammar_many" in comb) or ("typos_few" in comb and "typos_many" in comb)
                        or ("brand_alone" in comb and "brand_type" in comb) or
                        ("non_roman_brand" in comb and not any(
                            item in comb for item in ["brand_alone", "brand_type"]))):
                    continue
                all_combinations.append(comb)
                buckets[r] += 1
        # Second, continue drawing combinations until the number of instructions with optional items is met
        for r in range(min_items - 4 + 1, max_items - 4 + 1):
            combs = list(combinations(optional_items_counts.keys(), r))
            while nonzero_optional_num > buckets[r]:
                comb = combs[np.random.choice(np.arange(len(combs)))]
                # Skip combinations with contradicting information
                if (("grammar_few" in comb and "grammar_many" in comb) or ("typos_few" in comb and "typos_many" in comb)
                        or ("brand_alone" in comb and "brand_type" in comb) or
                        ("non_roman_brand" in comb and not any(
                            item in comb for item in ["brand_alone", "brand_type"]))):
                    continue
                all_combinations.append(comb)
                buckets[r] += 1
        # Third, continue drawing combinations until the number of instructions without optional items is met
        while zero_optional_num > buckets[0]:
            all_combinations.append(list(combinations(optional_items_counts.keys(), 0))[0])
            buckets[0] += 1

        random.shuffle(all_combinations)
        return all_combinations

    nonzero_optional_num = 150
    zero_optional_num = 50
    optional_combinations = generate_optional_combinations(nonzero_optional_num, zero_optional_num)

    item_counts = defaultdict(int)

    final_instructions = []

    # Continue generating instructions until all optional item counts are met.
    entity_counter = 0
    area_counter = 0
    type_counter = 0
    style_counter = 0
    while not all(item_counts[item] == optional_items_counts[item] for item in optional_items_counts
                  if item != "non_roman_brand"):
        for comb in optional_combinations:
            def draw_vals():
                """
                Helper function that draws the required instructions based on the current counter.

                :return: entity, area, type, style - The drawn values of the required pieces of information.
                """
                entity = entities[entity_counter]
                area = areas[area_counter]
                type = types[type_counter]
                style = styles[style_counter]

                return entity, area, type, style

            # Draw combinations based on counters, check if combinations makes logical sense, otherwise update counter
            # that causes contradiction and redraw.
            entity, area, type, style = draw_vals()
            while ((type in ["individual_distances",
                             "individual_distances_with_contains"] and entity != "3 Entities") or
                   (type in ["within_radius", "contains_relation"] and entity == "1 Entity") or
                   (type in ["within_radius", "in_area", "contains"] and "spatial_yes" in comb) or
                   (area == "No Area" and "non_roman_area" in comb)):

                if (type in ["within_radius", "in_area", "contains"] and "spatial_yes" in comb):
                    type_counter = (type_counter + 1) % len(types)
                elif (area == "No Area" and "non_roman_area" in comb):
                    area_counter = (area_counter + 1) % len(areas)
                else:
                    entity_counter = (entity_counter + 1) % len(entities)

                entity, area, type, style = draw_vals()

            # Add combination if it does not exceed the predefined optional item count
            if all(item_counts[item] < optional_items_counts[item] for item in comb):
                inst_ = (entity, area, type, style) + comb
                final_instructions.append(inst_)
                for item in comb:
                    item_counts[item] += 1

            # Loop through the required combinations in a way that generates all possible combinations
            entity_counter = (entity_counter + 1) % len(entities)
            area_counter = (area_counter + 1) % len(areas)
            type_counter = (type_counter + 2) % len(types)
            style_counter = (style_counter + 1) % len(styles)

    # Loop over the instructions and add "non_roman_brand" randomly to combinations where brand is used.
    for iid, instruction in enumerate(final_instructions):
        if (item_counts["non_roman_brand"] < optional_items_counts["non_roman_brand"] and
                len(instruction) < max_items and any(item in instruction for item in ["brand_alone", "brand_type"])
                and "non_roman_brand" not in instruction):
            final_instructions[iid] = instruction + ("non_roman_brand",)
            item_counts["non_roman_brand"] += 1

    random.shuffle(final_instructions)

    return final_instructions

def add_values(instructions, geolocations_file_path):
    """
    Function to add values to the drawn instructions. The only value left that truly needs to be drawn directly is
    the area name. The other values required are just the numbers of entities and properties of the query.

    :param instructions: The instructions drawn in the previous step.
    :param geolocations_file_path: The path to the geolocation file.
    :return: instructions_with_values - The previous instructions plus the newly drawn information.
    """
    area_generator = AreaGenerator(geolocation_file=geolocations_file_path, percentage_of_two_word_areas=0.5)

    add_properties = {"1_property": 55, "2_properties": 45, "3_properties": 20}
    add_properties = [id+1 for id, val in enumerate(add_properties.values()) for _ in range(val)]
    np.random.shuffle(add_properties)

    instructions_with_values = []
    for inst_id, instruction in enumerate(instructions):
        print("Generating ", inst_id+1, "/", len(instructions))
        num_entities = int(instruction[0][0]) # Number of entities

        ents_with_props = 0
        num_props = 0
        for _ in range(num_entities):
            if len(add_properties) > 0 and np.random.choice([True, False], p=[0.4, 0.6]):
                ents_with_props += 1
                num_props += add_properties.pop(-1)

        if instruction[1] == "No Area":
            area = area_generator.generate_no_area()
        elif instruction[1] == "City":
            area = area_generator.generate_city_area()
        elif instruction[1] == "Region":
            area = area_generator.generate_region_area()
        elif instruction[1] == "City, Country":
            area = area_generator.generate_city_and_country_area()
        elif instruction[1] == "Region, Country":
            area = area_generator.generate_region_and_country_area()
        elif instruction[1] == "City, Region, Country":
            area = area_generator.generate_city_and_region_and_country_area()

        padded_instruction = list(instruction) + [""] * (9 - len(instruction))

        instructions_with_values.append(list(padded_instruction) + [area, num_entities, ents_with_props, num_props])

    return instructions_with_values

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--geolocations_file_path', help='Path to a file containing cities, countries, etc.')
    args = parser.parse_args()

    geolocations_file_path = args.geolocations_file_path

    # Define the required instructions. One of each must be drawn for every query.
    entities = ["1 Entity", "2 Entities", "3 Entities"]  #3
    areas = ["No Area", "No Area", "City", "Region", "City, Country", "Region, Country", "City, Region, Country"]  #7
    types = ["in_area", "within_radius", "individual_distances", "contains_relation",
             "individual_distances_with_contains"]  #5
    styles = ["Simple language, all in one sentence.", "Simple language, multiple sentences.",
              "Elaborate wording, all in one sentence.", "Elaborate wording, multiple sentences.",
              "Short and precise, all in one sentence.", "Short and precise, multiple sentences."] #6

    # Define the optional instructions. They can optionally be drawn for a query, and occur a limited number of times.
    typos = {"few": 50, "many": 50}
    grammar_mistakes = {"few": 50, "many": 50}
    relative_spatial_terms = {"yes": 100}
    written_numbers = {"yes": 100}
    brand = {"brand_alone": 50, "brand+type": 50}
    multiple_of_one = {"yes": 100}
    names_or_areas_in_non_roman_alphabet = {"non_roman_area": 50, "non_roman_brand": 50}
    # typos = {"few": 2, "many": 2}
    # grammar_mistakes = {"few": 2, "many": 2}
    # relative_spatial_terms = {"yes": 5}
    # written_numbers = {"yes": 5}
    # brand = {"brand_alone": 2, "brand+type": 2}
    # multiple_of_one = {"yes": 5}
    # names_or_areas_in_non_roman_alphabet = {"non_roman_area": 2, "non_roman_brand": 2}

    min_items = 4
    max_items = 7

    instructions = generate_instructions(entities, areas, types, styles, typos, grammar_mistakes,
                                         relative_spatial_terms, written_numbers, brand, multiple_of_one,
                                         names_or_areas_in_non_roman_alphabet, min_items, max_items)  #

    count_result = Counter()
    for inst in instructions:
        count_result.update(inst)

    instructions_with_values = add_values(instructions, geolocations_file_path)

    instructions_with_formatted_text = format_text(instructions_with_values)

    column_names = ["entities", "areas", "types", "styles", "optional 1", "optional 2", "optional 3", "optional 4",
                    "optional 5", "instructions"]

    df = pd.DataFrame(instructions_with_formatted_text, columns=column_names)
    df.to_csv('benchmarking/results/instructions.csv', index=False, header=True)

    instructions_for_json = []
    with open("benchmarking/results/instructions.json", "w") as out_file:
        for id, inst in enumerate(instructions_with_formatted_text):
            json.dump({"id": "doc_" + str(id), "text": inst[-1]}, out_file)
            out_file.write('\n')

    print("\nOutput written to file!")
