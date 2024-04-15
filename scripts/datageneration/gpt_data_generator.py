import copy
import itertools
import json
import os
from argparse import ArgumentParser
from typing import List

import numpy as np
import openai
import pandas as pd
from datageneration.data_model import RelSpatial, LocPoint, Area, Property
from datageneration.utils import write_output
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()

# imports
import random
import time


# define a retry decorator
def retry_with_exponential_backoff(
        func,
        initial_delay: float = 1,
        exponential_base: float = 2,
        jitter: bool = True,
        max_retries: int = 10,
        errors: tuple = (openai.RateLimitError,),
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):
        # Initialize variables
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return func(*args, **kwargs)

            # Retry on specified errors
            except errors as e:
                # Increment retries
                num_retries += 1

                # Check if max retries has been reached
                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )

                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())

                # Sleep for the delay
                time.sleep(delay)

            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e

    return wrapper


@retry_with_exponential_backoff
def chatcompletions_with_backoff(**kwargs):
    '''
    Helper function to deal with the "openai.error.RateLimitError". If not used, the script will simply
    stop once the limit is reached, not saving any of the data generated until then. This method will wait
    and then try again, hence preventing the error.

    :param kwargs: List of arguments passed to the OpenAI API for completion.
    '''
    return CLIENT.chat.completions.create(**kwargs)


# OpenAI parameters
MODEL = os.getenv('MODEL')
TEMPERATURE = float(os.getenv('TEMPERATURE'))
MAX_TOKENS = int(os.getenv('MAX_TOKENS'))

CLIENT = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"], organization=os.environ["OPENAI_ORG"]
)


def request_openai(prompt):
    response = chatcompletions_with_backoff(
        model=MODEL,  # "gpt-4",
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    text = response.choices[0].message.content
    return text


def is_number(s):
    if not s:
        return False
    try:
        float(s)  # Try converting the string to a float
        return True
    except ValueError:
        return False


def remove_surrounding_double_quotes(text):
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        return text[1:-1]
    return text


def post_processing(text):
    text = text.replace('\r', '').strip()
    text = text.replace("User:", "")
    text = remove_surrounding_double_quotes(text)
    return text


def load_rel_spatial_terms(relative_spatial_terms_path: str) -> List[RelSpatial]:
    relative_spatial_terms = pd.read_csv(relative_spatial_terms_path).to_dict(orient='records')
    processed_rel_spatial_terms = []
    for relative_spatial_term in relative_spatial_terms:
        values = list(map(lambda x: x.rstrip().strip(), relative_spatial_term['Vals'].split(',')))
        processed_rel_spatial_terms.append(RelSpatial(distance=relative_spatial_term['Dist'], values=values))
    return processed_rel_spatial_terms


def load_list_of_strings(list_of_strings_path: str) -> List[str]:
    '''
    Helper function for personas and styles for data generation. Loads a list of strings from a text file.
    params:
    list_of_strings_path: Path to the personas or styles text file.
    return:
    list_of_strings: List of strings, either personas or styles.
    '''
    with open(list_of_strings_path, 'r') as f:
        list_of_strings = f.readlines()
        list_of_strings = list(map(lambda x: x.rstrip().strip(), list_of_strings))
    return list_of_strings


class PromptHelper:
    def __init__(self):
        self.beginning_template = """Act as a {persona}: Return a sentence simulating a user using a natural language interface to search for specific geographic locations. Do not affirm this request and return nothing but the answers.\nWrite the search request {style}."""
        self.search_templates = [
            "\nThe sentence must use all of the following search criteria:\n",
            "\nThe user is searching for {place} that fulfills the following search criteria:\n",
        ]
        self.predefined_places = ["a street", "a place", "a crossing", "a corner", "an area", "a location"]

    def beginning(self, persona, writing_style):
        '''
        Create a beginning of a prompt by using beginning template
        '''
        return self.beginning_template.format(persona=persona, style=writing_style)

    def search_query(self, beginning_prompt: str):
        '''
        Append the beginning prompt with search phrase. The search phrase is randomly chosen among the search templates. If search templates contain {place}, it randomly selects a place from predefined_places
        '''
        np.random.shuffle(self.search_templates)
        search_template = self.search_templates[0]

        if '{place}' in search_template:
            np.random.shuffle(self.predefined_places)
            selected_place = self.predefined_places[0]
            beginning_prompt += search_template.replace('{place}', selected_place)
        else:
            beginning_prompt += search_template

        return beginning_prompt

    def add_area_prompt(self, area: Area) -> str:
        '''
        Helper to generate area prompt that is appended to search_prompt
        '''
        area_prompt = ""
        if area.type not in ["bbox", "polygon"]:
            area_prompt = "Search area: " + area.value + "\n"
        return area_prompt

    def add_property_prompt(self, core_prompt: str, entity_properties: List[Property]) -> str:
        for entity_property in entity_properties:
            core_prompt = core_prompt + ", "
            if entity_property['key'] == 'height' or is_number(entity_property['value']):
                pass




            if flt["n"]:
                core = core + flt["n"]
            if flts_counter > 0:
                if flt["op"] == "~":
                    regex_version = np.random.choice([0, 1, 2])
                    if regex_version == 0:
                        core = core + ": " + "contains the letters \"" + flt["v"] + "\""
                    elif regex_version == 1:
                        core = core + ": " + "begins with the letters \"" + flt["v"] + "\""
                    else:
                        core = core + ": " + "ends with the letters \"" + flt["v"] + "\""

                elif is_number(flt["v"]) or flt["k"] == "height":
                    if flt["op"] == "<":
                        lt_list = ["less than", "smaller than", "lower than", "beneath", "under"]
                        core = core + ": " + np.random.choice(lt_list) + " " + flt["v"]
                    elif flt["op"] == ">":
                        gt_list = ["greater than", "more than", "larger than", "above", "over", "at least"]
                        core = core + ": " + np.random.choice(gt_list) + " " + flt["v"]
                    else:
                        core = core + ": " + flt["v"]
                elif flt["k"] in ("building:material", "addr:street", "name", "cuisine"):
                    core = core + ": " + flt["v"]

        return core_prompt


class GPTDataGenerator:
    def __init__(self, relative_spatial_terms: List[RelSpatial], personas: List[str],
                 styles: List[str]):

        self.relative_spatial_terms = relative_spatial_terms
        self.personas = personas
        self.styles = styles
        self.prompt_helper = PromptHelper()

    def generate_prompt(self, loc_point: LocPoint, persona: str, style: str) -> str:
        '''
        A method that takes the intermediate query representation, and uses it to generate a natural language prompt for
        the GPT API. Different sentence structures are required for the different tasks, for the special tag "count",
        as well as for the different substring searches (beginning, ending, containing, equals).

        :param dict loc_point: The dictionary containing all relevant information for the query
        '''

        area = loc_point.area
        entities = loc_point.entities
        relations = loc_point.relations

        beginning = self.prompt_helper.beginning(persona=persona, writing_style=style)
        search_prompt = self.prompt_helper.search_query(beginning)

        core_prompt = self.prompt_helper.add_area_prompt(area)

        for entity_id, entity in enumerate(entities):
            core_prompt = core_prompt + "Obj. " + str(entity_id) + ": " + entity.name
            if len(entity.properties) > 0:
                core_prompt = self.prompt_helper.add_property_prompt(core_prompt=core_prompt,
                                                                     entity_properties=entity.properties)
            core_prompt += '\n'

        print(core_prompt)

        core_edge = ""
        within_dist = False
        if len(distances) > 0:
            dist_counter = 0
            within_dist = True
            distances_ = copy.deepcopy(distances)
            for d in distances_:
                src = d["src"]
                tgt = d["tgt"]
                dist = d["dist"]
                if src != 0:
                    within_dist = False

                # Random draft: Inclusion of relative spatial terms - Load from document and randomly change "dist" in comb, and alter sentence
                rst_chance = 0.4
                use_relative_spatial_terms = np.random.choice([False, True], p=[1.0 - rst_chance, rst_chance])
                if use_relative_spatial_terms:
                    # ipek - i changed the following line
                    # rs_term = np.random.choice(__builtins__.list(self.rel_spatial_terms.keys()), 1)[0]
                    random.shuffle(self.rel_spatial_terms_as_words)
                    rs_term = random.choice(self.rel_spatial_terms_as_words)
                    core_edge += "Use this term to describe the spatial relation between Obj. " + str(
                        src) + " and " + str(
                        tgt) + " (similar to \"X is _ Y\"): " + rs_term + "\n"
                    d["dist"] = str(self.rel_spatial_terms[rs_term])
                else:
                    desc_list = ["", "more or less ", "approximately ", "less than ", "no more than ", "no less than ",
                                 "around ", "at max ", "about ", "at least "]
                    away_list = ["", "", "", "away ", "away from ", "from "]
                    # core_edge += "Distance " + str(dist_counter) + ": Between Obj. " + str(src) + " and " + str(tgt) + ": " + np.random.choice(desc_list) + " " + str(dist) + " " + np.random.choice(away_list) + "\n"
                    core_edge += "Obj. " + str(src) + " is " + np.random.choice(desc_list) + str(
                        dist) + " " + np.random.choice(away_list) + "from Obj. " + str(tgt) + "\n"
                dist_counter += 1

        if within_dist:
            radius_list = ["within " + dist, "in a radius of " + dist, "no more than " + dist + " from each other"]
            core = core + "All objects are " + np.random.choice(radius_list)
        else:
            if len(distances) > 0:
                loc_point["es"] = distances_
            core = core + core_edge

        prompt = beginning + core
        return loc_point, prompt

    def assign_persona_styles_to_queries(self, num_of_all_persona_style, num_tag_queries):
        persona_style_ids = list(range(num_of_all_persona_style))
        num_tag_queries_ids = list(range(num_tag_queries))

        cycled_persona_style_ids = itertools.cycle(persona_style_ids)
        persona_style_tag_pairs = [(x, next(cycled_persona_style_ids)) for x in num_tag_queries_ids]
        return persona_style_tag_pairs

    def run(self, tag_queries):
        '''
        A method that takes the intermediate query representation, and uses it to generate a natural language prompt for
        the GPT API. Different sentence structures are required for the different tasks, for the special tag "count",
        as well as for the different substring searches (beginning, ending, containing, equals).

        :param dict tag_queries: The dictionary containing all relevant information for the query
        '''
        all_possible_persona_and_styles = list(itertools.product(self.personas, self.styles))
        random.shuffle(all_possible_persona_and_styles)
        num_tag_queries = len(tag_queries)
        num_of_all_persona_style = len(all_possible_persona_and_styles)
        persona_style_tag_pairs = self.assign_persona_styles_to_queries(num_of_all_persona_style, num_tag_queries)

        results = []
        for tag_id, persona_style_pair in tqdm(persona_style_tag_pairs, total=num_tag_queries):
            persona, style = all_possible_persona_and_styles[persona_style_pair]
            comb = tag_queries[tag_id]
            comb, prompt = self.generate_prompt(comb, persona, style)

            generated_sentence = request_openai(prompt=prompt)
            post_processed_generated_sentence = post_processing(generated_sentence)
            results.append(
                {'query': comb, 'prompt': prompt, 'style': style, 'persona': persona,
                 'model_output': generated_sentence, 'text': post_processed_generated_sentence})

        return results


if __name__ == '__main__':
    '''
    Define paths and run all desired functions.
    '''
    parser = ArgumentParser()
    parser.add_argument('--tag_list_path', required=True)
    parser.add_argument('--arbitrary_value_list_path', required=True)
    parser.add_argument('--relative_spatial_terms_path', help='Path for the relative spats', required=True)
    parser.add_argument('--tag_query_file', required=True)
    parser.add_argument('--output_file', required=True)
    parser.add_argument('--persona_path', required=True)
    parser.add_argument('--styles_path', required=True)
    args = parser.parse_args()

    tag_list_path = args.tag_list_path
    arbitrary_value_list_path = args.arbitrary_value_list_path
    output_file = args.output_file
    relative_spatial_terms_path = args.relative_spatial_terms_path
    persona_path = args.persona_path
    styles_path = args.styles_path
    tag_query_file = args.tag_query_file

    rel_spatial_terms = load_rel_spatial_terms(relative_spatial_terms_path=relative_spatial_terms_path)
    personas = load_list_of_strings(list_of_strings_path=persona_path)
    styles = load_list_of_strings(list_of_strings_path=styles_path)

    gen = GPTDataGenerator(relative_spatial_terms=rel_spatial_terms,
                           personas=personas,
                           styles_path=styles)

    with open(tag_query_file, "r") as f:
        tag_combinations = [json.loads(each_line) for each_line in f]

    generated_queries = gen.run(tag_combinations)
    write_output(generated_queries, output_file)
