import itertools
import json
import os
from argparse import ArgumentParser
from typing import List

import numpy as np
import openai
import pandas as pd
from datageneration.data_model import RelSpatial, LocPoint, Area, Property, Relation, Relations, GeneratedPrompt, \
    GeneratedIMRSentence
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
MODEL = os.getenv('MODEL', 'gpt-3.5-turbo')
TEMPERATURE = float(os.getenv('TEMPERATURE', 0.7))
MAX_TOKENS = int(os.getenv('MAX_TOKENS', 8000))

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
    '''
    It is a helper class for prompt generation. It has templates and functions for paraphrasing prompts.
    '''

    def __init__(self, relative_spatial_terms):
        self.relative_spatial_terms = relative_spatial_terms
        self.beginning_template = """Act as a {persona}: Return a sentence simulating a user using a natural language interface to search for specific geographic locations. Do not affirm this request and return nothing but the answers.\nWrite the search request {style}."""
        self.search_templates = [
            "\nThe sentence must use all of the following search criteria:\n",
            "\nThe user is searching for {place} that fulfills the following search criteria:\n",
        ]
        self.predefined_places = ["a street", "a place", "a crossing", "a corner", "an area", "a location"]
        self.name_regex_templates = ["contains the letters", "begins with the letters", "ends with the letters"]
        self.phrases_for_numerical_comparison = {
            "<": ["less than", "smaller than", "lower than", "beneath", "under"],
            ">": ["greater than", "more than", "larger than", "above", "over", "at least"]
        }

        self.phrases_desc = ["", "more or less", "approximately", "less than", "no more than", "no less than",
                             "around", "at max", "about", "at least"]

        # todo: the below line does not make sense
        self.phrases_away = ["", "", "", "away", "away from", "from"]
        self.phrases_radius = ["within DIST", "in a radius of DIST", "no more than DIST", "DIST from each other"]

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

    def add_numerical_prompt(self, entity_property: Property) -> str:
        '''
        This helper generates a numerical prompt for numerical properties and properties such as height
        '''
        if entity_property.operator not in self.phrases_for_numerical_comparison:
            return f": {entity_property.value}"
        else:
            numerical_phrases = self.phrases_for_numerical_comparison[entity_property.operator]
            np.random.shuffle(numerical_phrases)
            selected_numerical_phrase = numerical_phrases[0]
            return f": {selected_numerical_phrase} {entity_property.value}"

    def add_name_regex_prompt(self, entity_property: Property) -> str:
        '''
        It is a helper function for name properties such as name, street names
        '''
        np.random.shuffle(self.name_regex_templates)
        selected_name_regex = self.name_regex_templates[0]

        if len(entity_property.value) > 1 and len(selected_name_regex) > 0:
            len_substring = np.random.choice(np.arange(1, len(entity_property.value)))
            idx = random.randrange(0, len(entity_property.value) - len_substring + 1)
            entity_property.value = entity_property.value[idx: (idx + len_substring)].strip()

        return f": {selected_name_regex} \"{entity_property.value}\""

    def add_other_non_numerical_prompt(self, entity_property: Property) -> str:
        '''
        handler for core/attr type of properties having no value and properties such as cuisine
        '''
        return f": {entity_property.value}" if entity_property.value else ""

    def add_property_prompt(self, core_prompt: str, entity_properties: List[Property]) -> str:
        for entity_property in entity_properties:
            core_prompt = core_prompt + ", "
            core_prompt = core_prompt + entity_property.name

            if entity_property.name == 'height' or is_number(entity_property.value):
                core_prompt = core_prompt + self.add_numerical_prompt(entity_property=entity_property)
            elif entity_property.operator == '~':
                core_prompt = core_prompt + self.add_name_regex_prompt(entity_property=entity_property)
            else:
                core_prompt = core_prompt + self.add_other_non_numerical_prompt(entity_property=entity_property)
        return core_prompt

    def add_relative_spatial_terms(self, relation: Relation) -> tuple:
        '''
        Randomly selects relative spatial term
        '''
        np.random.shuffle(self.relative_spatial_terms)
        selected_relative_spatial = self.relative_spatial_terms[0]

        # select randomly descriptor of relative special
        descriptors_of_relative_spatial_terms = selected_relative_spatial.values
        np.random.shuffle(descriptors_of_relative_spatial_terms)
        selected_relative_spatial_term = descriptors_of_relative_spatial_terms[0]
        generated_prompt, overwritten_distance = self.add_relative_spatial_term_helper(
            selected_relative_spatial_term, relation, selected_relative_spatial)

        return (generated_prompt, overwritten_distance)

    def add_relative_spatial_term_helper(self, selected_relative_spatial_term: str, relation: Relation,
                                         selected_relative_spatial: RelSpatial):
        generated_prompt = f"Use this term to describe the spatial relation between Obj. {relation.source} and {relation.target} similar to (similar to \"X is _ Y\"): {selected_relative_spatial_term}\n"
        overwritten_distance = selected_relative_spatial.distance
        return generated_prompt, overwritten_distance

    def add_desc_away_prompt_helper(self, relation: Relation, selected_phrases_desc: str, selected_phrases_away: str):
        '''Helper function for generating desc away prompts'''
        generated_prompt = f"Obj. {relation.source} is {selected_phrases_desc} {relation.value} {selected_phrases_away} Obj. {relation.target}\n"
        return generated_prompt

    def add_desc_away_prompt(self, relation: Relation) -> str:
        np.random.shuffle(self.phrases_desc)
        selected_phrases_desc = self.phrases_desc[0]

        np.random.shuffle(self.phrases_away)
        selected_phrases_away = self.phrases_away[0]

        generated_prompt = self.add_desc_away_prompt_helper(relation, selected_phrases_desc, selected_phrases_away)
        return generated_prompt

    def add_prompt_for_within_radius_relation(self, relations: Relations) -> str:
        np.random.shuffle(self.phrases_radius)
        selected_phrase = self.phrases_radius[0]
        selected_phrase = selected_phrase.replace('DIST', relations.relations[0].value)
        generated_prompt = f"All objects are {selected_phrase}"
        return generated_prompt


class GPTDataGenerator:
    def __init__(self, relative_spatial_terms: List[RelSpatial], personas: List[str],
                 styles: List[str], prob_usage_of_relative_spatial_terms: float = 0.6):

        self.relative_spatial_terms = relative_spatial_terms
        self.prob_usage_of_relative_spatial_terms = prob_usage_of_relative_spatial_terms
        self.personas = personas
        self.styles = styles
        self.prompt_helper = PromptHelper(relative_spatial_terms=relative_spatial_terms)

    def update_relation_distance(self, relations: Relations, relation_to_be_updated: Relation, distance: str):
        updated_relations = []
        for relation in relations.relations:
            if relation == relation_to_be_updated:
                relation.value = distance
                updated_relations.append(relation)
            else:
                updated_relations.append(relation)
        return relations.update(relations=updated_relations)

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

        core_relation = 'Distances:\n'

        if relations.type == "individual_distances":
            for relation in relations.relations:
                rst_chance = self.prob_usage_of_relative_spatial_terms
                use_relative_spatial_terms = np.random.choice([False, True], p=[1.0 - rst_chance, rst_chance])
                if use_relative_spatial_terms:
                    generated_prompt, overwritten_distance = self.prompt_helper.add_relative_spatial_terms(relation)
                    core_relation += generated_prompt
                    self.update_relation_distance(relations=relations,
                                                  relation_to_be_updated=relation,
                                                  distance=overwritten_distance)
                else:
                    core_relation += self.prompt_helper.add_desc_away_prompt(relation)
            core_relation = core_relation[:-1]  # remove trailing linebreak

        elif relations.type == "within_radius":
            core_relation = self.prompt_helper.add_prompt_for_within_radius_relation(relations)
        else:
            # this is the case when there is no relation
            core_relation = ''
            core_prompt = core_prompt[:-1]  # remove trailing linebreak
        core_prompt = core_prompt + core_relation
        core_prompt = search_prompt + core_prompt
        return loc_point, core_prompt

    def assign_persona_styles_to_queries(self, num_of_all_persona_style, num_tag_queries):
        persona_style_ids = list(range(num_of_all_persona_style))
        num_tag_queries_ids = list(range(num_tag_queries))

        cycled_persona_style_ids = itertools.cycle(persona_style_ids)
        persona_style_tag_pairs = [(x, next(cycled_persona_style_ids)) for x in num_tag_queries_ids]
        return persona_style_tag_pairs

    def generate_prompts(self, tag_queries: List[LocPoint]) -> List[GeneratedPrompt]:
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
            imr_sample = tag_queries[tag_id]
            imr_sample, prompt = self.generate_prompt(imr_sample, persona, style)
            results.append(GeneratedPrompt(query=imr_sample, prompt=prompt, style=style, persona=persona))

        return results

    def generate_sentences(self, generated_prompts) -> List[GeneratedIMRSentence]:
        generated_sentences = []
        for generated_prompt in tqdm(generated_prompts, total=len(generated_prompts)):
            generated_sentence = self.generate_sentence(generated_prompt)

            generated_imr_sentence = GeneratedIMRSentence(
                query=generated_prompt.query,
                prompt=generated_prompt.prompt,
                style=generated_prompt.style,
                persona=generated_prompt.persona,
                sentence=generated_sentence
            )
            generated_sentences.append(generated_imr_sentence)
        return generated_sentences

    def generate_sentence(self, generated_prompt: GeneratedPrompt) -> str:
        generated_sentence = request_openai(prompt=generated_prompt.prompt)
        return generated_sentence


if __name__ == '__main__':
    '''
    Define paths and run all desired functions.
    '''
    parser = ArgumentParser()
    parser.add_argument('--relative_spatial_terms_path', help='Path for the relative spats', required=True)
    parser.add_argument('--tag_query_file', required=True)
    parser.add_argument('--output_gpt_generations', required=True)
    parser.add_argument('--output_prompt_generations', required=True)
    parser.add_argument('--persona_path', required=True)
    parser.add_argument('--styles_path', required=True)
    parser.add_argument('--generate_prompts', action='store_true',
                        help='Activate it if you want to generate prompts that will be sent to LLM')
    parser.add_argument('--generate_sentences', action='store_true',
                        help='Activate it if you want to generate sentences with LLM')
    parser.add_argument('--prob_usage_of_relative_spatial_terms', type=float, default=0.4)
    args = parser.parse_args()

    output_prompt_generations = args.output_prompt_generations
    output_gpt_generations = args.output_gpt_generations
    relative_spatial_terms_path = args.relative_spatial_terms_path
    persona_path = args.persona_path
    styles_path = args.styles_path
    tag_query_file = args.tag_query_file
    prob_usage_of_relative_spatial_terms = args.prob_usage_of_relative_spatial_terms
    generate_sentences = args.generate_sentences
    generate_prompts = args.generate_prompts

    rel_spatial_terms = load_rel_spatial_terms(relative_spatial_terms_path=relative_spatial_terms_path)
    personas = load_list_of_strings(list_of_strings_path=persona_path)
    styles = load_list_of_strings(list_of_strings_path=styles_path)

    gen = GPTDataGenerator(relative_spatial_terms=rel_spatial_terms,
                           personas=personas,
                           styles=styles,
                           prob_usage_of_relative_spatial_terms=prob_usage_of_relative_spatial_terms)

    generated_queries = None
    if generate_prompts:
        with open(tag_query_file, "r") as f:
            candidate_loc_points = [LocPoint(**json.loads(each_line)) for each_line in f]
        generated_queries = gen.generate_prompts(candidate_loc_points)
        write_output(generated_queries, output_prompt_generations)

    if generate_sentences:
        if not generated_queries:
            with open(output_prompt_generations, "r") as f:
                generated_queries = [GeneratedPrompt(**json.loads(each_line)) for each_line in f]

            generated_sentences = gen.generate_sentences(generated_queries)
            write_output(generated_sentences, output_gpt_generations)
