import copy
import itertools
import json
import os
import random
from argparse import ArgumentParser

import numpy as np
import openai
import pandas as pd
from datageneration.utils import write_output

# In order to generate the natural language sentences, a valid OpenAI API key is required
openai_info_path = "data/openai_info.json"
if os.path.isfile(openai_info_path):
    with open(openai_info_path, "r") as jsonfile:
        openai_info = json.load(jsonfile)
        openai.organization = openai_info["openai.organization"]
        openai.api_key = openai_info["openai.api_key"]
else:
    print(
        "No JSON file containing OpenAI keys was found. Please provide file or enter info manually to use the OpenAI API.")
    openai.organization = ""
    openai.api_key = ""
    # openai.api_key = os.getenv("OPENAI_API_KEY")


# list = openai.Model.list()


def is_number(s):
    try:
        float(s)  # Try converting the string to a float
        return True
    except ValueError:
        return False


def remove_surrounding_double_quotes(text):
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        return text[1:-1]
    return text


class GPTDataGenerator:
    def __init__(self, tag_list_path, arbitrary_value_list_path, relative_spatial_terms_path, persona_path,
                 styles_path):
        _rel_spatial_terms = pd.read_csv(relative_spatial_terms_path).to_dict(orient='records')

        rel_spatial_terms = {}
        for row in _rel_spatial_terms:
            for rsd in row['Vals'].split(','):
                rel_spatial_terms[rsd.strip()] = row['Dist']
        self.rel_spatial_terms = rel_spatial_terms
        self.rel_spatial_terms_as_words = list(self.rel_spatial_terms.keys())

        with open(persona_path, 'r') as f:
            personas = f.readlines()
            self.personas = list(map(lambda x: x.strip(), personas))
        with open(styles_path, 'r') as f:
            styles = f.readlines()
            self.styles = list(map(lambda x: x.strip(), styles))

    def generate_prompt(self, comb, persona, style):
        '''
        A method that takes the intermediate query representation, and uses it to generate a natural language prompt for
        the GPT API. Different sentence structures are required for the different tasks, for the special tag "count",
        as well as for the different substring searches (beginning, ending, containing, equals).

        :param dict comb: The dictionary containing all relevant information for the query
        '''
        area = comb["a"]["v"]
        objects = comb["ns"]
        distances = comb["es"]

        # personas = ["political journalist", "investigative journalist", "expert fact checker", "hobby fact checker",
        #             "human rights abuse monitoring OSINT Expert", "OSINT beginner", "legal professional"]
        # persona = np.random.choice(personas)
        # styles = ["in perfect grammar and clear wording", "sloppy and quick, with spelling mistakes",
        #           "in simple language",
        #           "like someone in a hurry", "with very precise wording, short, to the point",
        #           "with very elaborate wording",
        #           "as a chain of thoughts split into multiple sentences"]
        # style = np.random.choice(styles)

        beginning = (
                "Act as a " + persona + ": Return a sentence simulating a user using a natural language interface to "
                                        "search for specific geographic locations. Do not affirm this request and return nothing but the "
                                        "answers. \nWrite the search request " + style + ".")

        # ipek- I commented out the below code, validate it
        # style = np.random.choice(["concise or list-style", "well-phrased"], p=[0.5, 0.5])
        # style = style + np.random.choice([", random typos and grammar mistakes", ""], p=[0.25, 0.75])
        # style = style + np.random.choice([", all in one sentence", ", split into multiple sentences"], p=[0.65, 0.35])

        query_phrasing = np.random.choice([0, 1], p=[0.65, 0.35])
        if query_phrasing == 0:
            beginning = beginning + style + "\nThe sentence must use all of the following search criteria:\n"
        elif query_phrasing == 1:
            place_list = ["a street", "a place", "a crossing", "a corner", "an area", "a location"]
            beginning = beginning + style + "\nThe user is searching for " + np.random.choice(
                place_list) + " that fulfills all of the following search criteria:\n"

        core = ""
        if area not in ["bbox", "polygon"]:
            core = "Search area: " + area + "\n"
        object_counter = 0
        for object in objects:
            core = core + "Obj. " + str(object_counter) + ": "
            flts_counter = 0
            for flt in object["flts"]:
                if flts_counter > 0:
                    core = core + ", "
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

                flts_counter += 1

            core = core + "\n"
            object_counter += 1

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
                comb["es"] = distances_
            core = core + core_edge

        prompt = beginning + core

        # ipek - why do we change the comb content??
        return comb, prompt

    def assign_persona_styles_to_queries(self, num_of_all_persona_style, num_tag_queries):
        persona_style_ids = list(range(1, num_of_all_persona_style + 1))
        num_tag_queries_ids = list(range(1, num_tag_queries + 1))

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
        # db_df = pd.DataFrame(columns=['query', 'prompt', 'sentence'])

        # add style, persona generations

        all_possible_persona_and_styles = list(itertools.product(self.personas, self.styles))
        random.shuffle(all_possible_persona_and_styles)
        num_tag_queries = len(tag_queries)
        num_of_all_persona_style = len(all_possible_persona_and_styles)

        self.assign_persona_styles_to_queries(num_of_all_persona_style, num_tag_queries)

        print(all_possible_persona_and_styles)

        # for id, comb in enumerate(comb_list):
        #     # if id >= 50:
        #     #     break
        #     print("Generating ", id + 1, "/", len(comb_list))
        #     comb, prompt = generate_prompt(comb)
        #
        #     # @backoff.on_exception(backoff.expo, openai.error.RateLimitError)
        #     # def completions_with_backoff(**kwargs):
        #     #     '''
        #     #     Helper function to deal with the "openai.error.RateLimitError". If not used, the script will simply
        #     #     stop once the limit is reached, not saving any of the data generated until then. This method will wait
        #     #     and then try again, hence preventing the error.
        #     #
        #     #     :param kwargs: List of arguments passed to the OpenAI API for completion.
        #     #     '''
        #     #     return openai.Completion.create(**kwargs)
        #     #
        #     # response = completions_with_backoff(
        #     #     # model="text-davinci-003",
        #     #     prompt=prompt,
        #     #     temperature=0.9,
        #     #     max_tokens=256,
        #     #     top_p=1.0,
        #     #     frequency_penalty=0.0,
        #     #     presence_penalty=0.0,
        #     #     # stop=["\n"]
        #     #     )
        #     # text = response['choices'][0]['text'].replace('\n', ' ').replace('\r', '').strip()
        #
        #     @backoff.on_exception(backoff.expo, (
        #             openai.error.RateLimitError, openai.error.APIError, openai.error.APIConnectionError,
        #             openai.error.Timeout,
        #             openai.error.ServiceUnavailableError))
        #     def chatcompletions_with_backoff(**kwargs):
        #         '''
        #         Helper function to deal with the "openai.error.RateLimitError". If not used, the script will simply
        #         stop once the limit is reached, not saving any of the data generated until then. This method will wait
        #         and then try again, hence preventing the error.
        #
        #         :param kwargs: List of arguments passed to the OpenAI API for completion.
        #         '''
        #         return openai.ChatCompletion.create(**kwargs)
        #
        #     response = chatcompletions_with_backoff(
        #         model="gpt-3.5-turbo",  # "gpt-4",
        #         temperature=0.9,
        #         max_tokens=1024,
        #         messages=[
        #             {"role": "user", "content": prompt}
        #         ]
        #     )
        #
        #     text = response['choices'][0]['message']['content'].replace('\r', '').strip()
        #     comb_list[id]["text"] = remove_surrounding_double_quotes(text)
        #     # text = ""
        #
        #     print(prompt)
        #     # print(comb_list[id])
        #     print(text)
        #     print("*****")
        #
        #     db_df.loc[id] = [comb, prompt, text]
        #
        #     return generated_queries


# ipek - i construct .env file storing the secret passwords
# def dump_keys():
#     '''
#     Helper function to store the OpenAI API key and organisation parameters as a JSON file for future use.
#     '''
#     team = {}
#     team['openai.organization'] = "YOUR_ORG_HERE"
#     team['openai.api_key'] = "YOUR_KEY_HERE"
#
#     with open('data/openai_info.json', 'w') as f:
#         json.dump(team, f)

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

    gen = GPTDataGenerator(tag_list_path, arbitrary_value_list_path, relative_spatial_terms_path, persona_path,
                           styles_path)
    tag_combinations = pd.read_json(tag_query_file, lines=True)
    generated_queries = gen.run(tag_combinations)
    write_output(generated_queries, output_file)
