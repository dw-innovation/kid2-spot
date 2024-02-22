import os
import numpy as np
import pandas as pd
import json
import openai
import backoff
import copy

from generate_combination_table import NpEncoder, generate_query_combinations

# In order to generate the natural language sentences, a valid OpenAI API key is required
openai_info_path = "data/openai_info.json"
if os.path.isfile(openai_info_path):
    with open(openai_info_path, "r") as jsonfile:
        openai_info = json.load(jsonfile)
        openai.organization = openai_info["openai.organization"]
        openai.api_key = openai_info["openai.api_key"]
else:
    print("No JSON file containing OpenAI keys was found. Please provide file or enter info manually to use the OpenAI API.")
    openai.organization = ""
    openai.api_key = ""
    # openai.api_key = os.getenv("OPENAI_API_KEY")

list = openai.Model.list()

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

def generate_prompt(comb, relspat):
    '''
    A method that takes the intermediate query representation, and uses it to generate a natural language prompt for
    the GPT API. Different sentence structures are required for the different tasks, for the special tag "count",
    as well as for the different substring searches (beginning, ending, containing, equals).

    :param dict comb: The dictionary containing all relevant information for the query
    '''
    area = comb["a"]["v"]
    objects = comb["ns"]
    distances = comb["es"]

    beginning = ("Return a sentence simulating a user using a natural language interface to search for specific "
                 "geographic locations. Do not affirm this request and return nothing but the answers. \n"
                 "Instructions for the style of the query: ")

    style = np.random.choice(["concise or list-style", "well-phrased"], p=[0.5, 0.5])
    style = style + np.random.choice([", random typos and grammar mistakes", ""], p=[0.25, 0.75])
    style = style + np.random.choice([", all in one sentence", ", split into multiple sentences"], p=[0.65, 0.35])

    query_phrasing = np.random.choice([0, 1], p=[0.65, 0.35])
    if query_phrasing == 0:
        beginning = beginning +  style + "\nThe sentence must use all of the following search criteria:\n"
    elif query_phrasing == 1:
        place_list = ["a street", "a place", "a crossing", "a corner", "an area", "a location"]
        beginning = beginning +  style + "\nThe user is searching for " + np.random.choice(place_list) + " that fulfills all of the following search criteria:\n"

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
                    regex_version = np.random.choice([0,1,2])
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
                rs_term = np.random.choice(__builtins__.list(relspat.keys()), 1)[0]

                core_edge += "Use this term to describe the spatial relation between Obj. " + str(src) + " and " + str(tgt) + " (similar to \"X is _ Y\"): " + rs_term + "\n"
                d["dist"] = str(relspat[rs_term])
            else:
                desc_list = ["", "more or less ", "approximately ", "less than ", "no more than ", "no less than ", "around ", "at max ", "about ", "at least "]
                away_list = ["", "", "", "away ", "away from ", "from "]
                # core_edge += "Distance " + str(dist_counter) + ": Between Obj. " + str(src) + " and " + str(tgt) + ": " + np.random.choice(desc_list) + " " + str(dist) + " " + np.random.choice(away_list) + "\n"
                core_edge += "Obj. " + str(src) + " is " + np.random.choice(desc_list) + str(dist) + " " + np.random.choice(away_list) + "from Obj. " + str(tgt) + "\n"
            dist_counter += 1

    if within_dist:
        radius_list = ["within " + dist, "in a radius of " + dist, "no more than " + dist + " from each other"]
        core = core + "All objects are " + np.random.choice(radius_list)
    else:
        if len(distances) > 0:
            comb["es"] = distances_
        core = core + core_edge

    prompt = beginning + core

    return comb, prompt


def query(comb_list, output_filename, relspat, version="train", save_files=True):
    '''
    A method that takes the intermediate query representation, and uses it to generate a natural language prompt for
    the GPT API. Different sentence structures are required for the different tasks, for the special tag "count",
    as well as for the different substring searches (beginning, ending, containing, equals).

    :param dict comb_list: The dictionary containing all relevant information for the query
    :param str output_filename: Name of the in- and output file (minus version specification)
    :param bool save_files: Boolean determining whether the result should be saved as a file
    '''
    db_df = pd.DataFrame(columns=['query', 'prompt', 'sentence'])

    for id, comb in enumerate(comb_list):
        # if id >= 50:
        #     break
        print("Generating ", id+1, "/", len(comb_list))
        comb, prompt = generate_prompt(comb, relspat)

        # @backoff.on_exception(backoff.expo, openai.error.RateLimitError)
        # def completions_with_backoff(**kwargs):
        #     '''
        #     Helper function to deal with the "openai.error.RateLimitError". If not used, the script will simply
        #     stop once the limit is reached, not saving any of the data generated until then. This method will wait
        #     and then try again, hence preventing the error.
        #
        #     :param kwargs: List of arguments passed to the OpenAI API for completion.
        #     '''
        #     return openai.Completion.create(**kwargs)
        #
        # response = completions_with_backoff(
        #     # model="text-davinci-003",
        #     prompt=prompt,
        #     temperature=0.9,
        #     max_tokens=256,
        #     top_p=1.0,
        #     frequency_penalty=0.0,
        #     presence_penalty=0.0,
        #     # stop=["\n"]
        #     )
        # text = response['choices'][0]['text'].replace('\n', ' ').replace('\r', '').strip()

        @backoff.on_exception(backoff.expo, (openai.error.RateLimitError, openai.error.APIError, openai.error.APIConnectionError, openai.error.Timeout, openai.error.ServiceUnavailableError))
        def chatcompletions_with_backoff(**kwargs):
            '''
            Helper function to deal with the "openai.error.RateLimitError". If not used, the script will simply
            stop once the limit is reached, not saving any of the data generated until then. This method will wait
            and then try again, hence preventing the error.

            :param kwargs: List of arguments passed to the OpenAI API for completion.
            '''
            return openai.ChatCompletion.create(**kwargs)

        response = chatcompletions_with_backoff(
            model="gpt-3.5-turbo", #"gpt-4",
            temperature=0.9,
            max_tokens=1024,
            messages=[
                    {"role": "user", "content": prompt}
            ]
        )

        text = response['choices'][0]['message']['content'].replace('\r', '').strip()
        comb_list[id]["text"] = remove_surrounding_double_quotes(text)
        # text = ""

        print(prompt)
        # print(comb_list[id])
        print(text)
        print("*****")

        db_df.loc[id] = [comb, prompt, text]

        # break

        if save_files:
            db_df.to_csv(output_filename + "_" + version + "_ChatNL.csv")
            with open(output_filename + "_" + version + "_ChatNL.json", "w") as jsonfile:
                json.dump(comb_list, jsonfile, cls=NpEncoder)
            print("Saved files to output path!")

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
    # dump_keys()

    tag_list_path = "data/Tag_List_v9.csv"
    arbitrary_value_list_path = "data/Arbitrary_Value_List_v9.csv"
    query_file = "results/IMR_Dataset_v10"
    output_filename = "results/IMR_Dataset_v10"
    relative_spat = "data/relative_spatial_terms.csv"

    relspat_df = pd.read_csv(relative_spat, header=0, index_col=0, names=['Dist', 'Vals'])
    relspat = {}
    for index, row in relspat_df.iterrows():
        for rsd in row['Vals'].split(","):
            relspat[rsd.strip()] = row['Dist']

    for version in ["train", "dev", "test"]:
        json_filename = query_file + "_" + version + ".json"
        if os.path.isfile(json_filename):
            with open(json_filename, "r") as jsonfile:
                comb_list = json.load(jsonfile)
        else:
            comb_list = generate_query_combinations(tag_list_path, arbitrary_value_list_path, output_filename, True)

        query(comb_list, output_filename, relspat, version, True)

