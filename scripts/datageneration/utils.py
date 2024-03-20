import json
import itertools
from typing import List

def write_output(generated_combs, output_file):
    with open(output_file, "w") as out_file:
        for generated_comb in generated_combs:
            json.dump(generated_comb, out_file)
            out_file.write('\n')


class CompoundTagAttributeProcessor:
    def expand_list(self, tag_compounds: str)-> List[str]:
        processed_tag_compounds = []
        tag_compounds = tag_compounds.split('|')
        for tag_compound in tag_compounds:
            tag_compound = tag_compound.replace('[', '').replace(']', '').replace('"', '')
            if len(tag_compound) != 0:
                processed_tag_compounds.append(tag_compound)
        return processed_tag_compounds

    def run(self, tag_compounds) -> List[str]:
        tag_compounds = tag_compounds.split('=')
        tag_compounds_keys = tag_compounds[0]
        tag_compounds_values = tag_compounds[1]

        if '[' in tag_compounds_keys:
            tag_compounds_keys = self.expand_list(tag_compounds_keys)

        if '[' in tag_compounds_values:
            tag_compounds_values = self.expand_list(tag_compounds_values)

        processed_tag_compounds = []
        for tag_key, tag_value in itertools.product(tag_compounds_keys, tag_compounds_values):
            processed_tag_compounds.append(f'{tag_key}={tag_value}')
        return processed_tag_compounds
