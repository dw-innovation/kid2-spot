import json
import os


def write_output(generated_combs, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w") as out_file:
        for generated_comb in generated_combs:
            json.dump(generated_comb.model_dump(mode="json"), out_file)
            out_file.write('\n')
