import json
def write_output(generated_combs, output_file):
    with open(output_file, "w") as out_file:
        for generated_comb in generated_combs:
            json.dump(generated_comb, out_file)
            out_file.write('\n')