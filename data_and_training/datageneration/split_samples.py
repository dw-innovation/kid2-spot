import os
import jsonlines


def split_jsonl(input_file, output_dir, max_samples_per_chunk=500):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with jsonlines.open(input_file, 'r') as reader:
        chunk_count = 1
        samples = []
        for sample in reader:
            samples.append(sample)
            if len(samples) == max_samples_per_chunk:
                output_file = os.path.join(output_dir, f'samples_chunk_{chunk_count}.jsonl')
                with jsonlines.open(output_file, 'w') as writer:
                    writer.write_all(samples)
                samples = []
                chunk_count += 1

        # Write remaining samples to the last chunk file
        if samples:
            output_file = os.path.join(output_dir, f'samples_chunk_{chunk_count}.jsonl')
            with jsonlines.open(output_file, 'w') as writer:
                writer.write_all(samples)


# Usage
split_jsonl('datageneration/results/v12/samples.jsonl', 'datageneration/results/v12')
