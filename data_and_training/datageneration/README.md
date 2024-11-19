# Database generation scripts
## Introduction

All scripts and files required to generate training data. The folders 
data & results contain samples of the original input, as well as the 
output of each script.

The original files needed to begin the process are the base bundle 
database *Primary_Keys_filtered5.csv*, an additional file 
containing definitions of relative spatial terms 
*relative_spatial_terms.csv* and a list of all countries, states and 
cities in the world *countries+states+cities.json*.

### Setup

Install the required libraries:
```shell
pip install -r requirements.txt
```

Here is a short description of all scripts in order of execution.

### 1) retrieve_combinations.py

After editing the database file, run this script to fill in the column 
"combinations". These are used to limit the random draft to tag 
combinations found in the OSM taginfo database.

```shell
bash scripts/retrieve_combinations.sh
```

### 2) generate_combination_table.py

Execute a random draft of artificial queries. Each query contains the
following information:
- Area definition
- List of objects with natural language descriptors and one or more assigned tags
- Relations / distances between the tags

```shell
bash scripts/generate_combinations.sh
```

### 3) gpt_data_generator.py

Use the random draft info to build a GPT prompt and generate an artificial
natural sentence simulating a user.

To execute this script, an additional file *.env* is required. It must contain the following contents:
```yaml
TEMPERATURE=0.9
MODEL=gpt-3.5-turbo
MAX_TOKENS=1024
OPENAI_API_KEY={OPENAI_API_KEY}
OPENAI_ORG=={OPENAI_ORG}
```

```shell
bash scripts/generate_samples_with_gpt.sh
```

### 4) tags_to_imr.py

A file that takes the bundle list and transforms it to a version in 
which all tag bundles are represented in the graph database format 
the model translates natural sentences into. This is required to be able
to assign the correct tags to the descriptors detected by the model.

```shell
bash scripts/convert_tags_to_imr.sh
```
