# KID2 Spot Application

## Introduction

We present Spot, a natural language interface for querying 
OpenStreetMap data and finding Spots, combinations of objects in the 
public space. Users' natural sentences are translated by a 
transformer model and turned into OSM database searches. The primary 
use case is geo-location verification, other use cases however are
also possible.

This is a provisionary readme, more information will follow.

## How it works

This is a collection of repositories and scripts linked to the KID2 
Spot application. Please visit the readme pages of all subtasks and 
repositories for more details.

![Spot Pipeline](https://github.com/dw-innovation/kid2-spot/blob/main/media/Spot-Pipeline.png?raw=true)

The general idea of the pipeline is the following:
1) Bundle similar tags, assign natural language descriptors to create better semantic connections between language and the OSM tagging system
2) Generate random artificial queries including area definition, objects (incl. tags and descriptors) and relations/distances
3) Call GPT API to generate artificial natural sentences from the draft
4) Use T5 transformer to extract relevant information from sentences
5) Perform Postgres query in OSM database & display in geographic map

The GUI allows for the display of results of the natural language query
in a geographic map, the editing of individual parts of the query, the integration
of OpenStreetMap to inspect the candidate locations and more.

Watch the demo video below to get an impression of some of Spot's functionality.

https://github.com/dw-innovation/kid2-spot/assets/23077479/110e3ef0-6fc6-4458-907a-0af5fa377370

## Call for participation

This is an open-source project whose code and eventually also website will always be publicly available.
If you would like to contribute to the project, please reach out to us via this repository. We would especially
appreciate the help of people with in-depth knowledge of the OSM tagging system.

Potential tasks:
- Improvement of bundle database, to increase the number of tags included and the quality of the bundles
- GPT prompt engineering to write better prompts that result in a more diverse language output
- Coding & model development
- User testing
- etc.

Feel free to reach out if you want to participate in this project.

## Contact

If you would like to participate or get in touch with us, please do so by
opening an issue, or via email: lynn.khellaf@dw.com

## Publications

Proceedings of OSM Science 2023: https://zenodo.org/records/10443346 (also available at https://arxiv.org/abs/2311.08093)
Other publication will be shared once available.

## Acknowledgments

This project is led by the Deutsche Welle Research and Cooperation Projects 
teams and was co-funded by BKM (”Beauftragte der Bundesregierung für Kultur 
und Medien,” the German Government’s Commissioner for Culture and Media).

Map data copyrighted OpenStreetMap contributors and available 
from https://www.openstreetmap.org.

