# 🌍 KID2 Spot Application

## Introduction

Spot is a natural language interface designed to query OpenStreetMap (OSM) data and identify "Spots" – combinations of objects in public space. By leveraging a transformer model, users' natural language inputs are translated into OSM database queries. While its primary use case is geo-location verification, the application can be adapted to other scenarios as well.

---

## How it Works

Spot allows users to find locations that meet specific requirements by prompting the system with a natural language sentence. The process works as follows:

1. **💬 Natural Language Input**: Users describe their requirements using a natural language sentence. 
2. **🔗 Graph Representation**: Spot transforms the sentence into a graph representation that captures all objects, their properties, and the relationships between them.
3. **🗺️ OSM Tag Retrieval**: The system maps the identified properties to OpenStreetMap (OSM) tags. This is achieved using Elasticsearch, which leverages predefined tag bundles along with openly available OSM tag data.
4. **⚙️ Database Query Construction and Execution**: Using the Spot query, a database query is constructed and executed against a local replica of OSM data.
5. **📍 Results Rendering**: The query results are rendered in the frontend, allowing users to visually explore the locations that satisfy their requirements.

The KID2 Spot application comprises a collection of repositories and scripts. For detailed information, please refer to the README pages of the respective subtasks and repositories.

![Spot Pipeline](https://github.com/dw-innovation/kid2-spot/blob/main/media/Spot-Pipeline.png?raw=true)

---

### Pipeline Overview

The pipeline operates as follows:

1. **🧩 Bundle Tags and Assign Descriptors**  
   Similar tags are grouped, and natural language descriptors are assigned to establish better semantic connections with the OSM tagging system.

2. **🔀 Generate Artificial Queries**  
   Random artificial queries are created, including area definitions, objects (with tags and descriptors), and relationships/distances.

3. **📝 Generate Natural Sentences**  
   The GPT API generates artificial natural language sentences from the the artifical queries. The generated sentences are used for fine-tuning Llama3.

4. **🔍 Extract Relevant Information**  
   Llama3 parse the generated sentences for constructing queries and the queries are enriched with their OSM tag which are fetched from a vector-based search engine.

5. **🗄️ Perform Database Query**  
   PostgreSQL queries the OSM database to fetch relevant data, which is then displayed on a geographic map.

---

### 🖥️ Front-end Functionality

The graphical user interface in Spot is a dynamic and versatile Next.js Leaflet map application. It offers multiple map layers, including satellite imagery, OpenStreetMap (OSM), and vector tiles, providing users with a flexible and interactive mapping experience. The key features include:

- **📌 Rendering Results on the Map**: The results of natural language queries are visualized directly on the map, allowing users to identify relevant locations.
- **🔗 Exploring Candidate Locations**: Users can investigate specific locations by integrating third-party map services, such as opening Google Maps or Google Street View at specified coordinates.
- **🛠️ Refining Search Queries Visually**: The interface enables users to adjust search parameters visually, such as modifying distance relations between objects.
- **💾 Session Management**: Users can save their current search sessions for future use or load previously saved sessions.
- **📤 Exporting Map Data**: The system supports exporting map data in various formats, enabling users to work with the data in external tools or applications.

---

### 🎥 Demo

Watch the demo video to see Spot in action:

https://github.com/dw-innovation/kid2-spot/assets/23077479/110e3ef0-6fc6-4458-907a-0af5fa377370

---

## 🚀 Call for Participation

Spot is an open-source project, with its code and (eventually) its website freely available to the public. We invite contributors, particularly those with expertise in the OSM tagging system, to collaborate with us.

### Potential Contribution Areas

- Enhancing the tag bundle database to expand its scope and improve quality.
- Developing better GPT prompts for more diverse natural language outputs.
- Coding and advancing model development.
- Conducting user testing.
- And more!

If you're interested in contributing, please get in touch via this repository.

---

## 📬 Contact

To participate or contact us, please open an issue or email us at:  
[lynn.khellaf@dw.com](mailto:lynn.khellaf@dw.com)

---

## 📚 Publications

- **Proceedings of OSM Science 2023**:  
  [Zenodo Link](https://zenodo.org/records/10443346) | [arXiv Link](https://arxiv.org/abs/2311.08093)  
  Additional publications will be shared as they become available.

---

## 🙏 Acknowledgments

This project is led by [Deutsche Welle's Research and Cooperation Projects](https://innovation.dw.com) team and co-funded by BKM ("Beauftragte der Bundesregierung für Kultur und Medien", the German Government’s Commissioner for Culture and Media).

Map data © OpenStreetMap contributors, available at [OpenStreetMap](https://www.openstreetmap.org).
