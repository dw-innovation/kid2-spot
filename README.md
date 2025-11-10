<img width="1280" height="200" alt="Github-Banner_spot" src="https://github.com/user-attachments/assets/bec5a984-2f1f-44e7-b50d-cc6354d823cd" />

# 🌍 SPOT: Natural Language Interface to OpenStreetMap

**Describe a place, get a map of matching locations.**  
SPOT is an open-source tool that lets you search the world using natural language. It turns user queries into geospatial searches over OpenStreetMap (OSM) data — with no code or tagging knowledge required.

[➡️ Try the Demo](https://www.findthatspot.io/)  
[📽️ Watch the Demo Video](https://github.com/dw-innovation/kid2-spot/assets/23077479/110e3ef0-6fc6-4458-907a-0af5fa377370)

---

## ✨ Key Features

- 🔎 Natural Language → Structured Geospatial Search
- 🧠 Fine-tuned open LLMs (Mistral, LLaMA3) interpret scene descriptions
- 🏷️ Semantic bundling of OSM tags using Elasticsearch
- 🗺️ Interactive map frontend (Next.js + Leaflet)
- 💾 Dockerized architecture with multiple composable modules
- 📚 Open source & published under **AGPLv3**

---

## 📝 OSM Tag Bundles – Check & Suggest

SPOT relies on a curated list of OSM tag bundles to interpret user queries correctly. If you notice a query doesn’t work as expected, it might be due to a missing or mismatched tag.

[📄 View the tag bundles list (CSV)](./SPOT_OSM-tag-bundles.csv)  
[💬 Submit a suggestion or correction → Pinned Issue](https://github.com/dw-innovation/kid2-spot/issues/12#issue-3606799099)  

---

## 📦 Architecture Overview

SPOT is composed of several Dockerized modules, managed centrally:

```
kid2-spot/
├── frontend/                 # Map-based UI
├── apis/
│   ├── osmquery/             # Queries local OSM DB
│   ├── osmtagsearch/         # Maps phrases to OSM tag bundles
│   └── central-nlp-api/      # Orchestrates inference & pipeline
├── data_and_training/
│   ├── datageneration/       # Synthetic data generation
│   └── unsloth-training/     # LLM training pipeline
└── docker-compose.yml        # Orchestrates all services
```

---

## 🔧 Configuration

- See **[ENVIRONMENT.md](./ENVIRONMENT.md)** for service ports and module env variables.
- See **[SECURITY.md](../shared_docs/SECURITY.md)** for secrets handling best practices.

---

## 🚀 Quickstart

To clone and run the full project locally:

```bash
git clone --recurse-submodules https://github.com/dw-innovation/kid2-spot.git
cd kid2-spot
docker compose up --build
```

> Note: Make sure Docker has enough memory and disk space. You may need to configure `.env` files (see submodule READMEs).

---

## 📚 Publications

- **ACL 2025 Demo Paper**: [aclanthology.org/2025.acl-demo.8](https://aclanthology.org/2025.acl-demo.8)
- **OSM Science 2023**: [arXiv](https://arxiv.org/abs/2311.08093)

---

## 🧠 Submodules Overview

| Module                  | Description                                           |
|-------------------------|-------------------------------------------------------|
| `frontend`              | Map UI (Leaflet + Next.js)                            |
| `central-nlp-api`       | Converts NL → YAML → OSM tags                         |
| `osm-tag-search-api`    | Maps phrases to OSM tag bundles (Elasticsearch)       |
| `osm-query-api`         | Executes spatial query on local OSM DB                |
| `datageneration`        | Synthetic YAML/sentence generator for training        |
| `unsloth-training`      | Training script for open LLMs                         |

Each module has its own README file. You can also run each independently for debugging.

---

## 🎥 Demo Video

[![Watch the demo](https://img.youtube.com/vi/N-A/0.jpg)](https://github.com/dw-innovation/kid2-spot/assets/23077479/110e3ef0-6fc6-4458-907a-0af5fa377370)

---

## 🧑‍💻 Contributing

We welcome contributors from all backgrounds – developers, mappers, researchers, journalists!

Ways to help:
- Improve tag bundles or suggest new ones
- Help with frontend or UX design
- Add tests or documentation
- Improve prompts or training data

Please see [CONTRIBUTING.md](../shared_docs/CONTRIBUTING.md) to get started.

---

## 🛡 License

This project is licensed under the **GNU AGPLv3**.  
If you improve SPOT, please share your changes with the community.

© Deutsche Welle Research & Cooperation Projects · [AGPLv3 License](../LICENSE)