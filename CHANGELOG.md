# 📓 Changelog

All notable changes to this project will be documented in this file.  
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),  
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- Unified README format for all SPOT modules
- CONTRIBUTING.md and CODE_OF_CONDUCT.md
- Initial AGPLv3 license
- Centralized Docker Compose file for all modules

---

## [v1.2.0] 

### Added
- **Cluster Entities**: Support for queries involving multiple entities of the same type (e.g. "three benches", "two windmills next to each other").
- **Ambiguous Term Definitions**: Added relative spatial terms (e.g. "next to", "nearby") with associated distance values; included new "contains" indicators (e.g. "in", "with").

### Changed
- **Updated Tag Bundle List**: Refreshed tags and descriptors used in semantic search.
- **Improved Prompting**: Enhanced prompts for training and inference to improve model accuracy.

### Fixed
- **Benchmark Improvements**: Addressed several recurring error patterns by refining data generation and prompt parameters.

---

## [v1.1.1] 

### Changed
- **Taglist Reset**: Search engine updated with a revised tag list based on recent user feedback.

---

## [v1.1.0] 

### Added
- **Property Support**: Recognizes color properties, including building and roof colors (e.g. “brown bench”).
- **OSM Expert Editor**: GUI feature for editing and updating tags in OSM queries.
- **User Feedback Form**: Direct input form for users embedded in the GUI.

### Fixed
- **Improved Training Data**: Enhanced entity and property detection via optimized data generation.

### Work in Progress
- **Entity Cluster Foundation**: Added groundwork for entity cluster queries like "3 Italian restaurants next to each other". Full support coming in future updates.

---

## [v1.0.0]

### Added

#### Entity Detection
- **Semantic Entities**: Detects general types (e.g. "restaurant", "train station").
- **Named Entities (Brands)**: Recognizes brand names (e.g. "McDonald's", "KFC", "Thalia bookstore").

#### Entity Properties
- **Named Properties**: Includes descriptive tags like "vegan (food shop)" or "Italian (restaurant)".
- **Numerical Properties**: Supports height, level, and house number detection.

#### Area Recognition
- **Named/Administrative Areas**: Supports multi-word areas and regions (e.g. "New York", "Nordrhein-Westfalen").
- **Bounding Box**: Fallback for area queries without specific names.

#### Distance Relations
- **Written & Numeric Distances**: Understands "100 meters" and "one hundred meters".
- **Relative Distance Terms**: Includes "next to", "opposite from", "beside".
- **Distance Chains & Radius**: Complex relations like "A to B and B to C".

#### Contains Relations
- **Basic Containment**: Handles relations like "a fountain within a park".
- **With Relations**: Adds support for "a hotel with a parking lot", etc.

#### Spatial Terms & Descriptors
- **Relative Phrasing**: Better interpretation of "close to", "behind", etc.
- **Descriptor Matching**: Handles minor differences and plurals (e.g. "bookshops" vs "bookshop").

#### Prompt & Linguistic Features
- **Typo Handling**: Tolerates common spelling mistakes.
- **Style Flexibility**: Supports both formal and casual query styles.
- **Multilingual Support**: Recognizes names in various alphabets (e.g. Cyrillic, Greek).
- **Multi-Sentence Queries**: Processes both single and compound sentence inputs.

#### User Interface
- **Map Result Rendering**: Displays results directly on the map.
- **Candidate Exploration**: Links to Google Maps and Street View.
- **Visual Query Editing**: GUI for refining spatial relations interactively.
- **Session Management**: Save/load search sessions.
- **Data Export**: Export results in various formats for external use.
