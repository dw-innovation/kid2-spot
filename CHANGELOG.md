# Changelog

---

## Version 1.1.0 (first public beta release) 

### **New Features**
- **Properties**: Added colors (including roof and building colors), e.g., a “brown (bench).”
- **OSM Expert Editor**: Added the first version of an editor to update and add tags to existing OSM queries in the GUI.
- **Feedback Form**: Added a form to the GUI to allow users to give direct feedback.

### **Fixes**
- **New Training Data**: Better entity and property detection via improvements to data generation and training parameters.

### **Work in Progress**
- **Laid Groundwork for Entity Clusters**: Ability to generate training data for clusters of entities, allowing queries like "3 Italian restaurants next to each other" or "at least 5 wind generators in a radius of 200 m." This feature will be available in a future model iteration.

---

## Version 1.0.0

### **New Features**

1. **Entity Detection**
   - **Semantic Entities**: Spot identifies general categories like "restaurant" and "train station," allowing recognition of places based on type.
   - **Named Entities (Brands)**: Detection of specific brand names, including "McDonald's," "KFC," "Tchibo," and compound names like "Thalia bookstore."

2. **Entity Properties**
   - **Named Properties**: Spot identifies properties like "vegan (food shop)" or "Italian (restaurant)" for refined queries.
   - **Numerical Properties**: Ability to interpret quantitative descriptors, including height, levels, and house numbers.

3. **Area Recognition**
   - **Named and Administrative Areas**: Support for cities, districts, and regions, including multi-word areas (e.g., "New York") and regions like "Nordrhein-Westfalen."
   - **Bounding Box for Undefined Area Queries**: Introduces bounding box support for identifying entities within a broader, undefined area.

4. **Distance Relations**
   - **Numerical and Written Distances**: Spot interprets both numeric distances (e.g., "100 meters") and written forms (e.g., "one hundred meters").
   - **Relative Distance Terms**: Supports terms like "next to," "opposite from," and "beside" to improve natural understanding of spatial relationships.
   - **Distance Chain and Radius Support**: Multiple distance-based relations are supported, including radius constraints (e.g., "A to B and C") and entity chains (e.g., "A to B and B to C").

5. **Contains Relations**
   - **Basic Containment**: Recognizes relationships such as "a fountain within a park" and "a shop inside a mall."
   - **With Relations**: Expanded containment to support "with" relationships, such as "a park with a fountain" or "hotel with a parking lot."

6. **Spatial Terms and Descriptors**
   - **Relative Distance Phrasing**: Enhanced natural language understanding for relative spatial terms like "close to," "next to," and "behind."
   - **Descriptor Matching**: Improved matching of descriptors with slight variations, such as plurals ("bookshops" vs. "bookshop") and minor differences ("bookstore" vs. "book shop").

7. **Prompt and Linguistic Features**
   - **Typo Handling**: Improved error tolerance to manage typos in names and common words (e.g., "MacDonalds" for "McDonald's").
   - **Language Style Variability**: Added support for both formal and casual query styles.
   - **Multilingual Area and Brand Names**: Recognizes area names and locations in multiple languages and alphabets, including non-Roman alphabets like Cyrillic and Greek.
   - **Multiple Sentence Queries**: Supports both single and multi-sentence structures in user queries.

8. **User Interface and Frontend Enhancements**
   - **Rendering Results on the Map**: Results from natural language queries are displayed directly on the map, allowing users to visually locate and assess relevant areas.
   - **Exploring Candidate Locations**: The interface integrates third-party map services, enabling users to investigate specific coordinates using tools like Google Maps or Google Street View.
   - **Refining Search Queries Visually**: Users can modify search parameters interactively, including adjusting distance relations between objects for more targeted results.
   - **Session Management**: Added functionality for saving search sessions for future use, as well as loading previously saved sessions to continue work seamlessly.
   - **Exporting Map Data**: The system supports exporting map data in multiple formats, allowing users to use the data in external applications or workflows.

### **Fixes**
- No fixes in version 1.0.0.