import unittest

from datageneration.data_model import Area, Property, Relation, RelSpatial, Relations, LocPoint, Entity
from datageneration.gpt_data_generator_v2 import GPTDataGenerator, load_rel_spatial_terms, load_list_of_strings, \
    PromptHelper

'''
Execute it as follows: python -m unittest datageneration.tests.test_gpt_generator_v2
'''


class TestGPTGenerator(unittest.TestCase):
    def setUp(self):
        relative_spatial_terms_path = 'datageneration/tests/data/relative_spatial_terms.csv'
        rel_spatial_terms = load_rel_spatial_terms(relative_spatial_terms_path=relative_spatial_terms_path)

        persona_path = 'datageneration/tests/data/prompts/personas.txt'
        personas = load_list_of_strings(list_of_strings_path=persona_path)

        styles_path = 'datageneration/tests/data/prompts/styles.txt'
        styles = load_list_of_strings(list_of_strings_path=styles_path)

        system_prompt = 'datageneration/tests/data/prompts/styles.txt'
        system_prompt = load_list_of_strings(list_of_strings_path=system_prompt)

        self.gen = GPTDataGenerator(
            system_prompt=system_prompt,
            relative_spatial_terms=rel_spatial_terms,
            personas=personas,
            styles=styles,
            prob_usage_of_relative_spatial_terms=0.5)

        self.prompt_helper = self.gen.prompt_helper
        self.randomness_limit = 50

    def test_generate_prompt(self):
        loc_point = LocPoint(area=Area(type="area", value="Dublin City, Leinster, Ireland"),
                             entities=[
                                 Entity(id=0, is_area=True, name="building under construction", type="nwr",
                                        properties=[
                                            Property(name="retail district"),
                                            Property(name="building levels", operator='>', value='75'),
                                        ]),
                                 Entity(id=1, is_area=False, name="attorney general, district attorney", type="nwr",
                                        properties=[Property(name="retail district"),
                                                    Property(name="floors", operator='<', value='72')
                                                    ]),
                                 Entity(id=2, is_area=False, name="outdoor stor", type="nwr", properties=[]),
                                 Entity(id=3, is_area=False, name="food counter", type="nwr", properties=[])
                             ],
                             relations=Relations(relations=[
                                 Relation(type="contains", source=0, target=1),
                                 Relation(type="contains", source=0, target=2),
                                 Relation(type="contains", source=3, target=1, value="185 mi"),
                             ], type="contains_relation")
                             )
        style = "like someone in a hurry"
        persona = "political journalist"

        loc_area, generated_prompt = self.gen.generate_prompt(loc_point=loc_point, persona=persona, style=style)

        expected_prompt = """===Input===
```yaml
area:
  type: area
  value: Dublin City, Leinster, Ireland
entities:
- id: 0
  is_area: true
  name: building under construction
  properties:
  - name: retail district
  - name: building levels
    operator: '>'
    value: '75'
  type: nwr
- id: 1
  is_area: false
  name: attorney general, district attorney
  properties:
  - name: retail district
  - name: floors
    operator: <
    value: '72'
  type: nwr
- id: 2
  is_area: false
  name: outdoor stor
  type: nwr
- id: 3
  is_area: false
  name: food counter
  type: nwr
relations:
- source: 0
  target: 1
  type: contains
- source: 0
  target: 2
  type: contains
- source: 3
  target: 1
  type: contains
  value: 185 mi
```

===Persona===
political journalist

===Style===
like someone in a hurry

===Sentence===
"""

        self.assertEqual(generated_prompt, expected_prompt)

        loc_point = LocPoint(area=Area(type="area", value="Dublin City, Leinster, Ireland"),
                             entities=[
                                 Entity(id=0, is_area=True, name="building under construction", type="nwr",
                                        properties=[
                                            Property(name="retail district"),
                                            Property(name="building levels", operator='>', value='75'),
                                        ]),
                                 Entity(id=1, is_area=False, name="attorney general, district attorney", type="nwr",
                                        properties=[Property(name="retail district"),
                                                    Property(name="floors", operator='<', value='72')
                                                    ]),
                                 Entity(id=2, is_area=False, name="outdoor stor", type="nwr", properties=[]),
                             ],
                             relations=Relations(relations=[
                                 Relation(type="contains", source=0, target=1),
                                 Relation(type="dist", source=2, target=1, value="1000 m"),
                             ], type="individual_distances_with_contains")
                             )
        style = "like someone in a hurry"
        persona = "political journalist"

        loc_area, generated_prompt = self.gen.generate_prompt(loc_point=loc_point, persona=persona, style=style)


        # todo make logical tests
        print(generated_prompt)


if __name__ == '__main__':
    unittest.main()
