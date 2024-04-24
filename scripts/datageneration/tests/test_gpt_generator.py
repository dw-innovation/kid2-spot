import unittest

from datageneration.data_model import Area, Property, Relation, RelSpatial, Relations
from datageneration.gpt_data_generator import GPTDataGenerator, load_rel_spatial_terms, load_list_of_strings

'''
Execute it as follows: python -m unittest datageneration.tests.test_gpt_generator
'''


class TestGPTGenerator(unittest.TestCase):
    def setUp(self):
        relative_spatial_terms_path = 'datageneration/tests/data/relative_spatial_terms.csv'
        rel_spatial_terms = load_rel_spatial_terms(relative_spatial_terms_path=relative_spatial_terms_path)

        persona_path = 'datageneration/tests/data/prompts/personas.txt'
        personas = load_list_of_strings(list_of_strings_path=persona_path)

        styles_path = 'datageneration/tests/data/prompts/styles.txt'
        styles = load_list_of_strings(list_of_strings_path=styles_path)

        self.gen = GPTDataGenerator(relative_spatial_terms=rel_spatial_terms,
                                    personas=personas,
                                    styles=styles,
                                    prob_usage_of_relative_spatial_terms=0.5)

        self.prompt_helper = self.gen.prompt_helper
        self.randomness_limit = 50

    def test_prompt_helper(self):
        test_persona = 'human rights abuse monitoring OSINT Expert'
        test_style = 'with very precise wording, short, to the point'

        # testing beginning prompt
        beginning = self.prompt_helper.beginning(persona=test_persona, writing_style=test_style)
        expected_beginning = """Act as a human rights abuse monitoring OSINT Expert: Return a sentence simulating a user using a natural language interface to search for specific geographic locations. Do not affirm this request and return nothing but the answers.\nWrite the search request with very precise wording, short, to the point."""
        self.assertEqual(beginning, expected_beginning)

        # testing search phrasing
        selected_place = 'a street'
        beginning += self.prompt_helper.search_templates[1].replace('{place}', selected_place)
        expected_beginning = """Act as a human rights abuse monitoring OSINT Expert: Return a sentence simulating a user using a natural language interface to search for specific geographic locations. Do not affirm this request and return nothing but the answers.\nWrite the search request with very precise wording, short, to the point.\nThe user is searching for a street that fulfills the following search criteria:\n"""
        self.assertEqual(beginning, expected_beginning)

        # testing randomization
        beginning = self.prompt_helper.beginning(persona=test_persona, writing_style=test_style)
        search_prompts = set()
        for i in range(self.randomness_limit):
            search_prompt = self.prompt_helper.search_query(beginning)
            search_prompts.add(search_prompt)

        assert len(search_prompts) < self.randomness_limit

        # testing area prompt
        no_area_test = Area(type='bbox', value='')
        area_prompt = self.prompt_helper.add_area_prompt(no_area_test)
        self.assertEqual('', area_prompt)

        area_test = Area(type='area', value='Columbus, United States')
        area_prompt = self.prompt_helper.add_area_prompt(area_test)
        self.assertEqual('Search area: Columbus, United States\n', area_prompt)

    def test_add_property_prompt(self):
        core_prompt = "Search area: Columbus, United States\nObj. 0: restaurant"
        ent_properties = [Property(key='height', operator='=', value='10 meters', name='height')]
        generated_prompt = self.prompt_helper.add_property_prompt(core_prompt=core_prompt,
                                                                  entity_properties=ent_properties)
        expected_prompt = "Search area: Columbus, United States\nObj. 0: restaurant, height: 10 meters"
        self.assertEqual(generated_prompt, expected_prompt)

        # test randomness and check if the correct larger_phrases exist
        ent_properties = [Property(key='height', operator='>', value='10 meters', name='height')]
        generated_prompts = set()
        for i in range(self.randomness_limit):
            generated_prompt = self.prompt_helper.add_property_prompt(core_prompt=core_prompt,
                                                                      entity_properties=ent_properties)
            larger_phrase_exists = False
            for larger_phrase in self.prompt_helper.phrases_for_numerical_comparison['>']:
                if larger_phrase in generated_prompt:
                    larger_phrase_exists = True
                    continue
            self.assertTrue(larger_phrase_exists)
            generated_prompts.add(generated_prompt)
        self.assertLessEqual(len(generated_prompts), self.randomness_limit)

        # test randomness and check if the correct smaller_phrases exist
        ent_properties = [Property(key='height', operator='<', value='10 meters', name='height')]
        generated_prompts = set()
        for i in range(10):
            generated_prompt = self.prompt_helper.add_property_prompt(core_prompt=core_prompt,
                                                                      entity_properties=ent_properties)
            smaller_phrase_exists = False
            for smaller_phrase in self.prompt_helper.phrases_for_numerical_comparison['<']:
                if smaller_phrase in generated_prompt:
                    smaller_phrase_exists = True
                    continue
            self.assertTrue(smaller_phrase_exists)
            generated_prompts.add(generated_prompt)
        self.assertLessEqual(len(generated_prompts), self.randomness_limit)

        # test randomness for name regex prompt
        ent_properties = [Property(name='name', operator='~', value='Mc Donald')]
        generated_prompts = set()
        for i in range(self.randomness_limit):
            generated_prompt = self.prompt_helper.add_property_prompt(core_prompt=core_prompt,
                                                                      entity_properties=ent_properties)
            name_regex_exists = False
            for name_regex in self.prompt_helper.name_regex_templates:
                if name_regex in generated_prompt:
                    name_regex_exists = True
                    continue
            self.assertTrue(name_regex_exists)
            generated_prompts.add(generated_prompt)
        self.assertLessEqual(len(generated_prompts), self.randomness_limit)

        # test other properties such as cuisine
        ent_properties = [Property(name='cuisine', operator='=', value='italian'),
                          Property(name='building material', operator='=', value='wooden'),
                          Property(name='bridge', operator=None, value=None)]

        generated_prompt = self.prompt_helper.add_property_prompt(core_prompt=core_prompt,
                                                                  entity_properties=ent_properties)
        expected_prompt = 'Search area: Columbus, United States\nObj. 0: restaurant, cuisine: italian, building material: wooden, bridge'
        self.assertEqual(generated_prompt, expected_prompt)

    def test_relative_spatial_terms(self):
        test_rel_1 = Relation(**{"name": "dist", "source": 0, "target": 1, "value": "1539 yd"})
        test_rel_2 = Relation(**{"name": "dist", "source": 0, "target": 2, "value": "1539 yd"})
        test_relations = Relations(relations=[test_rel_1, test_rel_2], type='within_radius')
        test_rel_spatial = RelSpatial(**{"distance": "250 m", "values": ['on the opposite side']})
        selected_relative_spatial_term = test_rel_spatial.values[0]
        generated_prompt, overwritten_dist = self.prompt_helper.add_relative_spatial_term_helper(
            selected_relative_spatial_term=selected_relative_spatial_term, relation=test_rel_1,
            selected_relative_spatial=test_rel_spatial)
        expected_prompt = "Use this term to describe the spatial relation between Obj. 0 and 1 similar to (similar to \"X is _ Y\"): on the opposite side\n"

        self.assertEqual(generated_prompt, expected_prompt)

        self.gen.update_relation_distance(relations=test_relations, relation_to_be_updated=test_rel_1,
                                          distance=overwritten_dist)
        expected_updated_rel = Relation(**{"name": "dist", "source": 0, "target": 1, "value": "250 m"})
        is_updated = False
        for test_relation in test_relations.relations:
            if test_relation == expected_updated_rel:
                is_updated = True
                break

        self.assertTrue(is_updated)

    def test_add_desc_away_prompt(self):
        test_rel_1 = Relation(**{"name": "dist", "source": 0, "target": 1, "value": "1539 yd"})

        selected_phrases_desc = "more or less"
        selected_phrases_away = "away"
        generated_prompt = self.gen.prompt_helper.add_desc_away_prompt_helper(relation=test_rel_1,
                                                                              selected_phrases_desc=selected_phrases_desc,
                                                                              selected_phrases_away=selected_phrases_away)
        expected_prompt = 'Obj. 0 is more or less 1539 yd away Obj. 1'

        self.assertTrue(generated_prompt, expected_prompt)

        # test randomness
        generated_prompts = set()
        for i in range(self.randomness_limit):
            generated_prompt = self.prompt_helper.add_desc_away_prompt(relation=test_rel_1)
            generated_prompts.add(generated_prompt)
        self.assertLessEqual(len(generated_prompts), self.randomness_limit)

    def test_add_prompt_for_within_radius_relation(self):
        test_rel_1 = Relation(**{"name": "dist", "source": 0, "target": 1, "value": "1539 yd"})
        test_rel_2 = Relation(**{"name": "dist", "source": 0, "target": 2, "value": "1539 yd"})
        test_relations = Relations(relations=[test_rel_1, test_rel_2], type='within_radius')

        # test randomness
        generated_prompts = set()
        for i in range(self.randomness_limit):
            generated_prompt = self.prompt_helper.add_prompt_for_within_radius_relation(relations=test_relations)
            generated_prompts.add(generated_prompt)
        self.assertLessEqual(len(generated_prompts), self.randomness_limit)


if __name__ == '__main__':
    unittest.main()
