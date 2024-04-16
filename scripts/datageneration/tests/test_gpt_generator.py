import unittest

from datageneration.data_model import Area, Property
from datageneration.gpt_data_generator import GPTDataGenerator, load_rel_spatial_terms, load_list_of_strings, \
    PromptHelper

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
                                    styles=styles)

        self.prompt_helper = PromptHelper()

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
        for i in range(10):
            search_prompt = self.prompt_helper.search_query(beginning)
            search_prompts.add(search_prompt)

        assert len(search_prompts) < 10

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
        for i in range(10):
            generated_prompt = self.prompt_helper.add_property_prompt(core_prompt=core_prompt,
                                                                      entity_properties=ent_properties)
            larger_phrase_exists = False
            for larger_phrase in self.prompt_helper.phrases_for_numerical_comparison['>']:
                if larger_phrase in generated_prompt:
                    larger_phrase_exists = True
                    continue
            self.assertTrue(larger_phrase_exists)
            generated_prompts.add(generated_prompt)
        self.assertLessEqual(len(generated_prompts), 10)

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
        self.assertLessEqual(len(generated_prompts), 10)

        # test randomness for name regex prompt
        ent_properties = [Property(key='name', operator='~', value='10 meters', name='name')]
        generated_prompts = set()
        for i in range(10):
            generated_prompt = self.prompt_helper.add_property_prompt(core_prompt=core_prompt,
                                                                        entity_properties=ent_properties)
            name_regex_exists = False
            for name_regex in self.prompt_helper.name_regex_templates:
                if name_regex in generated_prompt:
                    name_regex_exists = True
                    continue
            self.assertTrue(name_regex_exists)
            generated_prompts.add(generated_prompt)
        self.assertLessEqual(len(generated_prompts), 10)

        # test other properties such as cuisine
        ent_properties = [Property(key='cuisine', operator='=', value='italian', name='cuisine'), Property(key='building:material', operator='=', value='wooden', name='material')]
        generated_prompt = self.prompt_helper.add_property_prompt(core_prompt=core_prompt, entity_properties=ent_properties)
        expected_prompt = 'Search area: Columbus, United States\nObj. 0: restaurant, cuisine: italian, material: wooden'
        self.assertEqual(generated_prompt, expected_prompt)


    # def test_generate_prompt(self):
    #     test_comb = {"area": {"type": "area", "value": "Columbus, United States"}, "entities": [
    #         {"id": 0, "name": "downhill ski run", "type": "nwr",
    #          "properties": [{"key": "highway", "operator": "=", "value": "bus_guideway"}]},
    #         {"id": 1, "name": "post relay box", "type": "nwr",
    #          "properties": [{"key": "height", "operator": "=", "value": "1406 yd"},
    #                         {"key": "building", "operator": "=", "value": "bridge"},
    #                         {"key": "name", "operator": "=", "value": "KFC"}]},
    #         {"id": 2, "name": "office of a telecommunication company", "type": "nwr",
    #          "properties": [{"key": "internet_access", "operator": "=", "value": "wlan"},
    #                         {"key": "building", "operator": "=", "value": "bridge"},
    #                         {"key": "name", "operator": "=", "value": "Main Street"}]},
    #         {"id": 3, "name": "aerorotor", "type": "nwr",
    #          "properties": [{"key": "man_made", "operator": "=", "value": "tunnel"},
    #                         {"key": "roof:material", "operator": "=", "value": "cadjan_palmyrah_straw"}]}],
    #                  "relations": [{"name": "dist", "source": 0, "target": 1, "value": "1539 yd"},
    #                                {"name": "dist", "source": 0, "target": 2, "value": "24.2 mi"},
    #                                {"name": "dist", "source": 0, "target": 3, "value": "1195 mi"}]}
    #
    #     test_persona = 'human rights abuse monitoring OSINT Expert'
    #     test_style = 'with very precise wording, short, to the point'
    #
    #     # todo: recheck why do we need to change comb, hence return modified comb and a generated prompt
    #     _, generated_prompt = self.gen.generate_prompt(LocPoint(**test_comb), persona=test_persona,
    #                                                    style=test_style)
    #
    #     assert test_persona in generated_prompt
    #     assert test_style in generated_prompt

    # def test_generate_prompt(self):
    #     pairs = self.gen.assign_persona_styles_to_queries(3, 10)
    #
    #     assert len(pairs) == 10
    #
    #     predicted_persona_style_ids = set()
    #     for tag_id, persona_and_style_id in pairs:
    #         predicted_persona_style_ids.add(persona_and_style_id)
    #
    #     assert len(predicted_persona_style_ids) == 3

    # def test_generate_prompt_cuisine(self):
    #     # todo write a command for checking overwritten
    #     pass


if __name__ == '__main__':
    unittest.main()
