import unittest

from datageneration.gpt_data_generator import GPTDataGenerator

'''
Execute it as follows: python -m datageneration.tests.test_gpt_generator
'''


class TestGPTGenerator(unittest.TestCase):

    def setUp(self):
        tag_list_path = 'datageneration/tests/data/Tag_List_v10.csv'
        arbitrary_value_list_path = 'datageneration/tests/data/Arbitrary_Value_List_v10.csv'
        relative_spatial_terms_path = 'datageneration/tests/data/relative_spatial_terms.csv'
        persona_path = 'datageneration/tests/data/prompts/personas.txt'
        styles_path = 'datageneration/tests/data/prompts/styles.txt'

        self.gen = GPTDataGenerator(tag_list_path, arbitrary_value_list_path, relative_spatial_terms_path, persona_path,
                                    styles_path)

    def test_generate_prompt(self):
        test_comb = {"a": {"t": "area", "v": "San Lorenzo"}, "ns": [{"id": 0, "flts": [
            {"k": "power", "v": "cable", "op": "=", "n": "power cable"},
            {"k": "name", "v": "Willow Creek", "op": "=", "n": "name"}], "t": "nwr"}, {"id": 1, "flts": [
            {"k": "building", "v": "beach_hut", "op": "=", "n": "beach hut"}], "t": "nwr"}, {"id": 2, "flts": [
            {"k": "shop", "v": "florist", "op": "=", "n": "florist"},
            {"k": "addr:street", "v": "thg", "op": "~", "n": "street name"},
            {"k": "building", "v": "retail", "op": "=", "n": "shopping area"},
            {"k": "building:levels", "v": "22", "op": "=", "n": "floors"}], "t": "nwr"}, {"id": 3, "flts": [
            {"k": "man_made", "v": "pipeline", "op": "=", "n": "pipeline"},
            {"k": "name", "v": "Bloomen", "op": "~", "n": "name"},
            {"k": "water", "v": "river", "op": "=", "n": "stream"}], "t": "nwr"}],
                     "es": [{"src": 0, "tgt": 1, "t": "dist", "dist": "161 m"},
                            {"src": 1, "tgt": 2, "t": "dist", "dist": "496 mm"},
                            {"src": 2, "tgt": 3, "t": "dist", "dist": "16.06 km"}]}
        test_persona = 'human rights abuse monitoring OSINT Expert'
        test_style = 'with very precise wording, short, to the point'

        # todo: recheck why do we need to change comb, hence return modified comb and a generated prompt
        _, generated_prompt = self.gen.generate_prompt(test_comb, persona=test_persona,
                                                       style=test_style)

        assert test_persona in generated_prompt
        assert test_style in generated_prompt

    def test_generate_prompt(self):
        pairs = self.gen.assign_persona_styles_to_queries(3, 10)

        assert len(pairs) == 10

        predicted_persona_style_ids = set()
        for tag_id, persona_and_style_id in pairs:
            predicted_persona_style_ids.add(persona_and_style_id)

        assert len(predicted_persona_style_ids) == 3


if __name__ == '__main__':
    unittest.main()
