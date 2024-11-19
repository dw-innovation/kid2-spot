import unittest
import pandas as pd
from typing import List
from datageneration.area_generator import AreaGenerator, load_named_area_data, NamedAreaData
from datageneration.data_model import Area

'''Run python -m unittest datageneration.tests.test_area_generator'''


class TestAreaGenerator(unittest.TestCase):
    def setUp(self):
        self.geolocation_file = 'datageneration/tests/data/countries+states+cities.json'
        non_roman_vocab_file = 'datageneration/tests/data/area_non_roman_vocab.json'
        self.area_generator = AreaGenerator(geolocation_file=self.geolocation_file, non_roman_vocab_file=non_roman_vocab_file, prob_of_two_word_areas=0.5,
                                            prob_of_non_roman_areas=0.2)

        (self.locs_with_cities_single_word, self.locs_with_cities_two_words, self.locs_with_states_single_word,
         self.locs_with_states_two_words) = load_named_area_data(geolocation_file=self.geolocation_file)
        self.geolocation_data = [*self.locs_with_cities_single_word, *self.locs_with_cities_two_words]

    # def test_load_named_area_data(self):
    #
    #     must_example = NamedAreaData(city='Koblenz', state='Rhineland-Palatinate', country='Germany')
    #     assert must_example in self.locs_with_cities_single_word
    #     assert must_example in self.locs_with_states_single_word
    #     assert must_example not in self.locs_with_cities_two_words
    #     assert must_example not in self.locs_with_states_two_words
    #
    #     # todo: this is an example where there is no state, this does not work!!!
    #     # must_example = NamedAreaData(city='İzmir', state=None, country='Turkey')
    #     # assert must_example in area_data
    #
    # def test_generate_no_area(self):
    #     no_area_sample = self.area_generator.generate_no_area()
    #     assert no_area_sample == Area(type='bbox', value='')
    #
    # def test_generate_city_area(self):
    #     city_sample = self.area_generator.generate_city_area()
    #     assert city_sample.type == 'area'
    #     assert len(city_sample.value) > 1
    #
    # def test_generate_city_and_country_area(self):
    #     city_sample = self.area_generator.generate_city_and_country_area()
    #     assert city_sample.type == 'area'
    #     assert len(city_sample.value) > 1
    #     assert ',' in city_sample.value
    #     country = city_sample.value.split(',')[-1].lstrip().strip()
    #     country_present = any(area.country == country for area in self.geolocation_data)
    #     assert country_present
    #
    # def test_generate_city_and_region_and_country(self):
    #     city_sample = self.area_generator.generate_city_and_region_and_country_area()
    #     assert city_sample.type == 'area'
    #     assert len(city_sample.value) > 1
    #     assert ',' in city_sample.value
    #
    #     splits = city_sample.value.split(',')
    #
    #     country = splits[-1].lstrip().strip()
    #     country_present = any(area.country == country for area in self.geolocation_data)
    #     assert country_present
    #
    #     state = splits[1].lstrip().strip()
    #     state_present = any(area.state == state for area in self.geolocation_data)
    #     assert state_present
    #
    # def test_generate_region(self):
    #     state_sample = self.area_generator.generate_region_area()
    #     assert state_sample.type == 'area'
    #     assert len(state_sample.value) > 1
    #
    #     state_present = any(area.state == state_sample.value for area in self.geolocation_data)
    #     assert state_present
    #
    # def test_two_word_areas(self):
    #     for city in self.locs_with_cities_single_word:
    #         assert len(city.city.split()) == 1
    #
    #     for city in self.locs_with_cities_two_words:
    #         assert len(city.city.split()) > 1
    #
    #     for city in self.locs_with_states_single_word:
    #         assert len(city.state.split()) == 1
    #
    #     for city in self.locs_with_states_two_words:
    #         assert len(city.state.split()) > 1
    #
    #
    # def test_area_run(self):
    #     '''
    #     Run function must return non-empty, area object
    #     '''
    #     area_sample = self.area_generator.run()
    #
    #     assert area_sample
    #     assert isinstance(area_sample, Area)

    def test_non_roman_area(self):
        area = NamedAreaData(city='Barrón', state='Guanajuato', country='Mexico')
        target_lang = 'myv'
        self.area_generator.translate_into_non_roman(area=area, target_lang=target_lang)