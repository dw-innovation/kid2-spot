import unittest
import pandas as pd
from typing import List
from datageneration.area_generator import AreaGenerator, load_named_area_data, NamedAreaData
from datageneration.data_model import Area

'''Run python -m unittest datageneration.tests.test_area_generator'''


class TestAreaGenerator(unittest.TestCase):
    def setUp(self):
        self.geolocation_file = 'datageneration/tests/data/countries+states+cities.json'
        self.geolocation_data = load_named_area_data(self.geolocation_file)
        self.area_generator = AreaGenerator(geolocation_data=self.geolocation_data)

    def test_load_named_area_data(self):
        area_data = load_named_area_data(geolocation_file=self.geolocation_file)

        must_example = NamedAreaData(city='Koblenz', state='Rhineland-Palatinate', country='Germany')
        assert must_example in area_data

        # todo: this is an example where there is no state, this does not work!!!
        # must_example = NamedAreaData(city='İzmir', state=None, country='Turkey')
        # assert must_example in area_data

    def test_generate_no_area(self):
        no_area_sample = self.area_generator.generate_no_area()
        assert no_area_sample == Area(type='bbox', value='')

    def test_generate_city_area(self):
        city_sample = self.area_generator.generate_city_area()
        assert city_sample.type == 'area'
        assert len(city_sample.value) > 1

    def test_generate_city_and_country_area(self):
        city_sample = self.area_generator.generate_city_and_country_area()
        assert city_sample.type == 'area'
        assert len(city_sample.value) > 1
        assert ',' in city_sample.value
        country = city_sample.value.split(',')[-1].lstrip().strip()
        country_present = any(area.country == country for area in self.geolocation_data)
        assert country_present

    def test_generate_city_and_region_and_country(self):
        city_sample = self.area_generator.generate_city_and_region_and_country()
        assert city_sample.type == 'area'
        assert len(city_sample.value) > 1
        assert ',' in city_sample.value

        splits = city_sample.value.split(',')

        country = splits[-1].lstrip().strip()
        country_present = any(area.country == country for area in self.geolocation_data)
        assert country_present

        state = splits[1].lstrip().strip()
        state_present = any(area.state == state for area in self.geolocation_data)
        assert state_present

    def test_generate_administrative_region(self):
        state_sample = self.area_generator.generate_administrative_region()
        assert state_sample.type == 'area'
        assert len(state_sample.value) > 1

        state_present = any(area.state == state_sample.value for area in self.geolocation_data)
        assert state_present

    def test_two_word_areas(self):
        # todo not implemented
        pass

    def test_area_run(self):
        '''
        Run function must return non-empty, area object
        '''
        area_sample = self.area_generator.run()

        assert area_sample
        assert isinstance(area_sample, Area)
