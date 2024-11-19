import json
import numpy as np
import pandas as pd
from enum import Enum
from pydantic import BaseModel
from typing import List, Dict, Union

from datageneration.data_model import Area
from datageneration.enrich_geo_database import area_names_non_roman_vocab
from datageneration.utils import NON_ROMAN_LANG_GROUPS


class NamedAreaData(BaseModel):
    city: str
    state: str
    country: str


class AREA_TASKS(Enum):
    NO_AREA = 'no_area'
    CITY = 'city'
    CITY_AND_COUNTRY = 'city_and_country'
    REGION = 'administrative_region'
    REGION_AND_COUNTRY = 'administrative_region'
    CITY_AND_REGION_AND_COUNTRY = 'city_and_region_and_country'


def load_named_area_data(geolocation_file: str) -> List[NamedAreaData]:
    """
    Load the geo database and categorise them into four groups. It differentiates between one and two word cities and
    regions respectively and returns them as a list.

    :param geolocation_file: the path to the geo database file
    :return: four lists for one or two word cities or regions respectively
    """
    geolocation_data = pd.read_json(geolocation_file)

    processed_geolocation_data = []
    locs_with_cities_single_word = []
    locs_with_cities_two_words = []
    locs_with_states_single_word = []
    locs_with_states_two_words = []
    for sample in geolocation_data.to_dict(orient='records'):
        for state in sample['states']:
            for city in state['cities']:
                named_area = NamedAreaData(city=city['name'], state=state['name'], country=sample['name'])
                if len(state['name'].split()) > 1:
                    locs_with_states_two_words.append(named_area)
                else:
                    locs_with_states_single_word.append(named_area)
                if len(city['name'].split()) > 1:
                    locs_with_cities_two_words.append(named_area)
                else:
                    locs_with_cities_single_word.append(named_area)
                processed_geolocation_data.append(named_area)

    return (locs_with_cities_single_word, locs_with_cities_two_words, locs_with_states_single_word,
            locs_with_states_two_words)


def load_non_roman_vocab(non_roman_vocab_file: str) -> Dict:
    with open(non_roman_vocab_file, 'r') as json_file:
        non_roman_vocab = json.load(json_file)
    return non_roman_vocab


class NonRomanDoesNotExist(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return f"NonRomanDoesNotExist: {self.message}"


class AreaGenerator:
    def __init__(self, geolocation_file: str, non_roman_vocab_file: str, prob_of_two_word_areas: float,
                 prob_of_non_roman_areas: float):
        (self.locs_with_cities_single_word, self.locs_with_cities_two_words, self.locs_with_states_single_word,
         self.locs_with_states_two_words) = load_named_area_data(geolocation_file)
        self.area_non_roman_vocab = load_non_roman_vocab(non_roman_vocab_file)
        self.tasks = [area_task.value for area_task in AREA_TASKS]
        self.two_word_city_selection = [True, False]
        self.two_word_state_selection = [True, False]
        self.prob_of_two_word_areas = prob_of_two_word_areas
        self.prob_of_non_roman_areas = prob_of_non_roman_areas
        self.non_roman_lang_group_keys = list(NON_ROMAN_LANG_GROUPS.keys())

    def translate_into_non_roman(self, area: NamedAreaData, target_lang_group: List) -> Union[NamedAreaData, None]:
        target_langs = NON_ROMAN_LANG_GROUPS[target_lang_group]
        np.random.shuffle(target_langs)

        for target_lang in target_langs:
            if any((area.city not in self.area_non_roman_vocab,
                    area.state not in self.area_non_roman_vocab,
                    area.country not in self.area_non_roman_vocab)):
                raise NonRomanDoesNotExist(f"any of city, state, country in {area} does not have a non-roman version")
            if any((target_lang not in self.area_non_roman_vocab[area.city]['non_roman_versions'],
                    target_lang not in self.area_non_roman_vocab[area.state]['non_roman_versions'],
                    target_lang not in self.area_non_roman_vocab[area.country]['non_roman_versions'])):
                continue
            translated_city = self.area_non_roman_vocab[area.city]['non_roman_versions'][target_lang]
            translated_state = self.area_non_roman_vocab[area.state]['non_roman_versions'][target_lang]
            translated_country = self.area_non_roman_vocab[area.country]['non_roman_versions'][target_lang]
            return NamedAreaData(city=translated_city, state=translated_state, country=translated_country)

    def get_area(self, required_type=None) -> NamedAreaData:
        """
        A method that returns a random area. Probability of one or two word areas is determined by a class variable.
        The argument "required_type" can determine whether the one or two word specification must apply to either
        "city" or "state", in case that is the only used field of the two.

        :param required_type: determines what will be set to one/two words, either "city" or "state" or None for random
        :return: the drawn area
        """
        use_two_word_area = np.random.choice([True, False], p=[self.prob_of_two_word_areas,
                                                               1 - self.prob_of_two_word_areas])

        if use_two_word_area:
            if required_type == "city":
                draft_from = self.locs_with_cities_two_words
            elif required_type == "state":
                draft_from = self.locs_with_states_two_words
            else:
                idx = np.random.choice([0, 1])
                draft_from = [self.locs_with_cities_two_words, self.locs_with_states_two_words][idx]
        else:
            if required_type == "city":
                draft_from = self.locs_with_cities_single_word
            elif required_type == "state":
                draft_from = self.locs_with_states_single_word
            else:
                idx = np.random.choice([0, 1])
                draft_from = [self.locs_with_cities_single_word, self.locs_with_states_single_word][idx]

        selected_area = np.random.choice(draft_from)

        use_non_roman_alphabets = np.random.choice([True, False], p=[self.prob_of_non_roman_areas,
                                                                     1 - self.prob_of_non_roman_areas])

        if use_non_roman_alphabets:
            np.random.shuffle(self.non_roman_lang_group_keys)
            tmp_selected_area = None
            for selected_lang_group in self.non_roman_lang_group_keys:
                try:
                    tmp_selected_area = self.translate_into_non_roman(area=selected_area,
                                                                      target_lang_group=selected_lang_group)
                except NonRomanDoesNotExist:
                    selected_area = np.random.choice(draft_from)

                if not tmp_selected_area:
                    continue

                else:
                    selected_area = tmp_selected_area
                    break
        return selected_area


    def generate_no_area(self) -> Area:
        '''
        It returns no area, bbox
        '''
        return Area(type='bbox', value='')


    def generate_city_area(self) -> Area:
        '''
        Randomly select if select the city from cities with single word or two word. After corresponding category is
        selected, we will suffle the corresponding list and then select the city name
        e.g. Koblenz
        '''
        area = self.get_area("city")

        return Area(type='area', value=area.city)


    def generate_city_and_country_area(self) -> Area:
        '''
        Randomly shuffles the geolocation data point
        Selects the city name and return city_name, country_name where city is located.
        e.g Koblenz, Germany
        '''
        area = self.get_area("city")

        return Area(type='area', value=f'{area.city}, {area.country}')


    def generate_region_area(self) -> Area:
        '''
        It filters the unique states in geolocation data points
        Randomly shuffles it
        Selects the state
        e.g Rheinland-Palastine
        '''
        area = self.get_area("state")

        return Area(type='area', value=area.state)


    def generate_region_and_country_area(self) -> Area:
        '''
        Randomly shuffles the geolocation data point
        Selects the city name and return state_name and then country_name where city is located.
        e.g Koblenz, Rheinland-Palastine, Germany
        '''
        # todo: this would be problematic when the country does not have states
        area = self.get_area("state")

        return Area(type='area', value=f'{area.state}, {area.country}')


    def generate_city_and_region_and_country_area(self) -> Area:
        '''
        Randomly shuffles the geolocation data point
        Selects the city name and return city_name, state_name and then country_name where city is located.
        e.g Koblenz, Rheinland-Palastine, Germany
        '''
        # todo: this would be problematic when the country does not have states
        area = self.get_area()

        return Area(type='area', value=f'{area.city}, {area.state}, {area.country}')


    def run(self) -> List[Area]:
        '''
        This function a random generation pipeline. That randomly selects the task function which are defined in AREA_TASKS. Next, it calls the generator function that is corresponding to the selected task.
        '''
        np.random.shuffle(self.tasks)
        selected_task = self.tasks[0]

        if selected_task == AREA_TASKS.NO_AREA.value:
            return self.generate_no_area()

        elif selected_task == AREA_TASKS.CITY.value:
            return self.generate_city_area()

        # elif selected_task == AREA_TASKS.DISTRICT.value:
        #     # todo: we probably need more comprehensive geolocation data, districs are not in the dataset
        #     return NotImplemented

        elif selected_task == AREA_TASKS.CITY_AND_COUNTRY.value:
            return self.generate_city_and_country_area()

        elif selected_task == AREA_TASKS.REGION.value:
            return self.generate_region_area()

        elif selected_task == AREA_TASKS.REGION_AND_COUNTRY.value:
            return self.generate_region_and_country_area()

        elif selected_task == AREA_TASKS.CITY_AND_REGION_AND_COUNTRY.value:
            return self.generate_city_and_region_and_country_area()
