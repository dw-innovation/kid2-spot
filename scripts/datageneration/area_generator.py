import pandas as pd
import numpy as np
from enum import Enum
from pydantic import BaseModel
from typing import List
from datageneration.data_model import Area


class NamedAreaData(BaseModel):
    city: str
    state: str
    country: str


class AREA_TASKS(Enum):
    NO_AREA = 'no_area'
    # DISTRICT = 'district'
    CITY = 'city'
    CITY_AND_COUNTRY = 'city_and_country'
    CITY_AND_REGION_AND_COUNTRY = 'city_and_region_and_country'
    ADMINISTRATIVE_REGION = 'administrative_region'

    # todo: two word area


def load_named_area_data(geolocation_file: str) -> List[NamedAreaData]:
    geolocation_data = pd.read_json(geolocation_file)

    processed_geolocation_data = []
    for sample in geolocation_data.to_dict(orient='records'):
        for state in sample['states']:
            for city in state['cities']:
                processed_geolocation_data.append(
                    NamedAreaData(city=city['name'], state=state['name'], country=sample['name']))
    return processed_geolocation_data


class AreaGenerator:
    def __init__(self, geolocation_data: List[NamedAreaData]):
        self.geolocation_data = geolocation_data
        self.tasks = [area_task.value for area_task in AREA_TASKS]

    def generate_no_area(self) -> Area:
        '''
        It returns no area, bbox
        '''
        return Area(type='bbox', value='')

    def generate_city_area(self) -> Area:
        '''
        Randomly shuffles the geolocation data point
        Selects the city name
        e.g. Koblenz
        '''
        np.random.shuffle(self.geolocation_data)
        selected_area = self.geolocation_data[0]
        return Area(type='area', value=selected_area.city)

    def generate_city_and_country_area(self) -> Area:
        '''
        Randomly shuffles the geolocation data point
        Selects the city name and return city_name, country_name where city is located.
        e.g Koblenz, Germany
        '''
        np.random.shuffle(self.geolocation_data)
        selected_area = self.geolocation_data[0]
        return Area(type='area', value=f'{selected_area.city}, {selected_area.country}')

    def generate_city_and_region_and_country(self) -> Area:
        '''
        Randomly shuffles the geolocation data point
        Selects the city name and return city_name, state_name and then country_name where city is located.
        e.g Koblenz, Rheinland-Palastine, Germany
        '''
        # todo: this would be problematic when the country does not have states
        np.random.shuffle(self.geolocation_data)
        selected_area = self.geolocation_data[0]
        return Area(type='area', value=f'{selected_area.city}, {selected_area.state}, {selected_area.country}')

    def generate_administrative_region(self) -> Area:
        '''
        It filters the unique states in geolocation data points
        Randomly shuffles it
        Selects the state
        e.g Rheinland-Palastine
        '''
        states = [area.state for area in self.geolocation_data]
        np.random.shuffle(states)
        selected_state = states[0]
        return Area(type='area', value=selected_state)

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

        elif selected_task == AREA_TASKS.CITY_AND_REGION_AND_COUNTRY.value:
            return self.generate_city_and_region_and_country()

        elif selected_task == AREA_TASKS.ADMINISTRATIVE_REGION.value:
            return self.generate_administrative_region()
