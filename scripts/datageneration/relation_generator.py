from enum import Enum
from typing import List

import numpy as np
from datageneration.data_model import Relation


class RELATION_TASKS(Enum):
    INDIVIDUAL_DISTANCES = 'individual_distances'
    IN_AREA = 'in_area'
    WITHIN_RADIUS = 'within_radius'


def get_random_decimal_with_metric(range):
    '''
    TODO: this should be reworked -- threshold should be defined based on metric
    '''
    h_ = np.random.choice(np.arange(range), 1)[0]
    if np.random.choice([True, False], 1)[0]:
        h_ = h_ / np.random.choice([10, 100], 1)[0]

    h_ = str(h_) + " " + np.random.choice(["m", "km", "in", "ft", "yd", "mi", "le"], 1)[0]  # "cm",
    return h_


class RelationGenerator:
    def __init__(self, max_distance: int):
        self.MAX_DISTANCE = max_distance
        self.tasks = [relation_task.value for relation_task in RELATION_TASKS]

    def generate_individual_distances(self, num_entities: int) -> List[Relation]:
        relations = []
        for t_no in range(num_entities):
            if t_no != num_entities - 1:
                relations.append(
                    Relation(name='dist', source=t_no, target=t_no + 1,
                             value=get_random_decimal_with_metric(self.MAX_DISTANCE)))

    def within_radius(self, num_entities: int) -> List[Relation]:
        relations = []
        for t_no in range(num_entities):
            if t_no != num_entities - 1:
                relations.append(
                    Relation(name='dist', source=0, target=t_no + 1,
                             value=get_random_decimal_with_metric(self.MAX_DISTANCE)))
        return relations

    def run(self, num_entities: int) -> List[Relation]:
        np.random.shuffle(self.tasks)
        selected_task = self.tasks[0]
        print(selected_task)
        if selected_task == RELATION_TASKS.INDIVIDUAL_DISTANCES.value and num_entities > 2:  # Pick random distance between all individual objects
            relations = self.generate_individual_distances(num_entities=num_entities)
        elif selected_task == RELATION_TASKS.IN_AREA.value or num_entities == 1:  # Just search for all given objects in area, no distance required
            relations = None
        elif selected_task == RELATION_TASKS.WITHIN_RADIUS.value:  # Search for all places where all objects are within certain radius
            relations = self.within_radius(num_entities=num_entities)
        return relations