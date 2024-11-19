import numpy as np
from enum import Enum
from typing import List
import copy

from datageneration.data_model import Relation, Relations, Entity
from datageneration.utils import get_random_decimal_with_metric


RELATION_TYPE='distance'

class RELATION_TASKS(Enum):
    INDIVIDUAL_DISTANCES = 'individual_distances'
    WITHIN_RADIUS = 'within_radius'
    IN_AREA = 'in_area'

class RelationGenerator:
    def __init__(self, max_distance_digits: int, prob_generating_contain_rel: float):
        self.MAX_DISTANCE_DIGITS = max_distance_digits
        self.prob_generating_contain_rel = prob_generating_contain_rel
        self.tasks = [relation_task.value for relation_task in RELATION_TASKS]

    def generate_individual_distances(self, entity_ids: List[int]) -> List[Relation]:
        # np.random.shuffle(entity_ids)
        relations = []
        for t_no in range(len(entity_ids)-1):
            relations.append(
                Relation(type=RELATION_TYPE, source=entity_ids[t_no], target=entity_ids[t_no+1],
                         value=get_random_decimal_with_metric(self.MAX_DISTANCE_DIGITS)))
        return relations

    def generate_within_radius(self, num_entities: int) -> List[Relation]:
        """
        Generate relations representing entities within a certain radius.
        Args:
            num_entities (int): The number of entities for which relations need to be generated.
        Returns:
            List[Relation]: A list of Relation objects representing entities within a radius.
        """
        relations = []
        distance = get_random_decimal_with_metric(self.MAX_DISTANCE_DIGITS)
        for t_no in range(num_entities):
            if t_no != num_entities - 1:
                relations.append(
                    Relation(type=RELATION_TYPE, source=0, target=t_no + 1,
                             value=distance))
        return relations

    def generate_in_area(self, num_entities: int) -> None:
        '''
        It returns None, that indicates that the relation is not clear or one object exists
        '''
        return None

    def generate_relation_with_contain(self, area_entities: List[Entity], point_entities: List[Entity],
                                        max_within_combs: int) -> Relations:
        """
        This method generates relations that include at least one relation of type "contains". The contains relations
        are randomly drawn based on the possible combinations of "area" and "point" entities. Depending on the number
        of "contains" groups (meaning groups of areas and one or multiple entities contained within it), and other
        entities not part of the "contains" groups, one of three relation types is possible:
            - individual_distances_with_contains: Requires any combination of at least two groups and/or entities
            - contains_within_radius: Requires only one group, but at least two points connected to the area
            - contains_relation: Requires only one group, with any number of points connected to area

        :param area_entities: The entities of type "area"
        :param point_entities: The entities of type "point"
        :param max_within_combs: The maximum possible number of "contains" relation possible based on the entities
        :return: The generated relations
        """
        num_within_combs = np.random.choice(np.arange(1,max_within_combs+1))
        remaining_area_entities = copy.deepcopy(area_entities)
        remaining_point_entities = copy.deepcopy(point_entities)
        remaining_num_possible_connections = len(point_entities)
        drawn_area_entities = []
        point_entities_connecting_to_area_entity = []

        for area_num in range(1,num_within_combs+1):
            # For each contains relation, draw one are entity and filter it from the "remaining_areas" list
            area_entity = np.random.choice(remaining_area_entities)
            remaining_area_entities = [e for e in remaining_area_entities if e != area_entity]
            drawn_area_entities.append(area_entity)

            # Randomly select one or multiple entities that will be contained in this area, leave enough behind for
            # all other "contains" areas in this query, filter drawn entities from "remaining_entities" list
            num_of_point_entities_connecting_to_area_entity = \
                np.random.choice(np.arange(1,remaining_num_possible_connections-num_within_combs+area_num+1))
            remaining_num_possible_connections -= num_of_point_entities_connecting_to_area_entity
            np.random.shuffle(remaining_point_entities)
            point_entities_connecting_to_area_entity.append(
                remaining_point_entities[:num_of_point_entities_connecting_to_area_entity])
            for point_entity_connecting_to_area_entity in point_entities_connecting_to_area_entity:
                remaining_point_entities = [e for e in remaining_point_entities if e not in
                                            point_entity_connecting_to_area_entity]

            # these are must rule, helper for unittest
            assert len(point_entities_connecting_to_area_entity[-1]) == num_of_point_entities_connecting_to_area_entity
            # assert len(remaining_point_entities) == len(point_entities) - remaining_num_possible_connections
            assert point_entities_connecting_to_area_entity[-1] != remaining_point_entities

        # "Other entities" are all not in "contains relations"
        other_entities = [*remaining_point_entities, *remaining_area_entities]
        # Extend other drawn entities to the first point entity of each "contains" group, as they are used to show
        # individual distances between this group and other entities
        other_entities.extend([e[0] for e in point_entities_connecting_to_area_entity])
        other_entity_ids = [e.id for e in other_entities]

        assert len(drawn_area_entities) == len(point_entities_connecting_to_area_entity)

        # Check if only one "contains" group is present, otherwise use "individual_distances_with_contains"
        if len(other_entity_ids) > 1:
            relations = self.generate_relation_with_contain_helper(drawn_area_entities,
                                               point_entities_connecting_to_area_entity)
            relations.extend(self.generate_individual_distances(other_entity_ids))
            relation_type = "individual_distances_with_contains"
        else:
            relations = self.generate_relation_with_contain_helper(drawn_area_entities,
                                               point_entities_connecting_to_area_entity)
            relation_type = "contains_relation"

        return Relations(type=relation_type, relations=relations)

    def generate_relation_with_contain_helper(self, drawn_area_entities: List[Entity],
             point_entities_connecting_to_area_entity: List[List[Entity]]) -> List[Relation]:
        """
        Generate the relation format for contains relations. A distance value (replicating a "contains_within_radius"
        relation) is only assigned if the argument add_dist is True.

        :param drawn_area_entities: the entities that server as areas in the contains relations
        :param point_entities_connecting_to_area_entity: a list of connected points for each area
        :param add_dist: boolean whether distances should be assigned, or None
        :return: the list of relations
        """
        relations = []
        for aid, area in enumerate(drawn_area_entities):
            for point in point_entities_connecting_to_area_entity[aid]:
                    relations.append(
                        Relation(type='contains', source=area.id, target=point.id))

        return relations

    def get_task(self, num_entities: int):
        """
        This method of selecting the task of the query using an exponential decay method. It first filters out
        tasks that are not viable due to the number of entities. It then selects the task based on a probability
        distribution that assigns higher probabilities to task at the beginning of the list. This allows for
        a drafting system that prioritises certain tasks over others, as e.g. "individual_distances" is a
        far more difficult task than "in_area" and therefore requires more training data.

        Example probability distribution with decay of 0.5:
            - 3 or more entities: [0.50648039 0.30719589 0.18632372]
            - 2 entities: [0.62245933 0.37754067]
            - 1 entity: [1.0]

        :param num_entities: The number of entities of the query
        :return: The selected task
        """
        viable_tasks = [RELATION_TASKS.INDIVIDUAL_DISTANCES.value, RELATION_TASKS.WITHIN_RADIUS.value,
                        RELATION_TASKS.IN_AREA.value]
        if num_entities < 3:
            viable_tasks.pop(0)
        if num_entities == 1:
            viable_tasks.pop(0)

        decay_rate = 0.5
        task_nums = np.arange(1, len(viable_tasks) + 1)
        probabilities = np.exp(-decay_rate * task_nums)
        probabilities /= np.sum(probabilities)
        selected_task = np.random.choice(viable_tasks, p=probabilities)

        return selected_task
      
    def run(self, entities: List[Entity]) -> Relations:
        """
        This task runs the general pipeline for generating relations between entities.
        The specific task for relation generation is randomly selected.
        Once it is defined, it will execute the corresponding function.
        Args:
            entities (List): The entities involved in the task.

        Returns:
            List[Relation] or None: A list of Relation objects representing the task outcome.
        """
        area_entities = []
        point_entities = []
        for id, entity in enumerate(entities):
            if entity.is_area:
                area_entities.append(entity)
            else:
                point_entities.append(entity)
        max_within_combs = min(len(area_entities), len(point_entities))

        generating_contain_rel = np.random.choice([True, False], p=[self.prob_generating_contain_rel,
                                                                    1 - self.prob_generating_contain_rel])
        if generating_contain_rel and max_within_combs>0:
            relations = self.generate_relation_with_contain(area_entities, point_entities, max_within_combs)
        else:
            relations = self.standard_rel_tasks(np.arange(len(entities)))
        return relations

    def standard_rel_tasks(self, entity_ids):
        num_entities = len(entity_ids)
        selected_task = self.get_task(num_entities)
        if selected_task == RELATION_TASKS.INDIVIDUAL_DISTANCES.value:
            relations = Relations(type=selected_task,
                                  relations=self.generate_individual_distances(entity_ids=entity_ids))
        elif selected_task == RELATION_TASKS.WITHIN_RADIUS.value:  # Just search for all given objects in area, no distance required
            relations = Relations(type=selected_task,
                                  relations=self.generate_within_radius(num_entities=num_entities))
        elif selected_task == RELATION_TASKS.IN_AREA.value:
            relations = Relations(type=selected_task,
                                  relations=self.generate_in_area(num_entities=num_entities))
        return relations