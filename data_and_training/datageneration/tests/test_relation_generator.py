import unittest

from unittest.mock import patch
from datageneration.data_model import Entity, Property, Relation, Distance
from datageneration.relation_generator import RelationGenerator

'''Run python -m unittest datageneration.tests.test_relation_generator'''

class TestRelationGenerator(unittest.TestCase):
    def setUp(self):
        self.relation_generator = RelationGenerator(max_distance_digits=5, prob_generating_contain_rel=0.5,
                                                    ratio_within_radius_within=0.5)

    def test_individual_distances(self):
        entity_ids = [0, 1, 2]
        relations = self.relation_generator.generate_individual_distances(entity_ids=entity_ids)
        dists = [r.value for r in relations]
        sources = [r.source for r in relations]
        targets = [r.target for r in relations]

        assert 1 <= len(set(dists)) <= len(relations)
        assert len(set(sources)) == len(relations)
        assert len(set(targets)) == len(relations)

    def test_in_area(self):
        relations = self.relation_generator.generate_in_area(num_entities=1)

        assert relations == None

    def test_within_radius(self):
        relations = self.relation_generator.generate_within_radius(num_entities=3)

        dists = [r.value for r in relations]
        sources = [r.source for r in relations]
        targets = [r.target for r in relations]

        assert len(set(dists)) == 1
        assert len(set(sources)) == 1
        assert len(set(targets)) == len(relations)

    def test_contain_rel(self):
        area_entities = [Entity(id=0, is_area=True, name='astro station', type='nwr', properties=[])]
        point_entities = [Entity(id=1, is_area=False, name='block', type='nwr',
                           properties=[Property(name='height', operator='=', value='0.6 m')]),
                          Entity(id=2, is_area=False, name='scuba center', type='nwr', properties=[])]

        # case 1, expect no Exception
        try:
            relations = self.relation_generator.generate_relation_with_contain(area_entities=area_entities,
                                                                point_entities=point_entities, max_within_combs=1)
        except AssertionError:
            raise RuntimeError('The test should not be failed! Something is wrong.')

        # case 2, one area entity and the others are ...
        area_entity = [Entity(id=0, is_area=True, name='astro station', type='nwr', properties=[])]
        point_entities_connecting_to_area_entity = [[Entity(id=1, is_area=False, name='block', type='nwr',
                                                           properties=[
                                                               Property(name='height', operator='=',
                                                                        value='0.6 m')]),
                                                    Entity(id=2, is_area=False, name='scuba center', type='nwr',
                                                           properties=[])]]

        relations = self.relation_generator.generate_relation_with_contain_helper(drawn_area_entities=area_entity,
                    point_entities_connecting_to_area_entity=point_entities_connecting_to_area_entity)
        expected_relations = [Relation(type='contains', source=0, target=1, value=None),
                              Relation(type='contains', source=0, target=2, value=None)]

        self.assertEqual(relations, expected_relations)

        # Check the combination with "individual distances"
        area_entity = [Entity(id=0, is_area=True, name='astro station', type='nwr', properties=[])]
        point_entities_connecting_to_area_entity = [[
            Entity(id=1, is_area=False, name='scuba center', type='nwr', properties=[])]]

        relations = self.relation_generator.generate_relation_with_contain_helper(
                                drawn_area_entities=area_entity,
                                point_entities_connecting_to_area_entity=point_entities_connecting_to_area_entity)

        other_point_entities = [Entity(id=2, is_area=False, name='block', type='nwr',
                                       properties=[Property(name='height', operator='=', value='0.6 m')])]
        other_point_entities = point_entities_connecting_to_area_entity[0] + other_point_entities
        other_entity_ids = [e.id for e in other_point_entities]

        relations.extend(self.relation_generator.generate_individual_distances(entity_ids=other_entity_ids))
        expected_relations = [Relation(type='contains', source=0, target=1, value=None),
                              Relation(type='distance', source=1, target=2, value=Distance(magnitude="100", metric="m"))]

        relations[1].value = Distance(magnitude="100", metric="m")

        self.assertEqual(relations, expected_relations)