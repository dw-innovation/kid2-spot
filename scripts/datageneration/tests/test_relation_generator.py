import unittest

from datageneration.relation_generator import RelationGenerator

'''Run python -m unittest datageneration.tests.test_relation_generator'''


class TestPropertyGenerator(unittest.TestCase):
    def setUp(self):
        self.relation_generator = RelationGenerator(max_distance=2000)

    def test_individual_distances(self):
        relations = self.relation_generator.generate_individual_distances(num_entities=3)
        raise NotImplemented

    def test_within_radius(self):
        relations = self.relation_generator.generate_within_radius(num_entities=3)
        raise NotImplemented
