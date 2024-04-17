import unittest

from datageneration.relation_generator import RelationGenerator

'''Run python -m unittest datageneration.tests.test_relation_generator'''

class TestRelationGenerator(unittest.TestCase):
    def setUp(self):
        self.relation_generator = RelationGenerator(max_distance=2000)

    def test_individual_distances(self):
        relations = self.relation_generator.generate_individual_distances(num_entities=3)
        dists = [r.value for r in relations]
        sources = [r.source for r in relations]
        targets = [r.target for r in relations]

        assert 1 <= len(set(dists)) <= len(relations)
        assert len(set(sources)) == len(relations)
        assert len(set(targets)) == len(relations)

    def test_within_radius(self):
        relations = self.relation_generator.within_radius(num_entities=3)

        dists = [r.value for r in relations]
        sources = [r.source for r in relations]
        targets = [r.target for r in relations]

        assert len(set(dists)) == 1
        assert len(set(sources)) == 1
        assert len(set(targets)) == len(relations)

    def test_in_area(self):
        relations = self.relation_generator.generate_in_area(num_entities=1)

        assert relations == None