import pandas as pd
import unittest
from typing import List

from benchmarking.evaluate_results import EntityAnalyzer, ResultDataType

'''Run python -m unittest datageneration.tests.benchmarking.test_entity_analyzer'''


class TestEvaluation(unittest.TestCase):
    def setUp(self):
        pass

    def test_pipeline(self):
        entity_analyzer = EntityAnalyzer()

        predicted_result = entity_analyzer.compare_entities(
            entities1=[{'id': 0, 'type': 'nwr', 'name': 'elementary schools'},
                       {'id': 1, 'type': 'nwr', 'name': 'library'}, {'id': 2, 'type': 'nwr', 'name': 'driving school'}],
            entities2=[{'id': 0, 'is_area': True, 'name': 'elementary schools', 'type': 'nwr'},
                       {'id': 1, 'is_area': False, 'name': 'library', 'type': 'nwr'},
                       {'id': 2, 'is_area': False, 'name': 'driving school', 'properties': [{'name': 'door'}],
                        'type': 'nwr'}])
        self.assertEquals(predicted_result, ResultDataType.FALSE)

        predicted_result = entity_analyzer.compare_entities_exclude_props(
            entities1=[{'id': 0, 'type': 'nwr', 'name': 'elementary schools'},
                       {'id': 1, 'type': 'nwr', 'name': 'library'},
                       {'id': 2, 'type': 'nwr', 'name': 'driving school'}],
            entities2=[{'id': 0, 'is_area': True, 'name': 'elementary schools', 'type': 'nwr'},
                       {'id': 1, 'is_area': False, 'name': 'library', 'type': 'nwr'},
                       {'id': 2, 'is_area': False, 'name': 'driving school', 'properties': [{'name': 'door'}],
                        'type': 'nwr'}])
        self.assertEquals(predicted_result, ResultDataType.TRUE)



        predicted_result = entity_analyzer.compare_entities_partial_match_exclude_props(
            entities1=[{'id': 0, 'type': 'nwr', 'name': 'elementary schools'},
                       {'id': 1, 'type': 'nwr', 'name': 'library'},
                       {'id': 2, 'type': 'nwr', 'name': 'driving school'}],
            entities2=[{'id': 0, 'is_area': True, 'name': 'elementary schools', 'type': 'nwr'},
])

        self.assertEquals(predicted_result, ResultDataType.TRUE)