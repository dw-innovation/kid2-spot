import pandas as pd
import unittest
from typing import List

from benchmarking.evaluate_results import PropertyAnalyzer, ResultDataType

'''Run python -m unittest datageneration.tests.benchmarking.test_property_analyzer'''


class TestEvaluation(unittest.TestCase):
    def setUp(self):
        pass

    def test_pipeline(self):
        property_analyzer = PropertyAnalyzer()

        ref_entities = {
            'social facility': [{'name': 'building levels', 'operator': '<', 'value': 3},
                                {'name': 'name', 'operator': '~', 'value': 'ole'}]
        }

        prop_entities = {
            'social facility': [{'name': 'building levels', 'operator': '<', 'value': '3'},
                                {'name': 'name', 'operator': '~', 'value': 'ole'}]
        }

        results = property_analyzer.percentage_of_correctly_identified_properties(ref_entities, prop_entities)

        self.assertEquals(results, 1)
