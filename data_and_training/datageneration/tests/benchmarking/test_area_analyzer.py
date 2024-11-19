import pandas as pd
import unittest
from typing import List

from benchmarking.evaluate_results import AreaAnalyzer, ResultDataType

'''Run python -m unittest datageneration.tests.benchmarking.test_area_analyzer'''


class TestEvaluation(unittest.TestCase):
    def setUp(self):
        pass

    def test_pipeline(self):
        area_analyzer = AreaAnalyzer()

        predicted_result = area_analyzer.compare_areas_strict(area1={'type': 'area', 'value': 'Kadayanallur, India'}, area2={'type': 'area', 'value': 'Kadayanallur, india'})
        self.assertEquals(predicted_result, ResultDataType.FALSE)

        predicted_result = area_analyzer.compare_areas_light(area1={'type': 'area', 'value': 'Kadayanallur, India'},  area2={'type': 'area', 'value': 'Kadayanallur, india'})
        self.assertEquals(predicted_result, ResultDataType.TRUE)