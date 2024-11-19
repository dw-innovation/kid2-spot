import unittest
import pandas as pd
from typing import List
from benchmarking.evaluate_results import compare_yaml, is_parsable_yaml, AreaAnalyzer, EntityAnalyzer

'''Run python -m unittest datageneration.tests.benchmarking.test_evaluate_results'''

#     yaml_true_string = """area:
#   - type: area
#     value: Dâmbovița County, Romania
# entities:
#   - id: 0
#     name: social facility
#     type: nwr
#     properties:
#       - name: building levels
#         operator: <
#         value: 3
#       - name: name
#         operator: ~
#         value: ole
#   - id: 1
#     name: fabric shop
#     type: nwr
#   - id: 2
#     name: petrol station
#     type: nwr
# relations:
#   - type: distance
#     source: 1
#     target: 0
#     value: 400 m
#   - type: distance
#     source: 2
#     target: 1
#     value: 300 m"""
#
#     yaml_pred_string = """area:
#   - type: area
#     value: Dâmbovița County, Romania
# entities:
#   - id: 0
#     name: social facility
#     type: nwr
#     properties:
#       - name: building levels
#         operator: <
#         value: 3
#       - name: name
#         operator: ~
#         value: ole
#   - id: 1
#     name: petrol station
#     type: nwr
#   - id: 2
#     name: fabric shop
#     type: nwr
# relations:
#   - type: distance
#     source: 2
#     target: 0
#     value: 400 m
#   - type: distance
#     source: 1
#     target: 2
#     value: 300 m"""
class TestEvaluation(unittest.TestCase):
    def setUp(self):
        pass


    def test_parser(self):
        yaml_true_string = """area:
  Type: area
  value: Kadayanallur, India
entities:
  - id: 0
    type: nwr
    name: chess game
  - id: 1
    type: nwr
    name: bistro
    properties:
      - key: outdoor_seating
        operator: '='
        value: yes
  - id: 2
    type: nwr
    name: park
relations:
  - type: distance
    source: 0
    target: 1
    value: 10 m
  - type: contains
    source: 1
    target: 2
"""
        _, data1 = is_parsable_yaml(yaml_true_string)
        assert data1

    def test_pipeline(self):
        yaml_true_string = """area:
  - type: bbox
entities:
  - id: 0
    type: nwr
    name: vacant shop
    properties:
      - name: floors
        operator: <
        value: 10
  - id: 1
    type: nwr
    name: office building
  - id: 2
    type: nwr
    name: gambling den
relations:
  - type: contains
    source: 1
    target: 0 
  - type: distance
    source: 0
    target: 2
    value: 0.5 miles
"""

        yaml_pred_string = """area:
  - type: bbox
entities:
  - id: 0
    type: nwr
    name: gambling den
  - id: 1
    type: nwr
    name: office building
  - id: 2
    type: nwr
    name: vacant shop
    properties:
      - name: floors
        operator: <
        value: 10
relations:
  - type: distance
    source: 0
    target: 2
    value: 0.5 miles
  - type: contains
    source: 1
    target: 2
"""
        area_analyzer = AreaAnalyzer()
        entity_analyzer = EntityAnalyzer()
        result = compare_yaml(area_analyzer, entity_analyzer, yaml_true_string, yaml_pred_string)
        print("The YAML structures are the same:", result)