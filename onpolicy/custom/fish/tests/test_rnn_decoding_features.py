import os
import sys

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import analysis_rnn_decoding as decoding


def test_prepare_features_applies_exclude_from_decoding():
    dff = pd.DataFrame({
        "position_x": [0.0, 1.0],
        "position_y": [2.0, 3.0],
        "distance_to_nearest_agent": [4.0, 5.0],
        "distance_to_closest_food": [6.0, 7.0],
        "angle_to_closest_agent": [0.1, 0.2],
        "angle_to_closest_food": [0.3, 0.4],
        "displacement": [0.5, 0.6],
        "move_forward": [0.7, 0.8],
        "actual_turn": [0.9, 1.0],
        "agent_size": [1.1, 1.2],
        "energy": [1.3, 1.4],
        "has_nearby": [True, False],
        "emit_eod": [False, True],
        "eating_event": [True, False],
        "was_bitten": [False, True],
        "center_field.mormyromast": [np.array([1.0, 2.0]), np.array([3.0, 4.0])],
        "center_field.ampullary": [np.array([2.0, 1.0]), np.array([4.0, 3.0])],
        "center_field.knollen": [np.array([1.0, 0.0]), np.array([0.0, 1.0])],
        "orientation": [0.0, 0.5],
    })

    _, features = decoding.prepare_features(dff)

    assert "move_forward" not in features
    assert "emit_eod" not in features
    assert "eating_event" not in features
    assert "displacement" not in features
    assert "mormyromast_field_mag" not in features
    assert "ampullary_field_mag" not in features
    assert "position_x" in features
    assert "knollen_error_angle" in features
