import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sample_skewed_uniform(alpha, low, high, rng=None):
    u = rng.uniform(0, 1)
    return low + (high - low) * u**alpha


def flatten_list_of_lists(column):
    """Flatten a column that is a list of lists"""
    return column.apply(
        lambda x: (
            [item for sublist in x for item in sublist] if isinstance(x, list) else x
        )
    )


def cast_list_to_np_array(column):
    """Cast a column that is a list to a numpy array"""
    return column.apply(lambda x: np.array(x) if isinstance(x, list) else x)


def unlist_single_element_lists(column):
    """Un-list columns that contain single-element lists"""
    return column.apply(lambda x: x[0] if isinstance(x, list) and len(x) == 1 else x)


def grouped_train_test_split(df, group_cols, test_size=0.2, random_state=42):
    """
    Splits a DataFrame into train and test sets, ensuring no leakage across specified grouping columns.
    """
    # Create a unique group identifier
    df = df.copy()
    df["group_id"] = df[group_cols].apply(tuple, axis=1)

    # Get unique groups
    unique_groups = df["group_id"].unique()

    # Split groups into train and test
    train_groups, test_groups = train_test_split(
        unique_groups,
        test_size=test_size,
        random_state=random_state,
    )

    # Create train and test DataFrames
    train_df = df[df["group_id"].isin(train_groups)].drop(columns=["group_id"])
    test_df = df[df["group_id"].isin(test_groups)].drop(columns=["group_id"])

    return train_df, test_df

def polar_to_cartesian(r, theta):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y


def relative_to_global(position_polar, reference_position, reference_orientation):
    distance, angle = position_polar
    if distance < 0:  # Skip dummy observations
        return None
    x, y = polar_to_cartesian(distance, angle + reference_orientation)
    return reference_position + np.array([x, y])


def transform_to_relative(position, reference_position, reference_orientation):
    relative_position = position - reference_position
    rotation_matrix = np.array(
        [
            [np.cos(reference_orientation), -np.sin(reference_orientation)],
            [np.sin(reference_orientation), np.cos(reference_orientation)],
        ]
    )
    return np.dot(relative_position, rotation_matrix.T)


def cartesian_to_polar(xy):
    x, y = xy
    distance = np.sqrt(x**2 + y**2)
    angle = np.arctan2(y, x)
    return distance, angle
