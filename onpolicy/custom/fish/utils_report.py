import os
import glob
import pandas as pd
import numpy as np
import sys
import argparse
import matplotlib.pyplot as plt
import pickle


def get_latest_outputs_folder(base_path):
    if "outputs" in base_path and os.path.isdir(base_path):
        return base_path
    subdirs = [
        d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))
    ]
    latest_subdir = max(
        subdirs, key=lambda d: os.path.getctime(os.path.join(base_path, d))
    )
    outputs_folder = os.path.join(base_path, latest_subdir, "outputs/")
    return outputs_folder


def load_data(filename):
    try:
        return pd.read_pickle(filename)
    except Exception as e:
        print(f"Standard pickle loading failed: {e}")
        print("Attempting robust loading method...")
        return robust_load_pickle(filename)


def robust_load_pickle(filename):
    """
    A more robust pickle loader that can handle pandas version differences.
    This is useful when loading pickles created with a different pandas version.
    """
    try:
        # Method 1: Using pickle directly with encoding
        with open(filename, 'rb') as f:
            return pickle.load(f, encoding='latin1')
    except Exception as e:
        print(f"Method 1 failed: {e}")
        
        try:
            # Method 2: Using pandas with explicit encoding
            return pd.read_pickle(filename, compression=None)
        except Exception as e:
            print(f"Method 2 failed: {e}")
            
            try:
                # Method 3: Using a custom unpickler class
                class CustomUnpickler(pickle.Unpickler):
                    def find_class(self, module, name):
                        if module == 'pandas.core.internals.blocks' and name == 'new_block':
                            # Handle the specific missing attribute
                            from pandas.core.internals.blocks import make_block
                            return make_block
                        return super().find_class(module, name)

                with open(filename, 'rb') as f:
                    return CustomUnpickler(f).load()
            except Exception as e:
                print(f"All robust loading methods failed: {e}")
                raise ValueError(f"Unable to load pickle file {filename} due to version incompatibility. You may need to regenerate the file with your current pandas version.")


def print_column_shapes(data):
    for column in data.columns:
        try:
            # Attempt to get the shape of the first element if it's an array-like
            first_element_shape = np.shape(data[column].iloc[0])
        except Exception as e:
            # If it's not an array-like or if there's an error, print the type instead
            first_element_shape = type(data[column].iloc[0])
        print(f"{column}: {first_element_shape}")

def get_num_envs(data):
    for col in ["actions", "observations", "rnn_states", "rewards", "infos"]:
        first_element = data[col].iloc[0]
        if isinstance(first_element, np.ndarray):
            return first_element.shape[0]
        elif isinstance(first_element, list):
            return len(first_element)
    return 1  # Default to 1 if no valid columns are found


def recurse_type_info(element):
    print(type(element))
    if isinstance(element, np.ndarray):
        print(element.shape)
    elif isinstance(element, list):
        print(len(element))
        if len(element) > 0:
            recurse_type_info(element[0])


def get_pkl_file_containing_str(input_dir="./", pkl_str="", flattened=False):
    pkl_files = glob.glob(input_dir + f"/*{pkl_str}*.pkl")
    pkl_files = [f for f in pkl_files if "spis_by_context" not in f]  # Exclude
    if flattened:
        pkl_files = [f for f in pkl_files if "flat" in f]
    else:
        pkl_files = [f for f in pkl_files if "flat" not in f]  # Exclude
    if not pkl_files:
        raise FileNotFoundError(f"No .pkl files containing {pkl_str} found in the current directory.")
    pkl_file = pkl_files[0]
    return pkl_file


def get_all_raw_pkl_files_containing_str(input_dir="./", pkl_str="", flattened=False):
    pkl_files = glob.glob(input_dir + f"/*{pkl_str}*.pkl")
    pkl_files = [f for f in pkl_files if "spis_by_context" not in f]  # Exclude
    if flattened:
        pkl_files = [f for f in pkl_files if "flat" in f]
    else:
        pkl_files = [f for f in pkl_files if "flat" not in f]  # Exclude
    if not pkl_files:
        raise FileNotFoundError(f"No .pkl files containing {pkl_str} found in the current directory.")
    return pkl_files


def get_latest_pkl_file(input_dir="./"):
    pkl_files = glob.glob(input_dir + "/*.pkl")
    pkl_files = [f for f in pkl_files if "flat" not in f]  # Exclude
    pkl_files = [f for f in pkl_files if "spis_by_context" not in f]  # Exclude
    if not pkl_files:
        raise FileNotFoundError("No .pkl files found in the current directory.")
    latest_pkl_file = max(pkl_files, key=os.path.getctime)
    return latest_pkl_file


def get_latest_flat_pkl_file(input_dir="./"):
    pkl_files = glob.glob(input_dir + "/*.pkl")
    pkl_files = [f for f in pkl_files if "flat" in f]
    if not pkl_files:
        raise FileNotFoundError("No .pkl files found in the current directory.")
    latest_pkl_file = max(pkl_files, key=os.path.getctime)
    return latest_pkl_file


def get_latest_raw_pkl_files(input_dir="./", force=False):
    pkl_files = glob.glob(input_dir + "/*raw.pkl")
    pkl_files = [f for f in pkl_files if "flat" not in f and "spis_by_context" not in f]
    if not pkl_files:
        raise FileNotFoundError("No .pkl files found in the current directory.")

    latest_pkl_file = max(pkl_files, key=os.path.getctime)
    base_name = os.path.basename(latest_pkl_file)

    split_name = base_name[:-4].split("_")  # remove .pkl extension

    if split_name[-1] == "raw":
        pkl_str = split_name[-3]  # handle split pkls
    else:
        pkl_str = split_name[-1]  # handle non-split pkls

    flat_files = glob.glob(input_dir + f"/*{pkl_str}*flat*.pkl")
    if flat_files and not force:
        raise FileExistsError("Flattened pkl already generated for the latest pkl.")

    matched_files = glob.glob(input_dir + f"/*{pkl_str}*.pkl")
    if force:
        matched_files = [f for f in matched_files if "flat" not in f and "spis_by_context" not in f]  # Exclude

    return matched_files


def get_latest_behavior_bundle_stems(input_dir="./", force=False):
    """Return sorted unique stems for the new Recorder bundle format (*_behavior.pkl)."""
    behavior_files = sorted(glob.glob(os.path.join(input_dir, "*_behavior.pkl")))
    if not behavior_files:
        raise FileNotFoundError(f"No *_behavior.pkl files found in {input_dir}")

    # Derive output flat pkl name from the first file to check for existing output
    stems = [f[: -len("_behavior.pkl")] for f in behavior_files]
    common = _common_stem(stems)
    flat_file = os.path.join(input_dir, os.path.basename(common) + "_agg_flat.pkl")
    if os.path.exists(flat_file) and not force:
        raise FileExistsError(f"Flat pkl already exists: {flat_file}")

    return stems, flat_file


def _common_stem(stems):
    """Strip trailing _<episode_number> to get the run-level stem."""
    import re
    return re.sub(r"_\d+$", "", stems[0])


def get_expected_flattened_filename(pkl_files):
    # get path before basename
    dirpath = os.path.dirname(pkl_files[0])
    base_name = os.path.basename(pkl_files[0])
    split_name = base_name[:-4].split("_")  # remove .pkl extension

    if split_name[-1] == "raw":
        expected_flattened_filename = "_".join(split_name[:-2]) + "_agg_flattened.pkl"  # handle split pkls
    elif split_name[-1] == "behavior":
        expected_flattened_filename = "_".join(split_name[:-2]) + "_agg_flat.pkl"
    else:
        expected_flattened_filename = "_".join(split_name) + "_flattened.pkl"

    expected_flattened_filename = os.path.join(dirpath, expected_flattened_filename)
    return expected_flattened_filename


def get_df_from_pkls(pkl_files, drop_rnn=False, drop_attn_mask=False):
    data_all = []
    for i, pkl_file in enumerate(pkl_files):
        data = load_data(pkl_file)
        cols_to_drop = []
        if drop_rnn and "rnn_states" in data.columns:
            cols_to_drop.append("rnn_states")
        if drop_attn_mask and "attn_mask" in data.columns:
            cols_to_drop.append("attn_mask")
        if cols_to_drop:
            data = data.drop(columns=cols_to_drop)
        print(f"Loaded {pkl_file} ({i+1}/{len(pkl_files)}): {data.shape[0]} rows. Dropped columns: {cols_to_drop}")
        data_all.append(data)
    data = pd.concat(data_all)
    del data_all
    return data


def describe_dataframe(df):
    import io

    # General Information
    info_buf = io.StringIO()
    df.info(buf=info_buf)
    info_str = info_buf.getvalue()

    # First Few Rows
    head_str = df.head(n=1).to_string()

    # Summary Statistics
    describe_str = df.describe(include="all").to_string()

    # Data Types and Null Counts
    dtypes_str = df.dtypes.to_string()
    null_counts_str = df.isnull().sum().to_string()

    # Combine all parts
    description = f"""
    General Information:
    {info_str}
    
    First Few Rows:
    {head_str}
    
    Summary Statistics:
    {describe_str}
    
    Data Types:
    {dtypes_str}
    
    Null Counts:
    {null_counts_str}
    """

    return description


# Example usage
# description = describe_dataframe(data)
# print(description)



def remove_outliers(df, column):
    """
    Remove outliers from a specified column in a DataFrame using the IQR method.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The name of the column to remove outliers from.

    Returns:
    pd.DataFrame: The DataFrame with outliers removed from the specified column.
    """
    # Calculate the IQR for the specified column
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    # Define outlier boundaries
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter out the outliers
    df_no_outliers = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    return df_no_outliers


def get_sensor_bounds(dff):
    agent_args = dff['metadata'].iloc[0]['agent_args']

    return {
        "general_sensor_min": agent_args['general_sensor_min'],
        "mormyromast_sensor_min": agent_args['mormyromast_sensor_min'],
        "mormyromast_sensor_max": agent_args['mormyromast_sensor_max'],
        "ampullary_sensor_min": agent_args['ampullary_sensor_min'],
        "ampullary_sensor_max": agent_args['ampullary_sensor_max'],
        "knollen_sensor_min": agent_args['knollen_sensor_min'],
        "knollen_sensor_max": agent_args['knollen_sensor_max'],
        "knollen_binarize_threshold": agent_args['knollen_binarize_threshold'],
    }


def get_sensor_indices(dff):
    num_rays = dff['metadata'].iloc[0]['agent_args']['num_rays']
    num_knollen_sensors = dff['metadata'].iloc[0]['agent_args']['num_knollen_sensors']
    num_ampullary_sensors = dff['metadata'].iloc[0]['agent_args']['num_ampullary_sensors']
    num_agents = dff['metadata'].iloc[0]['multi_agent_args']['num_agents']

    mormyromast_indices = slice(0, num_rays)
    if 'knollen_metadata' in dff['metadata'].iloc[0]['agent_args']:
        ampullary_indices = slice(
            num_rays,
            num_rays + num_ampullary_sensors
        )
        knollen_indices = slice(
            num_rays + num_ampullary_sensors,
            num_rays + num_ampullary_sensors + num_knollen_sensors * (num_agents - 1)
        )
    else:
        knollen_indices = slice(
            num_rays,
            num_rays + num_knollen_sensors
        )
        ampullary_indices = slice(
            num_rays + num_knollen_sensors,
            num_rays + num_knollen_sensors + num_ampullary_sensors
        )
    return mormyromast_indices, ampullary_indices, knollen_indices

def load_flat_pkl_file(flat_pkl_file, task="foraging"):

    OUTPUTS_FOLDER = os.path.dirname(flat_pkl_file)
    # If not writeable, then save in current directory
    if not os.access(OUTPUTS_FOLDER, os.W_OK):
        print(
            f"Outputs folder {OUTPUTS_FOLDER} is not writable, using current directory instead."
        )
        OUTPUTS_FOLDER = "./"
    print(f"Using outputs folder: {OUTPUTS_FOLDER}")


    dff = pd.read_pickle(flat_pkl_file)
    # ru.print_column_shapes(dff)
    print("dff.shape", dff.shape)
    # print("dff.columns", dff.columns)

    pkl_str = flat_pkl_file.split("/")[-1].replace("MAFish_neural_", "").replace("_agg_flattened.pkl", "")
    print(f"pkl_str: {pkl_str}")

    # Task-specific filtering
    # If 2f1p task, only keep the 2f1p agents
    if task in ["2f1p", "2ff"]:
        dff = dff[dff["agent_id"].isin([0, 1])]
        print("dff.shape after filtering for 2f1p agents", dff.shape)
        print("dff.agent_id.unique()", dff.agent_id.unique())

    elif task == "homing":
        # Canonical homing preprocessing: extract agent 0 as target, keep agent 1,
        # trim to first trial per episode. This same logic is replicated in
        # analysis_homing.py for the new pipeline — keep the two in sync.
        # Extract target (agent_id == 0) location dataframe
        target_df = (
            dff.loc[dff["agent_id"] == 0, ["env_id", "episode_index", "time_step", "position_x", "position_y"]]
            .rename(columns={"position_x": "target_x", "position_y": "target_y"})
        )

        # Keep only the homing agent (agent_id == 1)
        dff = dff[dff["agent_id"] == 1]  # Only keep the homing agent data
        print("dff.shape after filtering for homing agent", dff.shape)
        print("dff.agent_id.unique()", dff.agent_id.unique())

        # HOMING SPECIFIC: Keep only first homing trial (before second "Trial-start")
        # For each (env_id, agent_id, episode_index), sort by time_step and only keep data before second "Trial-start"
        dff = dff.sort_values(
            by=["env_id", "agent_id", "episode_index", "time_step"]
        ).copy()
        def _keep_until_second_trial_start(group):
            trial_start_indices = group.index[group["trial_start"] == True].tolist()
            if len(trial_start_indices) < 2:
                return group  # Keep all if fewer than 2 trial_starts
            second_start_idx = trial_start_indices[1]
            return group.loc[
                : second_start_idx - 1
            ]  # Up to just before the second trial_start
        dff = dff.groupby(["env_id", "agent_id", "episode_index"], group_keys=False).apply(
            _keep_until_second_trial_start
        )
        print("dff.shape after keeping until second trial start", dff.shape)

         # Merge target locations back
        dff = pd.merge(
            dff,
            target_df,
            on=["env_id", "episode_index", "time_step"],
            how="left",
        )

        print("dff.shape after adding target locations", dff.shape)
        print("Added columns:", [c for c in dff.columns if "target" in c])


    return dff, OUTPUTS_FOLDER, pkl_str
