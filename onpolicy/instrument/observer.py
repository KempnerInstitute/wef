# Legacy recorder for the rays/ environment and MAEFish.py standalone observe=True paths.
# The fish training/eval pipeline uses onpolicy/instrument/recorder.py (Recorder) instead.
import pandas as pd
class Observer:
    def __init__(self):
        # Initialize an empty DataFrame with the desired columns
        self.list = []
        self.data = None
        print("Initialized observer")

    def record(self, episode_index, time_step, data):
        # This method will be called to add data to the DataFrame for all agents
        temp_data = {"episode_index": episode_index, "time_step": time_step, **data}
        self.list.append(temp_data)

    def _list_to_dataframe(self):
        if self.data is None:   
            self.data = pd.DataFrame(self.list)
            self.list = []
        print("Converted list to DataFrame")

    def _add_metadata(self, all_args, env_args):
        agent_args, multi_agent_args, arena_args = env_args
        multi_agent_args['frames'] = None
        all_args = vars(all_args)
        data_dict = {
            "all_args": all_args,
            "agent_args": agent_args,
            "multi_agent_args": multi_agent_args,
            "arena_args": arena_args,
        }
        mask = (self.data['episode_index'] == 0) & (self.data['time_step'] == 0)
        if not self.data[mask].empty:
            self.data.loc[mask, 'metadata'] = [data_dict]
        else:
            print("No matching row found for episode_index = 0, time_step = 0")

    def save(self, filename="render_data.pkl", metadata=None):
        if self.data is None:
            self._list_to_dataframe()
        if metadata is not None:
            all_args, env_args = metadata
            self._add_metadata(all_args, env_args)
        self.data.to_pickle(filename)
        print(f"Saved: {filename}")
