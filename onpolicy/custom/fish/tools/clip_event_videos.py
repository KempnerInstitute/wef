import os
import argparse
from pathlib import Path
import pandas as pd
import imageio

def extract_event_videos(pkl_path, vid_dir=None, event_col='bite_other_fish', window_size=5, n_events=10, output_dir=None):
    """
    Extracts short video clips around specified events from behavior GIFs.

    - Filters GIFs by config string.
    - Prefilters DataFrame to episodes and envs with events before opening any GIF.
    - Reads each GIF only once and processes its events sequentially.

    Clips are generated even if the window extends beyond the GIF start or end:
    - At the start, the clip begins at frame 0.
    - At the end, the clip stops at the last available frame.
    """
    pkl_path = Path(pkl_path)
    dff = pd.read_pickle(pkl_path)
    metadata = dff['metadata'].iloc[0]
    args = metadata['all_args']
    n_envs = args['n_eval_rollout_threads']
    episode_length = args['episode_length']
    max_vid_frames = args.get('max_vid_frames', n_envs * episode_length)

    # parse config string from filename (third-to-last underscore)
    parts = pkl_path.stem.split('_')
    config_str = parts[-3]
    print(f"Using config filter: '{config_str}'")

    # Determine other directories
    vid_dir = Path(vid_dir) if vid_dir else pkl_path.parent
    output_dir = Path(output_dir) if output_dir else pkl_path.parent

    # Find all behavior GIFs matching the same config directly
    config_str_underscore = config_str + '_'
    pattern = f'MAFish_behavior_*{config_str_underscore}*.gif'
    vid_files = sorted(vid_dir.glob(pattern))
    if not vid_files:
        raise FileNotFoundError(f"No behavior GIFs matching config '{config_str}' found with pattern '{pattern}' in {vid_dir}")
    print(f"Found {len(vid_files)} GIF(s) for config '{config_str}': {[f.name for f in vid_files]}")

    # prefilter events by episode and environment
    events = dff[dff[event_col]].groupby(['episode_index', 'env_id'])['time_step']
    epi_env_events = {}
    for (epi, env), times in events:
        epi_env_events.setdefault(epi, {})[env] = list(times.astype(int)[:n_events])

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def process_vid(epi, vid_path):
        vid_stem = vid_path.stem
        stem_parts = vid_stem.split('_')
        # drop last part (episode index)
        vid_prefix = '_'.join(stem_parts[:-1])
    
        reader = imageio.get_reader(str(vid_path))
        frames = list(reader)
        reader.close()
        total_frames = len(frames)
        if total_frames == 0:
            return

        for env, time_steps in epi_env_events.get(epi, {}).items():
            for idx, t in enumerate(time_steps):
                frame_idx = env * episode_length + t
                if frame_idx < 0 or frame_idx >= total_frames:
                    continue
                start = max(0, frame_idx - window_size)
                end = min(total_frames - 1, frame_idx + window_size)
                clip = frames[start:end+1]
                out_name = (
                    f"{vid_prefix}_evt_{event_col}_w{window_size}_epi{epi}_env{env}_idx{idx}.mp4"
                )
                out_path = out_dir / out_name
                imageio.mimsave(str(out_path), clip, fps=metadata.get('fps', 10))
                print(f"Saved clip: {out_path} (frames {start}-{end})")

    # sequentially process each matching GIF
    for vid_path in vid_files:
        epi_str = vid_path.stem.split('_')[-1]
        if epi_str.isdigit():
            epi = int(epi_str)
            if epi in epi_env_events:
                process_vid(epi, vid_path)


def main():
    parser = argparse.ArgumentParser(
        description="Extract event-centered video clips from behavior GIFs."
    )
    parser.add_argument('--pkl', required=True, help='Path to flattened pickle')
    parser.add_argument('--vid_dir', help='Directory with behavior GIFs (default: same as pkl)')
    parser.add_argument('--event_col', default='bite_other_fish', help='Event column name')
    parser.add_argument('--window', type=int, default=5, help='Frames before/after event')
    parser.add_argument('--n_events', type=int, default=10, help='Max events per episode')
    parser.add_argument('--out_dir', help='Output directory for clips (default: same as pkl)')
    args = parser.parse_args()

    extract_event_videos(
        args.pkl,
        vid_dir=args.vid_dir,
        event_col=args.event_col,
        window_size=args.window,
        n_events=args.n_events,
        output_dir=args.out_dir
    )

if __name__ == '__main__':
    main()
