import os
import shutil
import re

# BASE_DIR = './'
# TASK_PREFIXES = ['pick_bread', 'pick_cup', 'pick_lid', 'pour_coke', 'sweep_trash']

# BASE_DIR = './pick_pen'
# TASK_PREFIXES = ['pick_pen']
# TARGET_DIR = './'

BASE_DIR = '/home/onestar/FastUMI_data' # Dir of your data
TASK_PREFIXES = ['unplug_charger'] # this is the task description of you data
TARGET_DIR = '../'
def natural_key(text):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', text)]

def merge_task_folders(base_dir, task_prefix):
    pattern = re.compile(f'^{task_prefix}_v\\d+$')
    target_dir = os.path.join(TARGET_DIR, f'merged_{task_prefix}')

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    episode_index = 0
    for subdir in sorted(os.listdir(base_dir), key=natural_key):
        if pattern.match(subdir):
            subdir_path = os.path.join(base_dir, subdir)
            for file in sorted(os.listdir(subdir_path), key=natural_key):
                if file.endswith('.hdf5'):
                    src_file = os.path.join(subdir_path, file)
                    dst_file = os.path.join(target_dir, f'episode_{episode_index}.hdf5')
                    shutil.copy2(src_file, dst_file)
                    episode_index += 1
            print(f"Merged: {subdir} -> {target_dir}")

    print(f"Completed merging {task_prefix}. Total episodes: {episode_index}")

def main():
    for task in TASK_PREFIXES:
        merge_task_folders(BASE_DIR, task)

if __name__ == "__main__":
    main()