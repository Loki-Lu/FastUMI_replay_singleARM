# FastUMI_replay_singleARM
This is the demo for FastUMI data replay.

# How to download demo:
We use **unplug_charger** dataset as examples.

```bash
# need to login huggingface
hf auth login --token <YOUR_TOKEN>
#use hf-mirror
pip install -U huggingface_hub
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --repo-type dataset --resume-download IPEC-COMMUNITY/FastUMI-Data unplug_charger.tar.gz.part-001 --local-dir ~/fast_umi/
huggingface-cli download --repo-type dataset --resume-download IPEC-COMMUNITY/FastUMI-Data unplug_charger.tar.gz.part-002 --local-dir ~/fast_umi/
huggingface-cli download --repo-type dataset --resume-download IPEC-COMMUNITY/FastUMI-Data unplug_charger.tar.gz.part-003 --local-dir ~/fast_umi/
```

# Merging Data:
```bash
cd <your data>
cat unplug_charger.tar.gz.part-00* > unplug_charger.tar.gz
```

# Extract the Dataset:
```bash
tar -xzvf unplug_charger.tar.gz
```
# Data Process:
## Step 1:
Merge all episodes(unplug_charger_v0 - unplug_charger_v9)
```bash
python3 merge_episodes.py 
```
