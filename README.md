## PRINGLE

**PRINGLE** ( PeRception INteGrity pipeLinE) is a reproducible research pipeline for **ontology-guided semantic validation of vision–language perception in autonomous driving**, focused on **traffic-light perception under physical manipulation and ambiguity**. 

PRINGLE is described in:

[article reference]

---

Not included in this repository is the DTLD dataset; you can download it [here](https://www.uni-ulm.de/en/in/institute-of-measurement-control-and-microtechnology/research/data-sets/driveu-traffic-light-dataset/).

## Layout

```
.
  data/
    dtld/           # YOU provide: DTLD city data
    labels/         # YOU provide: DTLD_Labels_v2.0/v2.0 JSON (Berlin.json, …)
    laser/          # provided: Red/ Green/ IR_modulated/ reference traces
    crops_jpg/      # created by scripts/process_data.py
    synth_jpg/      # optional; created by scripts/process_data.py
  checkpoints/      # place or train *.pt checkpoints here
  output/           # laser ASR + cardinality-consistency results
  scripts/          # Python train / eval / data pipeline
```



## 1. Create Environment

```bash
conda env create -f environment.yml
conda activate traffic-vlm
```



## 2. Prepare `data/`

Copy (or symlink) inputs into place:


| Path                                          | Contents                                          |
| --------------------------------------------- | ------------------------------------------------- |
| `data/dtld/`                                  | DTLD image tree (city folders / `*_k0.tiff`, …)   |
| `data/labels/`                                | Per-city label JSON from `DTLD_Labels_v2.0/v2.0/` |
| `data/laser/Red`, `…/Green`, `…/IR_modulated` | Reference laser / IR frames                       |




## 3. Build crops (+ optional synth attacks)

From this directory:

```bash
# Crops only (needed for training / benign eval)
python scripts/process_data.py --crops_only

# Crops + synthetic laser-attack cache
python scripts/process_data.py

# Synth cache only (reuses existing crops / DTLD)
python scripts/process_data.py --synth_only
```

Defaults:

- `--dtld_root` -> `data/dtld`
- `--labels_dir` -> `data/labels`
- `--laser_root` -> `data/laser`
- `--out_root` -> `data` -> writes `data/crops_jpg` and `data/synth_jpg`



## 4. Train

Pretrained models are available upon request.

```bash
python scripts/train_models.py \
  --mode baseline \
  --image_dir data/crops_jpg \
  --labels_dir data/labels \
  --save_dir checkpoints/baseline

python scripts/train_models.py \
  --mode card_owned \
  --image_dir data/crops_jpg \
  --labels_dir data/labels \
  --save_dir checkpoints/card_owned
```

## 5. Evaluate

From this directory, after checkpoints exist:

```bash
bash run_eval.sh
```

Reads from `data/`, writes under `output/`. Default caps are 1000 attack images per combo and 1000 benign images per state. Full eval:

```bash
MAX_PER_ATTACK_COMBO=0 MAX_PER_STATE=0 bash run_eval.sh
```

Optional: `BATCH_SIZE`, `CONDA_ENV`.

The results can be visualized with `visualize_results.ipynb`

## Notes

- Override any path with the corresponding CLI flag or env var if your tree differs.
- Shared path constants live in `scripts/helper.py` (`DATA_ROOT`, `DEFAULT_*`).

