## PRINGLE

**PRINGLE** ( PeRception INteGrity pipeLinE) is a reproducible research pipeline for **ontology-guided semantic validation of vision–language perception in autonomous driving**, focused on **traffic-light perception under physical manipulation and ambiguity**. 

PRINGLE is described in:

[article reference]

The pipeline combines (i) a CLIP-based traffic-light classifier and (ii) a lightweight ontology-driven semantic consistency checker used at inference time.

---

## Repository structure

- `test_set/` : 25-image benchmark used for attack vs. not-attacked evaluation (manifests/splits included)

- `scripts/`
  - `1-convert_to_jpg.py` : converts DTLD images to JPEG
  - `2-crop_from_jpgs.py` : crops traffic-light patches using DTLD annotations
  - `3-make_json_from_crops.py` : builds `crops_json/` metadata for training/evaluation
  - `4_1-train_vlm.py` : trains the **baseline** VLM (no ontology)
  - `4_2-train_vlm_onto.py` : trains the **ontology-driven** VLM (PRINGLE)
  - `5-test_models.py` : evaluates baseline vs PRINGLE on the 25-image test set

- `TrafficLightOntology.rdf` : Traffic Light Ontology used by PRINGLE
- `logs_training_vlm/`, `logs_training_vlm_onto/` : training logs (optional, for auditability)


**Not included**
- **DTLD dataset** (too large): download separately (link below)
- **Baseline dataset for testing**: download separately (link below)
- **Trained weights** (`best_model_*.pt`): not tracked due to size; available upon request
- **crops_jpg.zip, crops_json.zip**: optional prebuilt artifacts (otherwise regenerate with the scripts in `scripts/`); available upon request

---

## Data

### DTLD (required for training/validation)
Download DTLD from the official source and keep it outside the repo:

[https://www.uni-ulm.de/en/in/iui-driveu/driving-benchmark/](https://www.uni-ulm.de/en/in/institute-of-measurement-control-and-microtechnology/research/data-sets/driveu-traffic-light-dataset/)

### Test Dataset

The 25-image attack benchmark used for evaluation is derived from:

Bhupathiraju, Sri Hrushikesh Varma, Takami Sato, Michael Clifford, Takeshi Sugawara, Qi Alfred Chen, and Sara Rampazzi. 2025. *To Go or Not to Go: Shedding Light on Traffic Light Signal Manipulation and Defense Strategies*. ACM Transactions on Cyber-Physical Systems. Association for Computing Machinery, New York, NY, USA. https://doi.org/10.1145/3777451

Dataset access (OSF repository):
https://osf.io/uet7h/overview?view_only=52dd96073b484435bb09723381aefabb

