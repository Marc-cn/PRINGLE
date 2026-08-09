# data/

Place inputs here, then run `python process_data.py` from `clean/`.

| Path | Role |
|------|------|
| `dtld/` | Raw DTLD frames (required) |
| `labels/` | `DTLD_Labels_v2.0/v2.0` per-city JSON (required for crops) |
| `laser/Red`, `laser/Green`, `laser/IR_modulated` | Laser / IR references (required for synth) |
| `crops_jpg/` | Output of `process_data.py` |
| `synth_jpg/` | Optional synth-attack cache from `process_data.py` |

See `../README.md` for the full workflow.
