# NeonatalSleepData

Code for processing, validating and analysing the NeonatalSleepData dataset:
200 term neonates and 45,879 labelled 30-second epochs. Download the data
from the Figshare.

## Contents

| File | Purpose |
| --- | --- |
| `prepare_data.py` | EDF processing, label alignment, EEG epochs and aEEG envelopes |
| `validate_data.py` | NPZ schemas, paired labels, stage counts, amplitudes and checksums |
| `reproduce_statistics.py` | 86 aEEG features, participant-stage summaries and descriptive plots |
| `run_baselines.py` | Random Forest and CNN evaluation |
| `src/neonatal_sleep/` | Implementation modules |
| `config.json` | Processing and model settings |
| `splits.json` | Five participant-level outer folds and four inner folds per outer fold |

## Installation

Use Python 3.11 or 3.12. From this directory:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Use `python3.12` instead if appropriate. On Windows, activate the environment
with `.venv\Scripts\activate`.

## Data

Extract the Figshare package and pass its directory to `--data-root`:

```text
NeonatalSleepData/
├── data/
│   ├── source_edf/NSDxxxx.edf
│   ├── eeg_100hz/NSDxxxx.npz
│   └── aeeg_epoch/NSDxxxx.npz
└── metadata/
    ├── labels.csv
    └── participants.csv
```

EEG arrays are `[N, 8, 3000]` in volts; aEEG arrays are `[N, 2, 30]` in
microvolts. Labels are `0=W`, `1=N`, `2=R`, `3=T`. See the dataset README
for complete field definitions.

## Usage

Replace `/path/to/NeonatalSleepData` with the extracted dataset directory.

```bash
python validate_data.py --data-root "/path/to/NeonatalSleepData"
python reproduce_statistics.py --data-root "/path/to/NeonatalSleepData"
python run_baselines.py --data-root "/path/to/NeonatalSleepData" --model random_forest
python run_baselines.py --data-root "/path/to/NeonatalSleepData" --model cnn
```

To regenerate aEEG from the distributed EEG:

```bash
python prepare_data.py --data-root "/path/to/NeonatalSleepData" --mode from-existing
```

EDF processing modes are `--mode from-edf` and `--mode sanitize-edf`.
Use `--labels-csv` to supply a public label table and `--subject NSD0001`
to process one participant. Run any entry point with `--help` for options.

Results go to `outputs/`. Prepared pairs go to `outputs/prepared/`; later
workflows read the original configured input directories until explicitly
changed. `--config` selects another configuration. Relative output and split
paths are resolved beside that file; `--data-root` resolves relative data
paths. Existing output files may be overwritten on subsequent runs.

## Reproducibility

The distributed CSV's consecutive `epoch_start_seconds` values are retained-
epoch coordinates, not verified original EDF onsets. EDF reconstruction uses
annotation-derived candidates when available; repeated labels can make
ordered matching ambiguous. Without annotations, it falls back to CSV starts
or consecutive windows, which require independently verified timing.

The inherited processing, feature and baseline algorithms and participant
splits are retained. Some processing settings are fixed in the implementation;
JSON edits alone do not change all sampling rates, filter settings or window
lengths. `sanitize-edf` includes filtering and resampling. Validation checks
NPZ files and paired labels rather than original EDF timing or all metadata.
The statistics workflow generates feature tables and two descriptive figures.
Full equivalence to historical EEG files and all manuscript results has not
been established.

## Tests

```bash
python -m pytest -q
```

Tests use temporary synthetic signals and require no dataset download.

## Licence

The MIT licence applies to the software. Data access and reuse are governed
by the Figshare dataset's terms.
