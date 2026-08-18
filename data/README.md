# Data

Raw workbooks are stored directly in this directory. `scripts/prepare_data.py`
resolves them by default and writes derived, auditable files to
`data/processed/`.

The source workbooks are never modified. Athlete identity columns are retained
only in audit data and Olympic progression joins; they are removed before model
tensors are created.
