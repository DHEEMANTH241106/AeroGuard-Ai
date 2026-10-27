# Dataset placement

Download NASA's C-MAPSS Turbofan Engine Degradation Simulation dataset from the
[NASA Prognostics Center of Excellence repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)
or the [NASA data portal](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data).

Place the unmodified files in `data/raw/`, for example:

```text
data/raw/train_FD001.txt
data/raw/test_FD001.txt
data/raw/RUL_FD001.txt
```

The app recognizes the equivalent FD002, FD003, and FD004 filenames. The dataset
is not bundled or downloaded automatically. Do not substitute fabricated training
records for unavailable NASA data. Compatible CSV or TXT files can instead be
uploaded in the Dataset Explorer.
