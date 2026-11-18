# Earthquake catalogue of Northeast India, 2013-2025 (raw and Mwg-homogenised)

Data and code accompanying:

> Pallavi P., Joshi S., Chauhan V., Saurabh P. *A Permissioned-Blockchain Framework for Verifiable, Reproducible Earthquake Catalogues: Design, Integrity Guarantees, and Application to Seismic Hazard in Northeast India.* (submitted)

## Contents

| Path | Description |
|---|---|
| `data/ComCat_NEIndia_2013-2025_raw.csv` | All 1,322 events of the ANSS Comprehensive Earthquake Catalog (ComCat) of the U.S. Geological Survey for 2013-2025, 20-30 N, 87-98 E, M >= 2.5, exactly as returned by the query below |
| `data/NEIndia_2013-2025_Mwg_homogenised.csv` | The same events with the homogenised magnitude Mwg, the conversion applied, the calibration-range status and the declustering result |
| `code/catalogue_analysis.py` | Magnitude homogenisation, declustering, completeness, Gutenberg-Richter parameters and all sensitivity analyses (Sections 3.8-3.10, 4.2-4.3 and 4.7-4.9 of the paper) |
| `code/montecarlo.py` | Monte Carlo experiments comparing the orthogonal fit, the GOR1 relation and SLR (Sections 3.7 and 4.4-4.6) |

## Source query

```
https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv&starttime=2013-01-01&endtime=2026-01-01&minlatitude=20&maxlatitude=30&minlongitude=87&maxlongitude=98&minmagnitude=2.5&orderby=time-asc
```

First retrieved on 15 March 2026 and repeated on 11 November 2026; both retrievals return the same 1,322 events.
Please cite the source catalogue as: U.S. Geological Survey (2017) Advanced National Seismic System (ANSS) Comprehensive Catalog of Earthquake Events and Products. https://doi.org/10.5066/F7MS3QZH

## Columns of the homogenised catalogue

| Column | Meaning |
|---|---|
| `id`, `time`, `latitude`, `longitude`, `depth`, `mag`, `magType` | As reported by ComCat |
| `Mwg` | Seismic moment magnitude (Das et al. 2019) |
| `conversion` | `Eq.18 (mb)`: Mwg = 1.21 mb - 1.391; `Eq.19 (ML)`: Mwg = 1.31 ML - 1.890 (Pallavi et al. 2023); `Das2019 via M0 (Mw)`: log10 M0 = 1.5 Mw + 16.05 (dyn cm), Mwg = log10 M0 / 1.36 - 12.68 |
| `calibration_range` | `inside` or `below` the calibration range of the relation used (mb 4.8-6.1, ML 5.0-6.6); `not applicable` for moment magnitudes |
| `cluster_id`, `mainshock` | Result of Uhrhammer-window declustering (r = exp(-1.022 + 0.804 M) km, t = exp(2.264 + 0.285 M) days) |

## Reproducing the results

Requires Python 3.9 or later with `numpy`, `pandas` and `matplotlib`.

```
pip install numpy pandas matplotlib
python code/catalogue_analysis.py data/ComCat_NEIndia_2013-2025_raw.csv results/
python code/montecarlo.py results/
```

`catalogue_analysis.py` writes `results.json` (Tables 6 and 7 of the paper), `fmd.png` (Fig. 8) and the homogenised catalogue.
`montecarlo.py` writes `montecarlo_results.json` (Tables 4 and 5) and `bprop.png` (Fig. 7). Random seeds are fixed, so the published numbers are reproduced exactly.

## Licence

Code: MIT licence. Data derived from ComCat: CC BY 4.0, with attribution to the U.S. Geological Survey as above.
