# SpaceX Falcon 9: Data Collection and Wrangling

**Question:** How can launch records be turned into a clean dataset for predicting whether a Falcon 9 first stage lands?

**Data:** IBM's static snapshot of the SpaceX API response (107 launches) and IBM's published Falcon 9 dataset (90 launches). The live SpaceX API was down when this was built (Cloudflare errors 522/525), so its ID lookups could not be run here.

## Approach

- GET request and JSON parsing with `requests` and `pd.json_normalize`
- Falcon 9 filter, and mean imputation for `PayloadMass` (6,104.96 kg)
- Launch counts by site, orbit and outcome
- Binary target `Class`: 1 for a successful landing, 0 otherwise

## Results

| Measure | Value |
|---|---|
| Falcon 9 launches | 90 |
| Successful landings | **60 (66.7%)** |
| Missing `LandingPad` values | 26 (no pad was used) |

| Launch site | Launches | Landing success |
|---|---|---|
| CCAFS SLC 40 | 55 | 60.0% |
| KSC LC 39A | 22 | 77.3% |
| VAFB SLC 4E | 13 | 76.9% |

## Key findings

- 41 of the 60 successful landings were on a drone ship (`True ASDS`), 14 on a ground pad and 5 in the ocean.
- GTO is the most common orbit (27 launches) and has one of the lowest success rates (51.9%), while VLEO has 85.7% over 14 launches.
- Several orbits have only 1 launch, so their 0% or 100% rates mean little.
- These are descriptive rates: they don't account for payload, booster version or date. Later labs would test those with a model.

## Stack

requests, pandas

## Run it

Open [`analysis.ipynb`](analysis.ipynb). [`dataset_part_2.csv`](dataset_part_2.csv) is the output with the `Class` column.
