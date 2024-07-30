# Water level data

## pars.cov
this lists the parameter names identified by the ts_id. the ts_id is found in the table of water level measurements.

the ts_id's with Day.Mean edit values are from wells with pressure transducers. these should be sampled to monthly frequency

## stats.csv
station locations, well perforations, well use, and other info.

## wl.csv
the waterlevel data. join with the pars.csv


## quality codes:
| Key | Code      | Description                            |
|-----|-----------|----------------------------------------|
| 0   | Excellent | Quality control data (Excellent)       |
| 40  | Good      | Quality control data (Good)            |
| 80  | Fair      | Quality control data (Fair)            |
| 120 | Suspect   | Quality control data (Suspect)         |
| 160 | Poor      | Quality control data (Poor)            |
| 200 | Unknown   | Quality control data (Unchecked)       |
