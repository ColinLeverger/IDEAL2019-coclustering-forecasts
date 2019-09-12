# "Toward a framework for seasonal time series forecasting using clustering" -- IDEAL 2019

This is the code associated to the paper "Toward a framework for seasonal time
series forecasting using clustering" published by C. Leverger et al. in
IDEAL 2019 Conference, Manchester.

## Quickstart

- Install Khiops on your machine,
- Install python requirements from ```requirements.txt```,
- Replace the string ```C:/ProgramData/Anaconda3/python.exe``` by your own
  python executable path on ```runit.sh``` file,
- Run ```./runit.sh``` in a shell,
- After execution, find and check results on ```res``` folder.

Note that without Khiops software installed on your machine it is pointless to
try to run this piece of software.

Khiops licenses and download available online for research purposes (3 months
trials), see https://khiops.diod.orange.com/.

## Code

Seeds provided directly on the code.

R scripts for AETSF provided in ```R``` folder.

## Datasets

- "om": Orange Money project hits per seconds,
- "om_cpus": Orange Money project CPUs per seconds,
- "aw": Australian walk,
- "melb_temp": temperature in Melbourne,
- "it": internet traffic,
- "rf_stjean": St Jean-related dataset (river flow and rainfall), datasets
  number 6 & 7 in the paper.
- "rf_niagara": river flow of Niagara,
- "hc": Hourly Consumption electricity, kaggle dataset.

In order to use with your own data, datasets must be recoded. Time series data
must be transformed to csv files which have 4 columns, and the header must
follow indicated norms and naming conventions. The first column, named "date\_",
is the date of the current season; this column must follow the format
"dd/mm/YYYY". The second column "val\_" is the values of the time series. Third
column "n\_day\_" is the identifier of the season (1 being the first season, 2
the second, etc.). Finally, the column "time\_" represents the index of the
value inside the season (1 being the first value of the considered season, 2 the
second, etc.) Moreover, Khiops, the software which is used for co-clustering,
requires a particular data format: the data frames must be alphanumerically
sorted following the column "n\_day\_". 

We first started the development only analysing days, thus the name of "n_day_"
column which is not generic. 

## External libraries

Crafting time series clustering for performances comparisons using https://github.com/rtavenar/tslearn / see the original paper "Tavenard, Romain. "tslearn: A machine learning toolkit dedicated to time-series data." (2017)."
