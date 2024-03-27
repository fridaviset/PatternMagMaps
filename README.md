# PatternMagMaps
Pattern detection kernel for extrapolating online magnetic field mapping

## Getting started via Docker
A Dockerfile is provided which should contain everything necessary to run stuff.
E.g. build by
```
docker build -t periodic-magmaps .
```
Run container
```
docker-compose up
```
This starts up a Jupyter lab session for you that you can open in your favourite browser.
You can also run the Python scripts in this environment.

## Getting started via pipenv
If you don't want to use Docker, pipenv is quite convenient, make sure to have it installed before proceeding.
##### Install Python 3.11 
```
python3 -m pipenv --python 3.11
```
##### Install requirements
```
python3 -m pipenv install -r requirements.txt -e . --skip-lock
```
##### Run environment
```
python3 -m pipenv shell
```

### Reproducing results from "Online discovery of global patterns and local variations in magnetic fields using Gaussian process regression"
The primary files for reproducing the results in the paper are: `hallway.ipynb`, `visionen.ipynb`, and `basement.ipynb`. These should be straightforward to run and instructions are contained within.

Other than that, `basement.py` provides a simple way of trying out different model orders for the basement dataset. The saved model files produced by that script can then be used in the `basement.ipynb`.
All of the plots are produced in the notebooks.


##### Notes on basement dataset
The hyperparameter optimization may consume a lot of time -- an optimized model file is available upon request.