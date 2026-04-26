# GreenRouting
This repository contains the code to reproduce the results of the thesis "Multi-Model Routing for Energy-Efficient LLM Code Generation". The code was written in Python 3.11.8 and designed for Linux devices with an NVIDIA GPU and AMD CPU using docker.


## Framework
The results are generated in four steps. The first step is to measure the data and obtain JSON and CSV files. We then load the files and aggragate the results. Afterwards, we run the code evaluation process and the last step is to evaluate the router.

### Measurements
To run the measurements run:
```
measure.sh
```

Alternatively download our dataset directly from google drive and place them in the results folder.

### Aggregation
To aggregate the results run
```
aggregate.py [results sub-folder Name 1] [results sub-folder Name 2]...
```
This creates an aggregated_data.json file in the results folder that is loaded in during our LLM evaluation step and generates the measurement related plots.

### Code Evaluation
!!!This script runs unsupervised AI generated code, recommended to run in a sandbox environment!!!
```
evaluateLLM.py
```
This creates labels for our dataset and is saved to results/aggregated_data_labaled.json

### Router Evaluation
Finally the routers can be trained and evaluated by running. This can take up to a few hours.
```
evaluateRouter.py --reproduce
```
Alternatively the plots can be regenerated from result checkpoints files located in the /Data folder.
```
evaluateRouter.py --plot
```
