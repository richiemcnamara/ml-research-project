# ml-research-project
**Final Project for Mathematical Principles of Machine Learning (MATH2805)**

This project investigates the impact of neural network (DNN) complexity on the construction of prediction intervals (PIs). Using bootstrapping and the pivot bootstrap method, the goal is to evaluate how different DNN architectures, trained on various data sets, can produce accurate and reliable PIs. The project also explores the relationship between the complexity of the neural network and the effectiveness of the resulting prediction intervals in terms of both width and coverage.

## Synthetic Data Experiments
This set of experiments uses synthetic data to investigate the effectiveness of bootstrapping-based prediction intervals for various DNN complexities. The experiments involve both linear and non-linear functions with two different types of error distributions: normal and uniform.

Key steps include:
- Generating synthetic data based on known functions.
- Training DNNs with varying complexity on the generated data.
- Constructing prediction intervals using the pivot bootstrap method.
- Evaluating the prediction intervals' width and coverage.

## Boston Housing Data Experiment
The Boston Housing dataset is used to evaluate how DNN complexity impacts the prediction intervals when applied to a real-world data set.

This experiment examines:
- How DNN complexity affects the width and coverage of prediction intervals.
- Comparison of results with synthetic data.

## Running the Code

1. Run `SynDataNormal.py`
2. Run `SynDataUniform.py`
3. Run `RealData.py`

## Viewing Results

1. Plots of synthetic data using normal error are located in `/normal_figures` and uniform error plots are located in `/uniform_figures`
2. Plots of the Boston Housing data are located in `BH_figures`

## Conclusions

My full research paper is located at Research_paper.pdf

Overall, the experiments demonstrated that as the complexity of the neural network increases, the better the prediction intervals (PIs) become. It is important to always consider PI width and coverage simultaneouly, because increasing complexity could decrease the width, but sacrifice coverage. Future work in this space could be optimizing the DNNs used because not all optimal hyperparameters were found due to time constraints of this project. One could also research different levels of confidence, as only 95% was considered in this project.
