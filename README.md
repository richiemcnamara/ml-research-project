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

Overall, the experiments demonstrated that as the complexity of the neural network increases, the width of the prediction intervals (PIs) also tends to increase, especially in cases involving non-linear synthetic data. In the synthetic experiments, PIs for the non-linear functions were wider compared to the linear functions, with normal error distributions producing more consistent results than uniform errors. For the Boston Housing data, DNN complexity showed similar trends, although the prediction intervals were more stable and narrower compared to the synthetic data experiments. The pivot bootstrap method was effective in constructing PIs with adequate coverage, though increasing model complexity led to less reliable PIs in some cases, highlighting a trade-off between model complexity and the reliability of PIs.
