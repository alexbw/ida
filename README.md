# IDA

This code accompanies the publication from the Whitesides lab to Lab on a Chip, concerning automated analysis of red blood cell health using affordable field tests. Specifically, this code implements the automated extraction of scanned AMPS tubes from flatbed scanner images, the distillation of those images into 1D data traces, and then the automated identification of the anemic disease state of the blood in those 1D data traces, as well as the prediction of continuously-varying red blood cell (RBC) parameters.

## Installation

This code requires Python, as well as some extra 3rd-party libraries. All routines have only been tested on Mac OS, but should work on Linux. No guarantees for Windows.
1) First, we recommend using the "Anaconda" Python distribution, specifically the Python 2.7 version. [Download and install here](https://www.continuum.io/downloads).
1) With anaconda installed, you will need one extra library, to read TIFF files.
```
pip install imageio
```
1) You should be able to run the notebooks now. Inside this code repository, start up an IPython notebook:
```
jupyter notebook
```
1) You should now be able to browser around the "extraction" and "analysis" folders, which contain the relevant code.

## Running extraction

The first step required will be to prepare the raw data of TIFF file scans of 4x3=12 tubes from a flatbed scanner. We will also have to associate metadata for each patient that the blood in a tube was drawn from.
Our end goal will be a 1D array for each tube, along with the corresponding patient data (anemic state & RBC parameters).

The extraction algorithm is explained step-by-step in the notebook [extraction/Explaining the Extraction Algorithm.ipynb](extraction/Explaining the Extraction Algorithm.ipynb).
The implementation we use in the paper (which also fuses patient metadata with AMPS image data) can be found in [extraction/Data Extraction Pipeline.ipynb](extraction/Data Extraction Pipeline.ipynb). Most of the code in this notebook is specific to the particular Excel file format that we used to record patient metadata, and may have to be largely rewritten for reuse.

## Running analysis

There are two sets of analyses done in the paper: classification and regression.

### Running classification analysis

The goal of this analysis is to predict disease state only from the 1D data trace extracted from each AMPS tube. We use logistic regression, a linear classifier, and transform the 1D data representation using PCA to remove redundancy and constrain the input dimensionality of the problem. We also used Bayesian Optimization (bayesopt) to tune the hyperparameters of the problem, including the specific output dimension of PCA, the regularization parameter of logistic regression, and some image preprocessing parameters. Unfortunately, the service we used for bayesopt is no longer available. If you would like to automatically tune these parameters, we recommend either using random search (a surprisingly effective hueristic), or the open source library "Spearmint", upon which the now-defunct service we used was based, or products from the company SigOpt, which also implements bayesopt. The file we used to automatically tune these hypers is [classify.py](prediction/classify.py).
The best set of hyperparameters is stored in the notebook [Analyzing best classification.ipynb](prediction/Analyzing best classification).
This notebook examines ROC performance of the classifier for discriminating different anemia types, as well as the effect of centrifugation time of the tube on IDA classification AUC performance.


### Running regression analysis.

Similar to above, we used a defunct bayesopt service to automatically tune some parameters of this algorithm. The original file is in [regress.py](prediction/regress.py). The original methodology will work well, even without automated tuning.
The best set of hyperparameters is stored in the notebook [Analyzing best regression.ipynb](prediction/Analyzing best regression).



## License

See the LICENSE file. It is under a GPL license, and this code may only be used for academic purposes.
