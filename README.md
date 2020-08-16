# Machine Learning for Higher-Order Mode Analysis

This repository contains the code that was used for the Lee Teng Internship Project involving the usage of Deep Learning to create surrogate models to analyze HOM's using APS Beam Data (Final Project Title: "Implementation of Machine Learning on Advanced Photon Source Beam Data for Analysis of Higher-Order Modes").
## Installation

This code was written in [python ver 3.6.4](https://www.python.org/downloads/release/python-364/) and is stable in that version. Using other versions of python or the packages required for this code to run (listed in the `requirements.txt` file), might cause compatibility issues.

It is recommended to set up a virtual environment using a environment-management utility like conda. [Miniconda](https://docs.conda.io/en/latest/miniconda.html) is a light version version of conda - that installs the required version of python with some basic packages. The following creates the required virtual environment after conda is installed (in the Anaconda Prompt).  

```bash
conda create --name <environment_name> python=3.6.4
```
Here are some useful [Conda commands](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf).

Install the required packages using the `requirements.txt` file as follows:

```bash
pip install -r requirements.txt
```
## Usage
Using the Anaconda Prompt command line - Navigate to the local path of this directory and activate your virtual environment.

User Input is facilitated by an Input file and parser. Alter the values in the `Input.txt` file according to your specific use case. There are also several pre-made example Input files in the `Input Examples` Folder. Either drag them to the root directory and rename them to Input.txt or change the Input File being parsed in `main.py` to use them accordingly.

Then run the following to execute the program:
```bash
python main.py
```
## Input File Explanation
The Input File has several selections and they need to make sense depending on the use case and use objective. There are several pre-made example Input files in the `Input Examples` Folder. Either drag them to the root directory and rename them to Input.txt  or change the Input File being parsed in `main.py` to use them accordingly.

Here are explanations and examples for some selections in the input file. **If any selection is not relevant to the current Use Case or Objective then its value is ignored.**
### File Naming Conventions for proper parsing
1. **Files containing the data**\
Data files need to be **.csv files** and can have multiple sets of data in one file. They need to be named as follows:
```bash
<mode number>_<number of sets>sets_<temperature>.csv or <mode number>_<number of sets>sets_<sector>_<temperature>.csv
Example:
1_8sets_S40_89F.csv
1_6sets_92F.csv
```
2. **Growth rate files**\
The file containing growth rates associated with a mode need to be **.mat files** and should be named as: `gr.mat`.

3. **Averaged Data/Clean Data**\
The averaged datasets should also be **.csv files**. Each mode might have multiple raw data scans for training but their output should be the same "clean/averaged" file.

	Also if bunches are being skipped in the raw scans, then the **averaged file should be preprocessed to skip the bunches. Unlike the raw scans (see bunches_to_skip parameter below) there is no functionality to skip bunches for averaged data.**

	These averaged files should be named as follows:
```bash
<associated mode number>.csv
Example:
  1.csv - will contain the average of all 14 sets in 1_8sets_S40_89F.csv and 1_6sets_92F.csv
, such that empty bunches are skipped.
```
4. **Predictions**\
Predictions for the denoising autoencoder are saved as **.csv files**, named as follows:
```bash
<associated mode number>_<set used>.csv
Example:
1_2.csv
```
Predictions for the growth rates are saved as a **.pickle file**, containing a python dictionary with key-pair values being **"Mode: Estimated growth rate"**. Read about pickling and unpickling dictionaries [here](https://stackoverflow.com/questions/11218477/how-can-i-use-pickle-to-save-a-dict).

### UseCase
Choose the type of model that needs to be used from:
- Growth Rate Identification (0)
- Data Denoising (1)

### UseObjective
Choose the objective of the current run, whether it is to:
- Train the model on new data and save it for future use (0)
- Train a saved model on data of similar shape and type (1)
- Predict the output using a saved model (2)
- Simply print details about the data (3)

Option 3 will print a Dataframe showing how many datasets are available for a particular excited mode - and then exit the program.

### Various Path Initializations
In this section paths to the relevant directories are initialized. **If any of these selections are irrelevant** to the current use case or objective **they will be ignored** - so either initialize them to None or an irrelevant directory.
1. **datasets** (dtype: list)
A list of strings containing the paths to the directories that contain the datasets. Read the Naming Conventions section (part 1) for details on how the data files should be named.
```python
The directory names should not end with a "/" for this selection.
Example: datasets = ["root/Directory1", "root/Directory2"]
```
2. **path_to_gr** (dtype: string)
A string containing the filepath to the .mat file containing growth rates - `gr.mat`. Refer to Naming Conventions section (part 2) for file naming.

3. **path_to_avg** (dtype: string)
A string containing the path to the directory containing the averaged/clean datasets. Refer to Naming Conventions section for file naming  (part 3).
```python
The directory names should end with a "/" for this selection.
Example: path_to_avg = "root/Averages/"
```
4. **predsavepath** (dtype: string)
A string containing the path to the directory where predictions will be saved. **Make sure that this directory exists**. Refer to Naming Conventions section for file naming.
```python
The directory names should end with a "/" for this selection.
Example: path_to_avg = "root/Results/"
```

### Data Related Selections
1. **num_scans_use**
Number of scans to use  for each excited mode from the total number of scans identified at the paths from `datasets`.

2. **bunches_to_skip**
If the datasets contain empty bunches that need to be removed before analysis - then the required skip can be specified here.
Example: If every **fourth bunch** is filled when the data was collected for a 1296 bunch sample,  then `bunches_to_skip = 4`

### Hyperparameters
1. **UseGPU**
Enable this only if [GPU Support](https://www.tensorflow.org/install/gpu) for Tensorflow has been installed and GPU needs to be used for training/predictions.

2.  **batch_size**
The batch size at which the data is fed to the model for training or predictions.\
	```python
	Use the following batch sizes if doing predictions:
	For Denoising Autoencoder use batch_size = number of filled bunches
	Example: batch_size = 324

	For Growth Rate Prediction use batch_size = 1
	```

3.  **kernel_size**
The kernel size for the 1D convolution layers in the denoising autoencoder.

4. **shuffle_samples**
Whether samples need to be shuffled while training the model. Sample shuffling is defaulted to False for predictions so this argument is ignored for `UseObjective = 2`.

5. **epochs**
The number of epochs for which to train the model.

6. **reduced_timesteps**
The Denoising Autoencoder is **not padded in order to preserve the length of the time-series**, so the output is reduced depending on the size of the kernel used. This is the input for how many timesteps need to be truncated from the end to match the output shape.

	If this value cannot be inferred directly, simply compile the model and look at the output shape of the model summary to identify how many timesteps have been reduced.

	Example: **reduced_timesteps = 5696-5680 = 16**
```bash
Model: "model"
  _________________________________________________________________
  Layer (type)                 Output Shape              Param #
  =================================================================
  input_1 (InputLayer)         [(None, 5696, 1)]         0
  _________________________________________________________________
  conv1d (Conv1D)              (None, 5692, 32)          192
  _________________________________________________________________
  conv1d_1 (Conv1D)            (None, 5688, 64)          10304
  _________________________________________________________________
  conv1d_2 (Conv1D)            (None, 5684, 64)          20544
  _________________________________________________________________
  conv1d_3 (Conv1D)            (None, 5680, 1)           321
  =================================================================
  ```
7. **truncate_at_timestep**
This will truncate the timeseries at the given timestep before using it for training.
Example: **truncate_at_timestep = 500** (for growth rate identification)

8. **max_queue**
Tensorflow Keras Model fit [parameter](https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit) - max_queue_size. It determines the maximum queue size of data loaded onto memory for training.

## License
Work was supported by the U.S. Department of Energy, Office of Science, Office of Nuclear Physics, under contract DE-AC02-06CH11357.

The submitted manuscript has been created by UChicago Argonne, LLC as Operator of Argonne National Laboratory (”Argonne”) under Contract No. DE-AC02- 06CH11357 with the U.S. Department of Energy. The U.S. Government retains for 12 itself, and others acting on its behalf, a paid-up, nonexclusive, irrevocable worldwide license in said article to reproduce, prepare derivative works, distribute copies to the public, and perform publicly and display publicly, by or on behalf of the Government. The Department of Energy will provide public access to these results of federally sponsored research in accordance with the DOE Public Access Plan.
