# Repository Overview

This repository contains code to generate figures for the paper:

**Evidence accumulation from experience and observation in the cingulate cortex**

## System requirements
- Code has been tested in MacOS version 12.7.6 and 15.3.1
- Python version 3.9
- The list of python dependencies is included in `paper.yml`.
- **R** is required. 

## Installation guide
- Install conda or miniconda on your system
- Run 'conda env create -f paper.yml' to set up a conda environment
- Install R on your system
- Set environment variable for R (replace path to R as needed): 
export R_HOME="/Library/Frameworks/R.framework/Resources"
export PATH="/Library/Frameworks/R.framework/Resources/bin:$PATH"
export DYLD_LIBRARY_PATH="/Library/Frameworks/R.framework/Resources/lib:$DYLD_LIBRARY_PATH"
- Typical install time: 20 minutes.

## Demo
- To create the plots, run `make_plots.sh`. This script executes Python scripts that produce PDF files corresponding to the paper panels (as indicated by the file names). The generated plots are saved in `data/plots_paper`.
- Expected runtime: 30 minutes.

## Instructions for use
- To run analysis on your own data, replace files in data/ with your own.

## Code Overview

```plaintext
code
|-- behavior      : Analysis of behavior in the task; produces the majority of behavior-related plots.
|-- eye           : Analysis of eye movements; produces eye gaze-related plots.
|-- msi_src       : RNN instantiation of architectural hypotheses; produces RNN-related plots.
|-- pyTdr         : A port of TDR (Mante et al., Nature, 2013) from MATLAB to Python; includes associated dimension analysis and produces related plots.
|-- single_unit   : Analysis of single unit activity; produces the majority of single unit plots.
|-- spikes        : Code for generating rasters and histograms.
|-- utils         : Supporting code.
|-- paper.yml     : Dependency list.
