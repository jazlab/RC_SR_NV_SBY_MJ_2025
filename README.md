# Repository Overview

This repository contains code to generate figures for the paper:

**Evidence accumulation from experience and observation in the cingulate cortex**

## Instructions

- The list of dependencies is included in `paper.yml`.
- **R** is required. Please ensure that the environment variable is set to your installation location.
- To create the plots, run `make_plots.sh`. This script executes Python scripts that produce PDF files corresponding to the paper panels (as indicated by the file names). The generated plots are saved in `data/plots_paper`.

## Code Overview

```plaintext
code
|-- behavior      : Analysis of behavior in the task; produces the majority of behavior-related plots.
|-- eye           : Analysis of eye movements; produces eye gaze-related plots.
|-- msi_src       : RNN instantiation of architectural hypotheses; produces RNN-related plots.
|-- pyTdr         : A port of TDR (xx cite) from MATLAB to Python; includes associated dimension analysis and produces related plots.
|-- single_unit   : Analysis of single unit activity; produces the majority of single unit plots.
|-- spikes        : Code for generating rasters and histograms.
|-- utils         : Supporting code.
|-- paper.yml     : Dependency list.
