Instructions

The list of dependencies is included in paper.yml
Additionally, R is required. Please make sure the environment variable is set to your installation location.
To make plots, run make_plots.sh which will run python scripts that produce pdf files corresponding to panels in the paper indicated by its file name. The result is saved in data/plots_paper.


Code overview

code
|--behavior: analysis of behavior in the task, produces majority of behavior related plots
|--eye: analysis of eye movement, produces eye gaze related plots
|--msi_src: RNN instantiation of architectural hypotheses, produces RNN-related plots.
|--pyTdr: a port of TDR (xx cite) from matlab to python, includes associated dimension analysis and produces related plots.
|--single_unit: analysis of single unit activity, produces majority of single unit plots.
|--spikes: code for making raster and histograms.
|--utils: supporting code.
|--paper.yml: dependencies
