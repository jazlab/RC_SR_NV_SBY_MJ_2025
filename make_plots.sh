# run through all plotting functions

# activate environment
source ~/.zshrc
conda activate phys

# export env variables for R
export R_HOME="/Library/Frameworks/R.framework/Resources"
export DYLD_LIBRARY_PATH="/Library/Frameworks/R.framework/Resources/lib:$DYLD_LIBRARY_PATH"

# set the demo variable to true by default
demo=true

# run plotting functions
CURRENT_DIR=$(pwd)

cd $CURRENT_DIR/behavior
python plot_stat_monkey_labeled.py
python plot_stat_human_labeled.py
python plot_stat_single_labeled.py
python plot_stat_single_human_labeled.py
python model_ignoring_agent_human_labeled.py
python model_ignoring_agent_labeled.py

# add a condition to skip this if the variable 'demo' is true
# this is because eye data is too large to fit in the repo
if [ "$demo" = false ]; then
    cd $CURRENT_DIR/eye
    python plot_session_heatmap_labeled.py
fi

cd $CURRENT_DIR/single_unit
python plot_roc_histograms_labeled.py
python plot_roc_nback_both_labeled.py

# spike data is too large to fit in the repo
if [ "$demo" = false ]; then
    python plot_ratediff_line_labeled.py
    cd $CURRENT_DIR/spikes
    python plot_raster_labeled.py
    python plot_rates_labeled.py
fi

cd $CURRENT_DIR/pyTdr
python plot_actor_observer_angles_labeled.py
python plot_actor_observer_projection_labeled.py
python plot_switch_dir_single_sessions_labeled.py
python plot_train_test_labeled.py

cd $CURRENT_DIR/msi_src
python plot_rnn_behavior_and_geometry.py