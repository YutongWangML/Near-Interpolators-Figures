# Figures for "Near-Interpolators"
Official code repository for 
"Near-Interpolators: Rapid Norm Growth and the Trade-Off between Interpolation and Generalization:

See below for instructions on reproducing the figures.

## Instructions: Figure 1 to 4
For each figure panel in the paper, there are three associated directories inside the directory `figure_and_experiment_code`:
- NAME-plot/
- NAME-run1/
- NAME-run2/

For instance, NAME can be one of `figure1-left`, `figure1-right`, `figure4-bottom-left`, and so on.

To recreate the figures, for say `figure1-left` do the following:

```
cd figure1-left-run1
source run_experiment.sh
cd ..
cd figure1-left-run2
source run_experiment.sh
cd ..
cd figure1-left-plot
jupyter run plot_figure.ipynb
```

The figure will be in `figure1-left-plot/outputs/`

## Instructions: Figure 5 and 6

Navigate to the respective folder `figure5` or `figure6`.

Run the first block 
```
!pip install git+https://github.com/treforevans/uci_datasets.git
```
if running for the first time.

Run all cells.
