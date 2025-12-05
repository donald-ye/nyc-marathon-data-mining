# nyc-marathon-data-mining

This repository contains all the code for our CISC 3500 Final Project, titled Mining the Marathon.

Our source dataset can be found at (https://huggingface.co/datasets/donaldye8812/nyc-2025-marathon-splits).

How to use:
1) Run the preprocessing notebook ```cisc3500_final_preprocessing.ipynb``` on the raw dataset to produce ```all_runners_2025.csv```.
2) Run the mining script ```cisc_3500_final_mining.ipynb``` end-to-end. This will instantiate all of the models and produce the complete set of results. The code can also be run through the ```.py``` files, but the notebooks more effectively show the step-by-step operations.
3) The scripts will output all results, diagrams, and visualizations.
