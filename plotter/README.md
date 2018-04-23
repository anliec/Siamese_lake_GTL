# Plotter

A notebook to analyse the results generated during the training.

## Requirement

 - Python >= 3.6
 - matplotlib
 - seaborn
 - pandas
 - numpy
 - sqlite3 (with extension function activated)
 
(Also use standard lib such as `os` and `glob`)

## How to use ?

The jupyter notebook load the results writen as scv in the `../runs_results/`
directory into a SQLite database. This database is then queried to display
the effects of various parameters.  



