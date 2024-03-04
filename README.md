# GalacticScreeningConditions

This repository produces the figures contained within [our paper on galactic (scalar-tensor) screening conditions](https://arxiv.org/abs/2310.19955). 

## Usage

The code to create the figures is contained within ```ScreeningPaperFigs.py.```

The solutions required to run this file can be made available upon request (email: bradley.march@nottingham.ac.uk), or can be produced by running ```run_solutions.py``` (*note: this will take considerable time*).

The $f(R)$ and symmetron scalar-tensor models we use are described in sections 2.3 and 2.4, respectfully.
Section 3.2 highlights the methods used in the ```Solvers``` folder to numerically derive the scalar field profile from the $f(R)$ and symmetron equation of motion.

In our paper, the galactic model is set up in section 3.1, with an overview of the pipeline to relate each density parameter shown in the Appendix. The code to describe our galactic model is contained in ```Packages/galaxy_relations.py```. 

Our analysis of the screening conditions, section 3.3 of the paper, is quantified in ```Packages/fR_functions.py``` and ```Packages/sym_functions.py```. 




