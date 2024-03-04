# GalacticScreeningConditions

This repository produces the figures contained within [our paper on galactic (scalar-tensor) screening conditions](https://arxiv.org/abs/2310.19955). 

## Usage

The code to create the figures is contained within ```ScreeningPaperFigs.py.```

The solutions required to run this file can be made available upon request (email: bradley.march@nottingham.ac.uk), or can be produced by running ```run_solutions.py``` (_note: this will take considerable time _).

The scalar-tensor models we use are described in section 2.3 ($f(R)$) and 2.4 (symmetron) of the paper.
Section 3.2 highlights the methods used in the ```Solvers``` folder to numerically derive the scalar field profiles.

The code to describe our galactic model is contained in ```Packages/galaxy_relations.py```In our paper the galactic model is set up section 3.1, with an overview of the pipeline to relate each density parameter shown in the Appendix. 

