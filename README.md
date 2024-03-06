# GalacticScreeningConditions

This repository produces the figures contained within [our paper on galactic (scalar-tensor) screening conditions](https://arxiv.org/abs/2310.19955). 

See each relevant section in the paper for a full description of the methods used throughout this repository.

## Usage

The code to create the figures is contained within ```ScreeningPaperFigs.py.```

The solutions required to run this file can be made available upon request (email: bradley.march@nottingham.ac.uk), or can be produced by running ```run_solutions.py``` (*note: this will take considerable time*).

The $f(R)$ and symmetron scalar-tensor models we use are described in sections 2.3 and 2.4, respectfully.
Section 3.2 highlights the methods used in the ```Solvers``` folder to numerically derive the scalar field profile from the $f(R)$ and symmetron equation of motion.

In our paper, the galactic model is set up in section 3.1, with an overview of the pipeline to relate each density parameter shown in Appendix B. The code to derive our galactic model is contained in ```Packages/galaxy_relations.py```. 

Our analysis of the screening conditions, section 3.3 of the paper, is quantified in ```Packages/fR_functions.py``` and ```Packages/sym_functions.py```. 

## Authors

```Solvers``` and code within was created by **Aneesh P. Naik**.

All other code was created by **Bradley March**.

## Citation

*Accepted for publishing in JCAP. Will update with full reference upon finalisation.*

If you use this code for your research please include the following citation:
```
@ARTICLE{2023arXiv231019955B,
       author = {{Burrage}, Clare and {March}, Bradley and {Naik}, Aneesh P.},
        title = "{Accurate Computation of the Screening of Scalar Fifth Forces in Galaxies}",
      journal = {arXiv e-prints},
     keywords = {Astrophysics - Cosmology and Nongalactic Astrophysics, Astrophysics - Astrophysics of Galaxies, General Relativity and Quantum Cosmology},
         year = 2023,
        month = oct,
          eid = {arXiv:2310.19955},
        pages = {arXiv:2310.19955},
          doi = {10.48550/arXiv.2310.19955},
archivePrefix = {arXiv},
       eprint = {2310.19955},
 primaryClass = {astro-ph.CO},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2023arXiv231019955B},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```


## License

Copyright (2024) Bradley March.

`GalacticScreeningConditions` is free software made available under the MIT license. For details see LICENSE.




