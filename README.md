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

```Solvers``` and code within was created by [**Aneesh P. Naik**](https://github.com/aneeshnaik).

All other code was created by **Bradley March**.

## Citation

If you use this code for your research please include the following citation:
```
@ARTICLE{2024JCAP...04..004B,
       author = {{Burrage}, Clare and {March}, Bradley and {Naik}, Aneesh P.},
        title = "{Accurate computation of the screening of scalar fifth forces in galaxies}",
      journal = {\jcap},
     keywords = {dark energy theory, galaxy surveys, modified gravity, Astrophysics - Cosmology and Nongalactic Astrophysics, Astrophysics - Astrophysics of Galaxies, General Relativity and Quantum Cosmology},
         year = 2024,
        month = apr,
       volume = {2024},
       number = {4},
          eid = {004},
        pages = {004},
          doi = {10.1088/1475-7516/2024/04/004},
archivePrefix = {arXiv},
       eprint = {2310.19955},
 primaryClass = {astro-ph.CO}
}
```


## License

Copyright (2024) Bradley March.

`GalacticScreeningConditions` is free software made available under the MIT license. For details see LICENSE.




