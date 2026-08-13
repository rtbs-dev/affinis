# Contributing

## Usage & Attribution

We would love to hear about how you use `affinis`!

### Citation

If you use `affinis` in your work, please cite the following record:

=== "Bibtex"
    ```bibtex
    @misc{dataset_key,
      author       = {Sexton, Rachael T.},
      title        = {Affinis},
      subtitle     = {tools for inferring relations from co-occurrence data}
      year         = {2026},
      publisher    = {National Institute of Standards and Technology},
      doi          = {10.18434/mds2-4107},
      howpublished = {\url{https://doi.org/10.18434/mds2-4107}}
    }
    ```
=== "Plain Text"
    > Sexton, Rachael T (2026), Affinis: tools for inferring relations from co-occurrence data, National Institute of Standards and Technology, <https://doi.org/10.18434/mds2-4107>


### Disclaimer

You are free to use `affinis` in your tools and research!
Affinis and all of it's associated documentation are in the public domain.

??? info "NIST Disclaimer"

    NIST-developed software is provided by NIST as a public service.
    You may use, copy, and distribute copies of the software in any
    medium, provided that you keep intact this entire notice. You may
    improve, modify, and create derivative works of the software or
    any portion of the software, and you may copy and distribute such
    modifications or works. Modified works should carry a notice
    stating that you changed the software and should note the date
    and nature of any such change. Please explicitly acknowledge the
    National Institute of Standards and Technology as the source of
    the software.

    NIST-developed software is expressly provided "AS IS." NIST MAKES
    NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT, OR ARISING BY
    OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
    WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE,
    NON-INFRINGEMENT, AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR
    WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED
    OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES
    NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE
    SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE
    CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE
    SOFTWARE.

    You are solely responsible for determining the appropriateness of
    using and distributing the software and you assume all risks
    associated with its use, including but not limited to the risks
    and costs of program errors, compliance with applicable laws,
    damage to or loss of data, programs or equipment, and the
    unavailability or interruption of operation. This software is not
    intended to be used in any situation where a failure could cause
    risk of injury or damage to property. The software developed by
    NIST employees is not subject to copyright protection within the
    United States.

## For Developers

If you would like to contribute to `affinis`, please fork the repository and open a pull-request.

### Dependency & Environment Setup
We use a number of developer tools, but the primary prerequisite to get started is `poetry`, which we have used to manage virtual environments and track dependencies.

Once you have [poetry installed](https://python-poetry.org/docs/#installation), clone this repository and run `poetry install` in the top level directory.
This will install the necessary development dependencies, like jupytext, zensical, pytest, and hypothesis.

### Project Structure


```
.
├── affinis            # Source for library
├── docs               # Source for documentation
│   ├── api
│   ├── case-studies
│   ├── css
│   ├── images
│   ├── javascripts
│   └── user-guide
├── notebooks          # Jupyter notebook source for tutorials
│   ├── case-studies
│   └── user-guide
└── tests              # Unit tests (pytest+hypothesis)
```

### Unit Tests

Tests should be added to the `tests/` directory n a per-module basis.
For instance, for the `associations.py` module, `tests/test_associations.py`.

To run unit tests during development, use this command to perform the property-based tests we have built for this library. 
```
poetry run pytest tests/
```

