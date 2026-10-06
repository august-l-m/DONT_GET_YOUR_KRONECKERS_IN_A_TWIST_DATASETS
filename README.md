# Repo with all datasets used in the numerical experiments section of "*Don't Get Your Kroneckers in a Twist: Gaussian Processes on High-Dimensional Incomplete Grids*"

## Overall data structure

### Exact GPR tests

These make use of a generic table.
The first line is a header line that is skipped.
Then the points come as x1 x2 ... xD y, all space separated.

### CUTS-GPR tests

The first line is the reference value for all modes (0.0) and the energy of this reference structure.
All following lines are structured as Displaced dimensions : Displacements along each dimension : Energy

## Data generation

### PES datasets

All PES datasets are generated using MIDASCPP with XTB.
These all need a .mmol file, all of which can be found in the Molecules folder.
Example calculation with midas input file and necessary interface files can be found in example.
Links:
MIDASCPP: https://midascpp.gitlab.io/
XTB: https://xtb-docs.readthedocs.io/en/latest/

### 6.2) Exact baseline comparison

Set of 8, 2 mode coupled potential energy surfaces with variying dimensionality

Molecules included
  - Water (3D)
  - Hydrogen cyanide (4D)
  - Formaldehyde (6D)
  - Acetylene (7D)
  - Difluoromethane (9D)
  - Cyanoacetylene (10D)
  - *trans*-Difluoroethylene (12D)
  - Diacetylene (13D)

### 6.4) Application to PES data

Set of 10, 24 dimensional, 3 mode coupled, potential energy surfaces.

Molecules included:
  - Butadiene
  - DMSO
  - Ethylamine
  - Ethylene Glycol
  - Nitroethane
  - Propanal
  - Pyrazine
  - Pyrrole
  - Thioacetone
  - Vinylformamide

### End-to-end Scaling in High Dimension

Synthetic data

More details will follow
