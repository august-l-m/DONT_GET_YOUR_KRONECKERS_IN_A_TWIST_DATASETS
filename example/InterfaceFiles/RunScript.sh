#!/bin/bash
#
#  Arguments:
#
#  $1 : The scratch directory
#  $2 : The done file to be touched upon
#       successful completion
#
#
#
# function for creating a define file
#
create_property_file() {
   energy=`grep "TOTAL ENERGY" output | awk '{print $4}'`
   NAT=`grep "number of atoms" output | awk '{print $5}'`

   echo "GROUND_STATE_ENERGY     " $energy > midasifc.prop_general

   echo $NAT > midasifc.cartrot_xyz
   echo " returned coordinates " >> midasifc.cartrot_xyz
   N=$(expr $NAT)
   BohrToAng=0.5291772108

    grep -A $(expr $NAT + 1)  "grad" gradient | tail -$N |    awk 'BEGIN {bohr=.5291772108; }
         {printf ("%2s      %20.15f  %20.15f  %20.15f \n",                        $4,$1*bohr,$2*bohr,$3*bohr)}
          END { }' >> midasifc.cartrot_xyz


}
#
# Change dir to $1 (the scratch dir)
cd $1
#
# All files are here, we may run
xtb coordinates.xyz --grad > output
create_property_file
