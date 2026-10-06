#!/bin/bash
#
NAT=`head -1 midasifc.xyz_input`
echo " number of atoms" $NAT
echo $NAT > coordinates.xyz
echo >> coordinates.xyz
cat midasifc.xyz_input | tail -$NAT >> coordinates.xyz
