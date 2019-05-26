#
###################################
#2018-04-11
#Jin Zhang@Stony Brook University
######################################

###For VASP
###----------------------------------Readme------------------------------------------------------------
###This script is to calculate the Tensile strength and shear strength using VASP.

###1. To calculate the tensile strength. ###
###If we want to calculate the tensile strength using this script, we should firstly add
###" FCELL(1,1)=0.0 " (add " FCELL(1,1)=0.0 " after REAL(q)FCELL(3,3)) to the constr_cell_relax.F
###in the VASP. This modification means that we only optimize the y and z axis, x axis will not be
###optimized. Then recompile the VASP again. Finally, we can use this script to calculate the tensile
###strength.

###For the tensile strength, this script firstly rotate the tensile axis which we want to calculate into
###[1, 0, 0] axis. The reason of rotating to [1, 0, 0] axis is that we set "FCELL(1,1)=0.0" in the
###constr_cell_relax.F in the VASP. e.g. if we want to calculate the tensile strength along [1, 1, 1] direction
###of diamond, we need to rotate [1, 1, 1] direction to [1, 0, 0] axis.

###2. To calculate the shear strength. ###
###If we want to calculate the shear strength using this script, we should firstly add
###" FCELL(1,3)=0.0 " and " FCELL(3,1)=0.0 " (add " FCELL(1,3)=0.0 " and " FCELL(3,1)=0.0 "
###after REAL(q)FCELL(3,3)) to the constr_cell_relax.F in the VASP. This modification means that
###we only optimize the axis except for the xz and zx axies. Then recompile the VASP again.
###Finally, we can use this script to calculate the tensile strength.

###For the shear strength, this script firstly rotate the normal axis of shear plane to
###[0, 0, 1] axis. And then rotate the shear direction to [1, 0, 0] axis. The reason is that
###we set "FCELL(1,3)=0.0" and "FCELL(3,1)=0.0" to the constr_cell_relax.F in the VASP. e.g. if we
###want to calculate the shear strength along (1, 1, 1) shear plane and [1, 1, -2] shear direction for
###diamond, we need to rotate the [1, 1, 1] direction to [0, 0, 1] axis and rotate [1, 1, -2] direction
###to [1, 0, 0] axis.
###--------------------------------------------------------------------------------------------------------


import numpy as np
import math
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d as plt3d
from itertools import product, combinations
from matplotlib import style
from fractions import gcd


#------------------------------------------------------------------------------------

###start to read the input file
###stra=0.02 ######provide the strain
###times=9
P0=[int(x) for x in input("If the dash green line is on X axis, then rotation is correct. \n What is the shear plane? e.g. 1 1 1 \n").split()]
L0=[int(x) for x in input("What is the shear direction? e.g. 1 1 -2 \n").split()]

#P0=np.array([1, 1, 1]) #provide the axis before rotation. P0 is the tensile direction we want to calculate.
###finish to read the input file
###Q0=np.array([1, 0, 0]) #after rotation. Q0 is the x axis. Since the x axis will not be optimized as we set in VASP, so Q0 is [1, 0, 0]
#print(type(P0))
print(len(P0))
if len(P0)==3:
    P0=P0
elif len(P0)==4:
    U=P0[1]+2*P0[0]
    V=P0[0]+2*P0[1]
    W=P0[3]
    UVW_gcd=gcd(gcd(U,V),W)
    P0=np.array([U/UVW_gcd, V/UVW_gcd, W/UVW_gcd])
    print("&&&&&&&&&&&&&&&&")
    print(P0)
    print("&&&&&&&&&&&&&&&&")

#------------------------------------------------------------------------------------
###open POSCAR
t=[]
t1=[]
with open ("POSCAR") as poscar0:
    pos0=poscar0.readlines()
    length=len(pos0)
    for i in range(2,5):
        f=pos0[i]
        f=f.split()
        t.extend(f)
print(t)

for i in range(len(t)):
        t1.extend([float(t[i])])
pos_array=np.array(t1).reshape((3,3)) ####read basis vector from POSCAR 
print(pos_array)

#----------
oo=np.array([0.0, 0.0, 0.0])
oa=np.array([pos_array[0][0], pos_array[0][1], pos_array[0][2]])
ob=np.array([pos_array[1][0], pos_array[1][1], pos_array[1][2]])
oc=np.array([pos_array[2][0], pos_array[2][1], pos_array[2][2]])
print(oa)
od=oa+ob
oe=oa+oc
og=ob+oc
of=od+oe-oa
print(of)
print(of[0])
print(of[1])
print(of[2])
#---------------------------------
#P0=np.array([1, 1, 1])
P0_abc=np.array([[P0[0], 0, 0],   
                [0, P0[1], 0],
                [0, 0, P0[2]]])

basis=np.array([oa, ob, oc])

P0_abc_basis=np.dot(P0_abc,basis)
nor_sh_pla=P0_abc_basis[0]+P0_abc_basis[1]+P0_abc_basis[2] # the position of the normal of shear plane 
print("*******************")
#print(ten_dir)
print("*******************")

#---------------------------------
#P0=np.array([1, 1, 1])
L0_abc=np.array([[L0[0], 0, 0],   
                [0, L0[1], 0],
                [0, 0, L0[2]]])

basis=np.array([oa, ob, oc])

L0_abc_basis=np.dot(L0_abc,basis)
sh_dir=L0_abc_basis[0]+L0_abc_basis[1]+L0_abc_basis[2] # the position of shear direction

###start to plot figure before rotation
fig = plt.figure()
fig.set_size_inches(10,10)
ax = fig.add_subplot(111, projection='3d', aspect='equal')
print(og)

### x, y, z coordinate system
ax.scatter(0.0, 0.0, 0.0, color='red', marker='o')
ox=np.array([6.0, 0.0, 0.0])
oy=np.array([0.0, 6.0, 0.0])
oz=np.array([0.0, 0.0, 6.0])
#ax.quiver([0, 0, 0], [0, 0, 0], [0, 0, 0], [8, 0, 0], [0, 8, 0], [0, 0, 8], length=0.1, normalize=True)
ax.quiver([0, 0, 0], [0, 0, 0], [0, 0, 0], [6, 0, 0], [0, 6, 0], [0, 0, 6], color='red', length=7.5, arrow_length_ratio=0.1, normalize=True)
#ax.arrow(0, 0, 0, 0, 0, 6, head_width=0.05, head_length=0.1, fc='k', ec='k')
#plt.text(7.7, 0.0, 0.0, r'$\vec x$', fontsize=24, color='red', fontweight='bold')
ax.text(8, 0, 0, "x", color='red', fontsize=14)
ax.text(0, 8, 0, "y", color='red', fontsize=14)
ax.text(0, 0, 8, "z", color='red', fontsize=14)

line1 = plt3d.art3d.Line3D((oa[0],od[0],ob[0],og[0],oc[0],oe[0]),(oa[1],od[1],ob[1],og[1],oc[1],oe[1]),(oa[2],od[2],ob[2],og[2],oc[2],oe[2]), color='blue')
line2 = plt3d.art3d.Line3D((oe[0],oa[0],oo[0],oc[0],og[0],of[0]),(oe[1],oa[1],oo[1],oc[1],og[1],of[1]),(oe[2],oa[2],oo[2],oc[2],og[2],of[2]), color='blue')
line3 = plt3d.art3d.Line3D((oe[0],of[0],od[0],ob[0],oo[0]),(oe[1],of[1],od[1],ob[1],oo[1]),(oe[2],of[2],od[2],ob[2],oo[2]), color='blue')

###the normal of shear plane 
line4 = plt3d.art3d.Line3D((oo[0],nor_sh_pla[0]),(oo[1],nor_sh_pla[1]),(oo[2],nor_sh_pla[2]), ls='--', color='blue')

###the shear direction
line5 = plt3d.art3d.Line3D((oo[0],sh_dir[0]),(oo[1],sh_dir[1]),(oo[2],sh_dir[2]), ls=':', color='blue')

#-----------------------------------------------
###open POSCAR_rota
t2=[]
t3=[]
with open ("POSCAR_rota") as poscar0:
    pos0=poscar0.readlines()
    length=len(pos0)
    for i in range(2,5):
        f=pos0[i]
        f=f.split()
        t2.extend(f)
print(t2)

for i in range(len(t1)):
        t3.extend([float(t2[i])])
pos_array_rota=np.array(t3).reshape((3,3)) ####read basis vector from POSCAR 
print(pos_array_rota)

#----------
oo_rota=np.array([0.0, 0.0, 0.0])
oa_rota=np.array([pos_array_rota[0][0], pos_array_rota[0][1], pos_array_rota[0][2]])
ob_rota=np.array([pos_array_rota[1][0], pos_array_rota[1][1], pos_array_rota[1][2]])
oc_rota=np.array([pos_array_rota[2][0], pos_array_rota[2][1], pos_array_rota[2][2]])
print(oa)
od_rota=oa_rota+ob_rota
oe_rota=oa_rota+oc_rota
og_rota=ob_rota+oc_rota
of_rota=od_rota+oe_rota-oa_rota

basis_rota=np.array([oa_rota, ob_rota, oc_rota])

P0_abc_basis_rota=np.dot(P0_abc,basis_rota)
nor_sh_pla_rota=P0_abc_basis_rota[0]+P0_abc_basis_rota[1]+P0_abc_basis_rota[2] # the position of tensile direction


L0_abc_basis_rota=np.dot(L0_abc,basis_rota)
sh_dir_rota=L0_abc_basis_rota[0]+L0_abc_basis_rota[1]+L0_abc_basis_rota[2] # the position of tensile direction
###start to plot figure after rotation

#line1 = plt3d.art3d.Line3D((oa), (od), (ob), (og), (oc), (oe), (oa), (oo), (oc), (og), (of), (od), color='blue')
line6 = plt3d.art3d.Line3D((oa_rota[0],od_rota[0],ob_rota[0],og_rota[0],oc_rota[0],oe_rota[0]),(oa_rota[1],od_rota[1],ob_rota[1],og_rota[1],oc_rota[1],oe_rota[1]),(oa_rota[2],od_rota[2],ob_rota[2],og_rota[2],oc_rota[2],oe_rota[2]), color='green')
line7 = plt3d.art3d.Line3D((oe_rota[0],oa_rota[0],oo_rota[0],oc_rota[0],og_rota[0],of_rota[0]),(oe_rota[1],oa_rota[1],oo_rota[1],oc_rota[1],og_rota[1],of_rota[1]),(oe_rota[2],oa_rota[2],oo_rota[2],oc_rota[2],og_rota[2],of_rota[2]), color='green')
line8 = plt3d.art3d.Line3D((oe_rota[0],of_rota[0],od_rota[0],ob_rota[0],oo_rota[0]),(oe_rota[1],of_rota[1],od_rota[1],ob_rota[1],oo_rota[1]),(oe_rota[2],of_rota[2],od_rota[2],ob_rota[2],oo_rota[2]), color='green')

###tensile direction after rotation
line9 = plt3d.art3d.Line3D((oo_rota[0],nor_sh_pla_rota[0]),(oo_rota[1],nor_sh_pla_rota[1]),(oo_rota[2],nor_sh_pla_rota[2]), ls='--', color='green')

line10 = plt3d.art3d.Line3D((oo_rota[0],sh_dir_rota[0]),(oo_rota[1],sh_dir_rota[1]),(oo_rota[2],sh_dir_rota[2]), ls=':', color='green')


ax.add_line(line1)
ax.add_line(line2)
ax.add_line(line3)
ax.add_line(line4)
ax.add_line(line5)
ax.add_line(line6)
ax.add_line(line7)
ax.add_line(line8)
ax.add_line(line9)
ax.add_line(line10)
ax.set_xlim(-7, 7)
ax.set_ylim(-7, 7)
ax.set_zlim(-7, 7)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

###rotat figure
##for angle in range(0, 360):
##  ax.view_init(30, angle)
##  plt.draw()
##  plt.pause(.001)

plt.show()











