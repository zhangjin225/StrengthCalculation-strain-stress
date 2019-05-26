#
###################################
#2018-04-11
#Jin Zhang@Stony Brook University
######################################

###For QE
###----------------------------------Readme------------------------------------------------------------
###This script is to calculate the Tensile strength and shear strength using QE.

###1. To calculate the tensile strength. ###
###If we want to calculate the tensile strength using this script, we should firstly add the following 
#            CASE ('xfixed')  ! don't optimize X axis, optimize Y and Z axes
#              iforceh      = 1
#              iforceh(1,1) = 0
#              iforceh(1,2) = 0
#              iforceh(1,3) = 0
### to the qe/Modules/cell_base.f90--init_dofree in QE. This modification means that we only optimize
###the y and z axis, x axis will not be optimized. Then recompile the QE again. Finally, we can use
###this script to calculate the tensile strength.

###For the tensile strength, this script firstly rotate the tensile axis which we want to calculate into
###[1, 0, 0] axis. The reason of rotating to [1, 0, 0] axis is that we add above these to the
###qe/Modules/cell_base.f90--init_dofree in QE. e.g. if we want to calculate the tensile strength
###along [1, 1, 1] direction of diamond, we need to rotate [1, 1, 1] direction to [1, 0, 0] axis.

###2. To calculate the shear strength. ###
###If we want to calculate the shear strength using this script, we should firstly add the following
#            CASE ('xzfixed') ! don't optimize XZ and ZX axes.
#              iforceh      = 1
#              iforceh(1,1) = 0
#              iforceh(1,2) = 0
#              iforceh(1,3) = 0
#              iforceh(3,1) = 0
#              iforceh(3,2) = 0
#              iforceh(3,3) = 0

###to the qe/Modules/cell_base.f90--init_dofree in QE. This modification means that
###we only optimize the axis except for the xz and zx axies. Then recompile the QE again.
###Finally, we can use this script to calculate the tensile strength.

###For the shear strength, this script firstly rotate the normal axis of shear plane to
###[0, 0, 1] axis. And then rotate the shear direction to [1, 0, 0] axis. The reason is that
###we add above these to the qe/Modules/cell_base.f90--init_dofree in QE. e.g. if we
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
P0=[int(x) for x in input("If the dash green line is on X axis, then rotation is correct. \n What is the tensile direction? e.g. 1 1 1 \n").split()]
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
with open ("relax.in") as poscar0:
    pos0=poscar0.readlines()
    length=len(pos0)
    for i in range(length):
        if "CELL_PARAMETERS" in pos0[i]:
            index=i
    for j in range(index+1,index+4):
        f=pos0[j]
        f=f.split()
        t.extend(f)
    for k in range(len(t)):
        t1.extend([float(t[k])])
#for i in range(len(t)):
#        t1.extend([float(t[i])])
pos_array=np.array(t1).reshape((3,3)) ####read basis vector from POSCAR 
print(pos_array)

#----------
oo=np.array([0.0, 0.0, 0.0])
oa=np.array([pos_array[0][0], pos_array[0][1], pos_array[0][2]])
ob=np.array([pos_array[1][0], pos_array[1][1], pos_array[1][2]])
oc=np.array([pos_array[2][0], pos_array[2][1], pos_array[2][2]])
#print(oa)
od=oa+ob
oe=oa+oc
og=ob+oc
of=od+oe-oa
#print(of)
#print(of[0])
#print(of[1])
print(of[2])
#---------------------------------
#P0=np.array([1, 1, 1])
P0_abc=np.array([[P0[0], 0, 0],   
                [0, P0[1], 0],
                [0, 0, P0[2]]])

basis=np.array([oa, ob, oc])

P0_abc_basis=np.dot(P0_abc,basis)
ten_dir=P0_abc_basis[0]+P0_abc_basis[1]+P0_abc_basis[2] # the position of tensile direction
print("*******************")
print(ten_dir)
print("*******************")


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

###tensile direction
line4 = plt3d.art3d.Line3D((oo[0],ten_dir[0]),(oo[1],ten_dir[1]),(oo[2],ten_dir[2]), ls='--', color='blue')
#-----------------------------------------------
###open POSCAR_rota
t2=[]
t3=[]
with open ("relax_rota.in") as poscar0:
    pos0=poscar0.readlines()
    length=len(pos0)
    for i in range(length):
        if "CELL_PARAMETERS" in pos0[i]:
            index=i
    for j in range(index+1,index+4):
        f=pos0[j]
        f=f.split()
        t2.extend(f)
    for k in range(len(t2)):
        t3.extend([float(t2[k])])
        
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
ten_dir_rota=P0_abc_basis_rota[0]+P0_abc_basis_rota[1]+P0_abc_basis_rota[2] # the position of tensile direction

###start to plot figure after rotation

#line1 = plt3d.art3d.Line3D((oa), (od), (ob), (og), (oc), (oe), (oa), (oo), (oc), (og), (of), (od), color='blue')
line5 = plt3d.art3d.Line3D((oa_rota[0],od_rota[0],ob_rota[0],og_rota[0],oc_rota[0],oe_rota[0]),(oa_rota[1],od_rota[1],ob_rota[1],og_rota[1],oc_rota[1],oe_rota[1]),(oa_rota[2],od_rota[2],ob_rota[2],og_rota[2],oc_rota[2],oe_rota[2]), color='green')
line6 = plt3d.art3d.Line3D((oe_rota[0],oa_rota[0],oo_rota[0],oc_rota[0],og_rota[0],of_rota[0]),(oe_rota[1],oa_rota[1],oo_rota[1],oc_rota[1],og_rota[1],of_rota[1]),(oe_rota[2],oa_rota[2],oo_rota[2],oc_rota[2],og_rota[2],of_rota[2]), color='green')
line7 = plt3d.art3d.Line3D((oe_rota[0],of_rota[0],od_rota[0],ob_rota[0],oo_rota[0]),(oe_rota[1],of_rota[1],od_rota[1],ob_rota[1],oo_rota[1]),(oe_rota[2],of_rota[2],od_rota[2],ob_rota[2],oo_rota[2]), color='green')

###tensile direction after rotation
line8 = plt3d.art3d.Line3D((oo_rota[0],ten_dir_rota[0]),(oo_rota[1],ten_dir_rota[1]),(oo_rota[2],ten_dir_rota[2]), ls='--', color='green')


ax.add_line(line1)
ax.add_line(line2)
ax.add_line(line3)
ax.add_line(line4)
ax.add_line(line5)
ax.add_line(line6)
ax.add_line(line7)
ax.add_line(line8)
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











