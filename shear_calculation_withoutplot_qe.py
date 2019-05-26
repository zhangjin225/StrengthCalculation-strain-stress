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
stra=0.02 ######provide the strain
times=9
Sh_pla=[int(x) for x in input("What is the shear plane? e.g. 1 1 1 \n").split()]

if len(Sh_pla)==3:
    Sh_pla=Sh_pla
elif len(Sh_pla)==4:
    h=Sh_pla[0]
    k=Sh_pla[1]
    l=Sh_pla[3]
    #hkl_gcd=gcd(gcd(h,k),l)
    Sh_pla=np.array([h, k, l])

Sh_pla_point1=np.array([1/Sh_pla[0], 0, 0])
Sh_pla_point2=np.array([0, 1/Sh_pla[1], 0])
Sh_pla_point3=np.array([0, 0, 1/Sh_pla[2]])

Sh_pla_vec1=np.array([Sh_pla_point1[0]-Sh_pla_point2[0], Sh_pla_point1[1]-Sh_pla_point2[1], Sh_pla_point1[2]-Sh_pla_point2[2]])
Sh_pla_vec2=np.array([Sh_pla_point3[0]-Sh_pla_point2[0], Sh_pla_point3[1]-Sh_pla_point2[1], Sh_pla_point3[2]-Sh_pla_point2[2]])

Sh_pla_normal_x=Sh_pla_vec1[1]*Sh_pla_vec2[2]-Sh_pla_vec2[1]*Sh_pla_vec1[2]
Sh_pla_normal_y=-(Sh_pla_vec1[0]*Sh_pla_vec2[2]-Sh_pla_vec2[0]*Sh_pla_vec1[2])
Sh_pla_normal_z=Sh_pla_vec1[0]*Sh_pla_vec2[1]-Sh_pla_vec2[0]*Sh_pla_vec1[1]
#Sh_pla_normal=Sh_pla # the value of the normal of the shear plane is equal to shear plan
Sh_pla_normal_gcd=gcd(gcd(Sh_pla_normal_x,Sh_pla_normal_y),Sh_pla_normal_z)

P0=np.array([Sh_pla_normal_x/Sh_pla_normal_gcd, Sh_pla_normal_y/Sh_pla_normal_gcd, Sh_pla_normal_z/Sh_pla_normal_gcd])#P0 is the normal of the shear plane,divide the greatest common dividor
Q0=np.array([1, 0, 0]) #after rotation. Q0 is the x axis. Since the x axis will not be optimized as we set in VASP, so Q0 is [1, 0, 0]

Sh_dir=[int(x) for x in input("What is the shear direction? e.g. 1 1 -2 \n").split()]

if len(Sh_dir)==3:
    Sh_dir=Sh_dir
elif len(Sh_pla)==4:
    U=Sh_dir[1]+2*Sh_dir[0]
    V=Sh_dir[0]+2*Sh_dir[1]
    W=Sh_dir[3]
    UVW_gcd=gcd(gcd(U,V),W)
    Sh_dir=np.array([U/UVW_gcd, V/UVW_gcd, W/UVW_gcd])

L0=Sh_dir
K0=np.array([0, 0, 1]) #after rotation. K0 is the z axis. Since the x axis will not be optimized as we set in VASP, so Q0 is [1, 0, 0]


#P0=np.array([1, 1, 1]) #provide the axis before rotation.. P0 is the tensile direction we want to calculate.
###finish to read the input file
#Q0=np.array([1, 0, 0]) #after rotation. Q0 is the x axis. Since the x axis will not be optimized as we set in VASP, so Q0 is [1, 0, 0]

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

pos_array=np.array(t1).reshape((3,3)) ####read basis vector from POSCAR 
print(pos_array)
oa=np.array([pos_array[0][0], pos_array[0][1], pos_array[0][2]])
ob=np.array([pos_array[1][0], pos_array[1][1], pos_array[1][2]])
oc=np.array([pos_array[2][0], pos_array[2][1], pos_array[2][2]])
#------------------------------------------------------------------------------------
###start to calculate the structure data before rotation######
##a1_len=(float(pos_array[0][0])**2+float(pos_array[0][1])**2+float(pos_array[0][2])**2)**0.5
##a2_len=(float(pos_array[1][0])**2+float(pos_array[1][1])**2+float(pos_array[1][2])**2)**0.5
##a3_len=(float(pos_array[2][0])**2+float(pos_array[2][1])**2+float(pos_array[2][2])**2)**0.5
##
##P=np.array([P0[0]*a1_len,P0[1]*a2_len,P0[2]*a3_len])
##Q=np.array([Q0[0]*a1_len,Q0[1]*a2_len,Q0[2]*a3_len])
##
##XYZ_a=np.array([[a1_len, 0, 0],   
##                [0, a2_len, 0],
##                [0, 0, a3_len]])

#----------------------------------------

basis=np.array([oa, ob, oc])
P0_abc=np.array([[P0[0], 0, 0],   
                [0, P0[1], 0],
                [0, 0, P0[2]]])
P0_abc_basis=np.dot(P0_abc,basis)
P=P0_abc_basis[0]+P0_abc_basis[1]+P0_abc_basis[2] # the position of the normal of shear plane before rotation

Q0_abc=np.array([[Q0[0], 0, 0],   
                [0, Q0[1], 0],
                [0, 0, Q0[2]]])
Q0_abc_basis=np.dot(Q0_abc,basis)
Q=Q0_abc_basis[0]+Q0_abc_basis[1]+Q0_abc_basis[2] # the position of the normal of shear plane after rotation
#----------------------------------------
#basis=np.array([oa, ob, oc])
L0_abc=np.array([[L0[0], 0, 0],   
                [0, L0[1], 0],
                [0, 0, L0[2]]])
L0_abc_basis=np.dot(L0_abc,basis)
L=L0_abc_basis[0]+L0_abc_basis[1]+L0_abc_basis[2] # the position of shear direction before rotation

K0_abc=np.array([[K0[0], 0, 0],   
                [0, K0[1], 0],
                [0, 0, K0[2]]])
K0_abc_basis=np.dot(K0_abc,basis)
K=K0_abc_basis[0]+K0_abc_basis[1]+K0_abc_basis[2] # the position of shear direction after rotation
#------------------------------------------------------------------------------------
###start to calculate the structure data after rotation######
############## the first rotation begin##############
######1.get the angel between two vectors########
P_norm=(P[0]**2+P[1]**2+P[2]**2)**0.5 #the norm of vetor before roation
Q_norm=(Q[0]**2+Q[1]**2+Q[2]**2)**0.5 #the norm of vetor after roation
PQ_dot=np.dot(P,Q)
theta=math.acos(PQ_dot/(P_norm*Q_norm))##obtain theta, the unit is radian instead of degree
theta_cos=PQ_dot/(P_norm*Q_norm)
theta_sin=math.sin(theta)

######2.get the rotation axis##############
M=np.array([P[1]*Q[2]-P[2]*Q[1], P[2]*Q[0]-P[0]*Q[2], P[0]*Q[1]-P[1]*Q[0]])
M_norm=(M[0]**2+M[1]**2+M[2]**2)**0.5 #the norm of M vetor
M_unit=np.array([M[0]/M_norm, M[1]/M_norm, M[2]/M_norm])

######3.get the rotation matrix###########
Nx=M_unit[0]
Ny=M_unit[1]
Nz=M_unit[2]

PQ_rot=np.array([[Nx*Nx*(1-theta_cos)+theta_cos, Nx*Ny*(1-theta_cos)+Nz*theta_sin, Nx*Nz*(1-theta_cos)-Ny*theta_sin],
             [Nx*Ny*(1-theta_cos)-Nz*theta_sin, Ny*Ny*(1-theta_cos)+theta_cos, Ny*Nz*(1-theta_cos)+Nx*theta_sin],
             [Nx*Nz*(1-theta_cos)+Ny*theta_sin, Ny*Nz*(1-theta_cos)-Nx*theta_sin, Nz*Nz*(1-theta_cos)+theta_cos]])

###### #####
a_PQ_rot=np.dot(pos_array,PQ_rot) #three basis vector after rotation
a_PQ_rot=a_PQ_rot.tolist()
L_rot=np.dot(L,PQ_rot)

############## the second rotation begin##############
######1.get the angel between two vectors########
L_norm=(L_rot[0]**2+L_rot[1]**2+L_rot[2]**2)**0.5 #the norm of vetor before roation
K_norm=(K[0]**2+K[1]**2+K[2]**2)**0.5 #the norm of vetor after roation
LK_dot=np.dot(L_rot,K)
theta=math.acos(LK_dot/(L_norm*K_norm))##obtain theta, the unit is radian instead of degree
theta_cos=LK_dot/(L_norm*K_norm)
theta_sin=math.sin(theta)

######2.get the rotation axis##############
H=np.array([L_rot[1]*K[2]-L_rot[2]*K[1], L_rot[2]*K[0]-L_rot[0]*K[2], L_rot[0]*K[1]-L_rot[1]*K[0]])
H_norm=(H[0]**2+H[1]**2+H[2]**2)**0.5 #the norm of M vetor
H_unit=np.array([H[0]/H_norm, H[1]/H_norm, H[2]/H_norm])

######3.get the rotation matrix###########
Ex=H_unit[0]
Ey=H_unit[1]
Ez=H_unit[2]

LK_rot=np.array([[Ex*Ex*(1-theta_cos)+theta_cos, Ex*Ey*(1-theta_cos)+Ez*theta_sin, Ex*Ez*(1-theta_cos)-Ey*theta_sin],
             [Ex*Ey*(1-theta_cos)-Ez*theta_sin, Ey*Ey*(1-theta_cos)+theta_cos, Ey*Ez*(1-theta_cos)+Ex*theta_sin],
             [Ex*Ez*(1-theta_cos)+Ey*theta_sin, Ey*Nz*(1-theta_cos)-Ex*theta_sin, Ez*Ez*(1-theta_cos)+theta_cos]])

###### #####
a_LK_rot=np.dot(a_PQ_rot,LK_rot) #three basis vector after rotation 
a_LK_rot=a_LK_rot.tolist()


os.system("cp relax.in relax.in_original") #copy original POSCAR

r=[]
with open ("relax.in") as poscar1:
    for line in poscar1:
        r.append(line)
f=open("relax_rota.in","w")   #write POSCAR after rotation as POSCAR_rota
for i in range(0,index+1):
    f.write(r[i])
for j in range(len(a_LK_rot)):
    f.write(str(a_LK_rot[j][0])+' ')
    f.write(str(a_LK_rot[j][1])+' ')
    f.write(str(a_LK_rot[j][2]))
    f.write('\n')
for x in range(index+4,len(r)):
    f.write(r[x])
f.close()
os.system("cp relax_rota.in relax.in")  ## copy POSCAR_rota as POSCAR

###### Thirdly,apply tensile strain to the rotated basis vector
def postrain(poscar): 
    t2=[]
    t3=[]
    with open (poscar) as poscar2:
        pos2=poscar2.readlines()
        length=len(pos2)
        for k in range(length):
            if "CELL_PARAMETERS" in pos0[k]:
                index=k
        for i in range(index+1,index+4):
            j=pos2[i]
            j=j.split()
            t2.extend(j)
    for i in range(len(t2)):
        t3.extend([float(t2[i])])    
    pos_array3=np.array(t3).reshape((3,3))
    if (Q0==np.array([1, 0, 0])).all():
        stra_matr=np.array([[1,0,stra],
                            [0,1,0],
                            [stra,0,1]]) # strain(xz) and strain(zx)
        a_stra=np.dot(pos_array3,stra_matr)
    with open (poscar) as poscar:
        s=[]
        for line in poscar:
            s.append(line)
    f=open("relax_stra.in","w")
    for i in range(0,index+1):
        f.write(s[i])
    for j in range(len(a_stra)):
        f.write(str(a_stra[j][0])+' ')
        f.write(str(a_stra[j][1])+' ')
        f.write(str(a_stra[j][2]))
        f.write('\n')
    for x in range(index+4,len(s)):
        f.write(s[x])
    f.close()

###### Fourthly, performing QE calculation
for i in np.arange(stra,stra*(times+1),stra):
    print("****************************")
    i=round(i,2)
    print(i)
    postrain("relax.in")
    os.system("cp relax_stra.in relax.in")
    os.system("cp relax_stra.in relax.in_"+str(i))####
    #os.system("bsub < Job_qsh.sh")
    #os.system("srun -n 20 /scratch/jin.zhang3_397857/Software/vasp.5.4.4/bin/vasp_std > vasp.out")
    os.system("srun -n 20 /home/jin.zhang3/Software/qe-6.3_strain/qe-6.3_strain/bin/pw.x -nk 4 < relax.in > relax.out")
    chek=os.popen("grep DONE relax.out").read()
    while "DONE" not in chek:
        chek=os.popen("grep DONE OUTCAR").read()
        if "DONE" in chek:
            break
    os.system("cp relax.out relax.out_"+str(i))####
    #os.system("cp OUTCAR OUTCAR_"+str(i))####
    #os.system("cp CONTCAR CONTCAR_"+str(i))####
    #os.system("grep 'in kB' OUTCAR > kB")
    #fin=os.popen("tail -1 kB").read()
    #fin=fin.split()
    os.system("awk '/kbar/{getline; print}' relax.out | tail -1 > kB")
    fin=os.popen("awk '{print $6}' kB").read()
    #os.system("cp CONTCAR POSCAR")
    #os.system("bsub < Job_qsh.sh")
    #os.system("srun -n 20 ~/Software/VASP/VASP5.4.4/vasp.5.4.4/bin/vasp_std > vasp.out")
    if (Q0==np.array([1, 0, 0])).all():
        stress=(float(fin)/10)*(-1)
    strain=((1.0+float(stra))**(float(i)/float(stra))-1.0)*2  ##output strain
    u=open("strain_shearStress.dat","a")
    u.write(str(strain)+' ')
    u.write(str(stress))
    u.write('\n')
    t4=[]
    t5=[]
    with open ("relax.out") as out:
        out=out.readlines()
        length=len(out)
        for i in range(length):
            if "Begin final coordinates" in out[i]:
                index=i
        for j in range(index+5,index+8):
            f=out[j]
            f=f.split()
            t4.extend(f)
        for k in range(len(t4)):
            t5.extend([float(t4[k])])
    pos_new=np.array(t5).reshape((3,3))
    with open ("relax.in") as poscar0:
        pos0=poscar0.readlines()
        length=len(pos0)
        for i in range(length):
            if "CELL_PARAMETERS" in pos0[i]:
                index=i
    with open ("relax.in") as poscar0:
        s2=[]
        for line in poscar0:
            s2.append(line)
    f=open("relax_stra.in","w")
    for i in range(0,index+1):
        f.write(s2[i])
    for j in range(len(pos_new)):
        f.write(str(pos_new[j][0])+' ')
        f.write(str(pos_new[j][1])+' ')
        f.write(str(pos_new[j][2]))
        f.write('\n')
    for x in range(index+4,len(s2)):
        f.write(s2[x])
    f.close()
    os.system("cp relax_stra.in relax.in")
    #print(i)
u.close()
os.system("rm relax_stra.in")


