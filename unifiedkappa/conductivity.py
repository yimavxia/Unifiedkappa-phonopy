import numpy as np
import math
import cmath
import os
import sys

global debug
debug = True

class class_kappa:
    """
    input: obj_poscar
    """
    def __init__(self, obj_poscar):
        self.struct =  obj_poscar

    @staticmethod
    def get_mesh_rates(path_BTEw, path_BTEqfl):
        """
        usage: extract phonon scattering rates from ShengBTE calculations
        """
        with open(path_BTEw, "r") as file:
            wlines = file.readlines()
        with open(path_BTEqfl, "r") as file:
            qlines = file.readlines()
        qlines = [ int(qline.split()[1]) for qline in qlines ]
        num_qir = len(set(qlines))
        print (f"Number of qpoints in IBZ: {num_qir}")
        # postprocess scattering rates
        rates_qir = []
        num_band = int(len(wlines)/num_qir)
        for i in range(num_qir):
            rates_tmp = []
            for j in range(num_band):
                rates_tmp.append(float(wlines[i+num_qir*j].split()[-1]))
            rates_qir.append(rates_tmp)
            # expand rates from IBZ to full IBZ
        rates_qfl = []
        for qline in qlines:
            rates_qfl.append(rates_qir[qline-1])
        # rates_mesh
        return rates_qfl


    def get_kappa_phonopy(self,
                          mesh_in = [8,8,8],
                          sc_mat = np.eye(3)*2,
                          pm_mat = np.eye(3),
                          list_temp = [300.0],
                          name_pcell = "POSCAR-prim",
                          name_ifc2nd = "FORCE_CONSTANTS",
                          is_minikappa = True,
                          is_planckian = False,
                          is_sbtetau = True,
                          path_sbtetau = "./",
                          list_taufactor = [2.0],
                          override_error = True):
        try:
            import phonopy
            from phonopy import Phonopy
            from phonopy.structure.atoms import PhonopyAtoms
            from phonopy.interface.calculator import read_crystal_structure
        except:
            print ("Phonopy API version of 2.7.1 is required!")
            
        # load phonopy object
        phonon = phonopy.load(supercell_matrix = sc_mat,
                              primitive_matrix = pm_mat,
                              unitcell_filename = name_pcell,
                              is_symmetry = False,
                              force_constants_filename = name_ifc2nd)
        primcell = (phonon.get_primitive()).cell.T
        volpc = np.abs(np.dot(np.cross(primcell[1],primcell[2]),primcell[0]))
        print (f"Number of atoms in primitive cell: {len(phonon.primitive.masses)}")

        # get phonon properties on mesh
        phonon.run_mesh(mesh_in, #mesh_density,                                                                  
                        with_eigenvectors = False,
                        is_gamma_center = True,
                        with_group_velocities = True,
                        is_time_reversal = False,
                        is_mesh_symmetry = False)
        mesh_dict = phonon.get_mesh_dict()
        qpoints = mesh_dict['qpoints']
        print (qpoints)
        weights = mesh_dict['weights']
        freqs = mesh_dict['frequencies']
        eigs = mesh_dict['eigenvectors']
        gvfull = mesh_dict['group_velocities']
        
        # transform unit
        hbar = 1.054571726470000E-022
        kB = 1.380648813000000E-023
        pi = np.pi
        volpc = volpc/1000.0 # angstrom --> nm
        gvfull = gvfull/10.0 # angstrom/THz --> nm/THz
        freqs = freqs*2*pi   # THz --> 2*Pi*THz
        nband = len(freqs[0])
        freqcf = 0.1 #THz

        def shift_qpt(qpt_in):
            '''
            for qpt in qpoints:
            print (shift_qpt(qpt)) 
            '''
            qpt_out = []
            for xyz in qpt_in:
                if xyz < 0.0:
                    qpt_out.append(xyz+1.0)
                else:
                    qpt_out.append(xyz)
            return np.array(qpt_out)

        # compute thermal conductivity
        for temp in list_temp:
            if not is_minikappa:
                list_taufactor = [1.0]
            for factor in list_taufactor:
                kappaband=np.zeros((nband,nband,3,3), dtype=np.complex128, order='C')
                nqpt=len(qpoints)
                nband=len(freqs[0])
                # assign phonon lifetime
                gamma_mode = np.ones_like(freqs) * 0.0
                if is_minikappa:
                    gamma_mode = np.ones_like(freqs) * 1E10
                    for i in range(nband):
                        for iq in range(nqpt):
                            if freqs[iq,i] > freqcf:
                                if not is_planckian:
                                    gamma_mode[iq,i] = freqs[iq,i]/2/pi*factor
                                else:
                                    #1/(6.582119*10^(-16)/(tmpp1*8.61733*10^-5))/10^12
                                    gamma_mode[iq,i] = 0.13092 * temp
                if is_sbtetau:
                    from sbte_io import class_shengbte
                    obj_sbte = class_shengbte(path_dir = path_sbtetau,
                                              is_kappa = False,
                                              is_phase = False,
                                              is_rates3rd = True,
                                              override_error = override_error)
                    for i in range(nband):
                        for iq in range(nqpt):
                            if freqs[iq,i] > freqcf:
                                rate_tmp = obj_sbte.values['rates3rd_fl'][str(temp)][iq,i]
                                if rate_tmp > 0:
                                    gamma_mode[iq,i] = rate_tmp
                                else:
                                    gamma_mode[iq,i] = 1E10
                # proceed to compute kappa
                for i in range(nband):
                    for j in range(nband):
                        for iq in range(nqpt):
                            for k in range(3):
                                for kp in range(3):
                                    omega1=freqs[iq,i]
                                    omega2=freqs[iq,j]
                                    if omega1>freqcf and omega2>freqcf:
                                        if False:
                                            if (freqs[iq,i]/2/pi) > 0:
                                                Gamma1=freqs[iq,i]/2/pi*factor
                                            else:
                                                Gamma1=1E10
                                            if (freqs[iq,j]/2/pi) > 0:
                                                Gamma2=freqs[iq,j]/2/pi*factor
                                            else:
                                                Gamma2=1E10
                                        else:
                                            Gamma1 = gamma_mode[iq,i]
                                            Gamma2 = gamma_mode[iq,j]
                                        fBE1=1.0/(np.exp(hbar*omega1/kB/temp)-1.0)
                                        fBE2=1.0/(np.exp(hbar*omega2/kB/temp)-1.0)
                                        tmpv=(gvfull[iq,i,j,k]*gvfull[iq,j,i,kp]).real
                                        kappaband[i,j,k,kp]=kappaband[i,j,k,kp]+(omega1+omega2)/2* \
                                            (fBE1*(fBE1+1)*omega1+fBE2*(fBE2+1)*omega2)*tmpv \
                                            /(4*(omega1-omega2)**2+(Gamma1+Gamma2)**2)*\
                                            (Gamma1+Gamma2)
                # convert to thermal conductivity
                if is_minikappa:
                    print (f"Factor: {factor}"+"-"*10)
                kappaband=kappaband*1E21*hbar**2/(kB*temp*temp*volpc*nqpt)
                kappaD  = np.zeros((3,3), dtype=np.complex128, order='C')
                kappaOD = np.zeros((3,3), dtype=np.complex128, order='C')
                kappaF  = np.zeros((3,3), dtype=np.complex128, order='C')
                for i in range(nband):
                    for j in range(nband):
                        kappaF=kappaF+kappaband[i,j]
                        if i==j:
                            kappaD=kappaD+kappaband[i,j]
                        else:
                            kappaOD=kappaOD+kappaband[i,j]
                if is_minikappa:
                    f = open("minikappa"+"-"+str(temp)+"-"+str(factor)+".dat", "w")
                    #print (kappaF.real.reshape((1,9)))
                    f.write("   ".join(map(str, np.round(kappaD.real.reshape((9,1)).flatten(),decimals=8) ))+"\n")
                    f.write("   ".join(map(str, np.round(kappaOD.real.reshape((9,1)).flatten(),decimals=8) ))+"\n")
                    f.write("   ".join(map(str, np.round(kappaF.real.reshape((9,1)).flatten(),decimals=8) ))+"\n")
                    f.close()
                if True:
                    print ("Diagonal part of thermal conductivity: ")
                    print (kappaD.real[0,0])
                    print (kappaD.real)
                    print ("Off-diagonal part of thermal conductivity: ")
                    print (kappaOD.real[0,0])
                    print (kappaOD.real)
                    print ("Full thermal conductivity: ")
                    print (kappaF.real[0,0])
                    print (kappaF.real)
        # return kappa
        #return [kappaD.real[0,0], kappaOD.real[0,0]]
        return [kappaD.real, kappaOD.real]
    


    def get_minikappa_phonopy(self,
                          mesh_in = [8,8,8],
                          sc_mat = np.eye(3)*2,
                          pm_mat = np.eye(3),
                          list_temp = [300.0],
                          name_pcell = "POSCAR-prim",
                          name_ifc2nd = "FORCE_CONSTANTS",
                          list_taufactor = [2.0]):
        try:
            import phonopy
            from phonopy import Phonopy
            from phonopy.structure.atoms import PhonopyAtoms
            from phonopy.interface.calculator import read_crystal_structure
        except:
            print ("Phonopy API version of 2.7.1 is required!")
            
        # load phonopy object
        phonon = phonopy.load(supercell_matrix = sc_mat,
                              primitive_matrix = pm_mat,
                              unitcell_filename = name_pcell,
                              is_symmetry = False,
                              force_constants_filename = name_ifc2nd)
        primcell = (phonon.get_primitive()).cell.T
        volpc = np.abs(np.dot(np.cross(primcell[1],primcell[2]),primcell[0]))
        print (f"Number of atoms in primitive cell: {len(phonon.primitive.masses)}")

        # get phonon properties on mesh
        phonon.run_mesh(mesh_in, #mesh_density,                                                                  
                        with_eigenvectors = False,
                        is_gamma_center = True,
                        with_group_velocities = True,
                        is_time_reversal = False,
                        is_mesh_symmetry = False)
        mesh_dict = phonon.get_mesh_dict()
        qpoints = mesh_dict['qpoints']
        weights = mesh_dict['weights']
        freqs = mesh_dict['frequencies']
        eigs = mesh_dict['eigenvectors']
        gvfull = mesh_dict['group_velocities']
        
        # transform unit
        hbar = 1.054571726470000E-022
        kB = 1.380648813000000E-023
        pi = np.pi
        volpc = volpc/1000.0 # angstrom --> nm
        gvfull = gvfull/10.0 # angstrom/THz --> nm/THz
        freqs = freqs*2*pi   # THz --> 2*Pi*THz
        nband = len(freqs[0])
        freqcf = 0.1 #THz

        # compute thermal conductivity
        for temp in list_temp:
            for factor in list_taufactor:
                kappaband=np.zeros((nband,nband,3,3), dtype=np.complex128, order='C')
                nqpt=len(qpoints)
                nband=len(freqs[0])
                for i in range(nband):
                    for j in range(nband):
                        for iq in range(nqpt):
                            for k in range(3):
                                for kp in range(3):
                                    omega1=freqs[iq,i]
                                    omega2=freqs[iq,j]
                                    if omega1>freqcf and omega2>freqcf:
                                        if (freqs[iq,i]/2/pi) > 0:
                                            Gamma1=freqs[iq,i]/2/pi*factor
                                        else:
                                            Gamma1=1E10
                                        if (freqs[iq,j]/2/pi) > 0:
                                            Gamma2=freqs[iq,j]/2/pi*factor
                                        else:
                                            Gamma2=1E10
                                        fBE1=1.0/(np.exp(hbar*omega1/kB/temp)-1.0)
                                        fBE2=1.0/(np.exp(hbar*omega2/kB/temp)-1.0)
                                        tmpv=(gvfull[iq,i,j,k]*gvfull[iq,j,i,kp]).real
                                        kappaband[i,j,k,kp]=kappaband[i,j,k,kp]+(omega1+omega2)/2* \
                                            (fBE1*(fBE1+1)*omega1+fBE2*(fBE2+1)*omega2)*tmpv \
                                            /(4*(omega1-omega2)**2+(Gamma1+Gamma2)**2)*\
                                            (Gamma1+Gamma2)
                # convert to thermal conductivity
                print (f"Factor: {factor}"+"-"*10)
                kappaband=kappaband*1E21*hbar**2/(kB*temp*temp*volpc*nqpt)
                kappaD  = np.zeros((3,3), dtype=np.complex128, order='C')
                kappaOD = np.zeros((3,3), dtype=np.complex128, order='C')
                kappaF  = np.zeros((3,3), dtype=np.complex128, order='C')
                for i in range(nband):
                    for j in range(nband):
                        kappaF=kappaF+kappaband[i,j]
                        if i==j:
                            kappaD=kappaD+kappaband[i,j]
                        else:
                            kappaOD=kappaOD+kappaband[i,j]
                f = open("minikappa"+"-"+str(temp)+"-"+str(factor)+".dat", "w")
                #print (kappaF.real.reshape((1,9)))
                f.write("   ".join(map(str, np.round(kappaD.real.reshape((9,1)).flatten(),decimals=8) ))+"\n")
                f.write("   ".join(map(str, np.round(kappaOD.real.reshape((9,1)).flatten(),decimals=8) ))+"\n")
                f.write("   ".join(map(str, np.round(kappaF.real.reshape((9,1)).flatten(),decimals=8) ))+"\n")
                f.close()
                #os.system("touch minikappa")
                if True:
                    print ("Diagonal part of thermal conductivity: ")
                    print (kappaD.real[0,0])
                    print ("Off-diagonal part of thermal conductivity: ")
                    print (kappaOD.real[0,0])
                    print ("Full thermal conductivity: ")
                    print (kappaF.real[0,0])
        #return [kappaD.real[0,0], kappaOD.real[0,0]]

        
    def print_gvfull(self, mesh_phonon):
        freqs = []
        gvfull = []
        weights = []
        for iterm in mesh_phonon:
            freqs.append(iterm[0])
            gvfull.append(iterm[1])
            weights.append(iterm[3])
        freqs = np.array(freqs)
        gvfull = np.array(gvfull)
        gvfull = gvfull/1000.0
        nqpt=len(freqs)
        nband = len(freqs[0])
        gv_out = []
        gv_xx_out = []
        gv_yy_out = []
        gv_zz_out = []
        for iq in range(nqpt):
            print (f"Loop over {iq}th qpoint")
            for i in range(nband):
                for j in range(i):
                    freq = (freqs[iq,i]+freqs[iq,j])/2
                    gv_xx = (gvfull[iq,i,j,0]*gvfull[iq,j,i,0]).real
                    gv_yy = (gvfull[iq,i,j,1]*gvfull[iq,j,i,1]).real
                    gv_zz = (gvfull[iq,i,j,2]*gvfull[iq,j,i,2]).real
                    if abs(gv_xx) > 10**-6 or abs(gv_yy) > 10**-6 or abs(gv_zz) > 10**-6:
                        gv_out.append([freq, gv_xx, gv_yy, gv_zz])
                    if abs(gv_xx) > 10**-6:
                        gv_xx_out.append([freq, gv_xx])
                    if abs(gv_yy) > 10**-6:
                        gv_yy_out.append([freq, gv_yy])
                    if abs(gv_zz) > 10**-6:
                        gv_zz_out.append([freq, gv_zz])
            #print (gvfull[iq,i,j,k])
            #print (gvfull[iq])
        np.savetxt('gv_xyz.txt', np.array(gv_out))
        np.savetxt('gv_x.txt', np.array(gv_xx_out))
        np.savetxt('gv_y.txt', np.array(gv_yy_out))
        np.savetxt('gv_z.txt', np.array(gv_zz_out))
            
    
    def get_kappa(self, mesh_phonon, mesh_rates, temperature, minikappa = True):
        """
        usage: compuate lattice thermal conductivity including coherent part
        """
        rates=np.array(mesh_rates)
        freqs = []
        gvfull = []
        weights = []
        for iterm in mesh_phonon:
            freqs.append(iterm[0])
            gvfull.append(iterm[1])
            weights.append(iterm[3])
        freqs = np.array(freqs)
        gvfull = np.array(gvfull)
        #------------------------------------
        # /ocean/projects/mat220008p/yidaniu/northwestern/bridge2-yixia/manuscripts/TlVSe
        hbar = 1.054571726470000E-022 #J/THz
        kB = 1.380648813000000E-023 #J/K
        volpc = self.struct.primcell['vol']/1000.0 # A^3 --> nm^3
        gvfull = gvfull/1000.0 # m/s --> km/s or nm*THz
        freqs = freqs*2*math.pi # THz --> 2*Pi*THz
        freqcf = 0.05 # THz
        temp_list = [temperature] # K
        nband = len(freqs[0])

        for temp in temp_list:
            for factor in [2]:
                print ("Temperature: "+str(temp))
                print ("Lifetime factor: "+str(factor))
                kappaband=np.zeros((nband,nband,3,3), dtype=np.complex128, order='C')
                nqpt=len(freqs)
                nband=len(freqs[0])
                for i in range(nband):
                    for j in range(nband):
                        for iq in range(nqpt):                            
                            for k in range(3):
                                for kp in range(3):
                                    omega1=freqs[iq,i]
                                    omega2=freqs[iq,j]
                                    if omega1>freqcf and omega2>freqcf:
                                        if minikappa:
                                            if (freqs[iq,i]/2/math.pi) > 0:
                                                Gamma1=freqs[iq,i]/2/math.pi*factor
                                            else:
                                                Gamma1=1E10
                                            if (freqs[iq,j]/2/math.pi) > 0:                    
                                                Gamma2=freqs[iq,j]/2/math.pi*factor
                                            else:
                                                Gamma2=1E10
                                        else:
                                            if (freqs[iq,i]/2/math.pi) > 0:
                                                Gamma1=rates[iq,i]
                                            else:
                                                Gamma1=1E10
                                            if (freqs[iq,j]/2/math.pi) > 0:
                                                Gamma2=rates[iq,j]
                                            else:
                                                Gamma2=1E10
                                        fBE1=1.0/(np.exp(hbar*omega1/kB/temp)-1.0)
                                        fBE2=1.0/(np.exp(hbar*omega2/kB/temp)-1.0)
                                        tmpv=(gvfull[iq,i,j,k]*gvfull[iq,j,i,kp]).real          #eq=1
                                        #tmpv=(gvfull[iq,i,j,k]).real*(gvfull[iq,j,i,kp]).real
                                        #tmpv=(gvfull[iq,i,j,k]*gvfull[iq,i,j,kp].conj()).real   #eq=1
                                        kappaband[i,j,k,kp]=kappaband[i,j,k,kp]+(omega1+omega2)/2* \
                                            (fBE1*(fBE1+1)*omega1+fBE2*(fBE2+1)*omega2)*tmpv \
                                            /(4*(omega1-omega2)**2+(Gamma1+Gamma2)**2)*(Gamma1+Gamma2)*\
                                            weights[iq] # put on weight
                #factor
                kappaband=kappaband*1E21*hbar**2/(kB*temp*temp*volpc*sum(weights)) # put on weight
                kappaD  = np.zeros((3,3), dtype=np.complex128, order='C')
                kappaOD = np.zeros((3,3), dtype=np.complex128, order='C')
                kappaF  = np.zeros((3,3), dtype=np.complex128, order='C') 
                for i in range(nband):
                    for j in range(nband):
                        kappaF=kappaF+kappaband[i,j] 
                        if i==j:
                            kappaD=kappaD+kappaband[i,j]
                        else:
                            kappaOD=kappaOD+kappaband[i,j]
                f = open("kappa_tensor"+"-"+str(temp)+"-"+str(factor)+".dat", "w")
                self.kappaD =  kappaD.real.reshape((1,9))
                self.kappaOD = kappaOD.real.reshape((1,9))
                print (kappaF.real.reshape((1,9)))
                f.write("   ".join(map(str, np.round(kappaD.real.reshape((9,1)).flatten(),decimals=8) ))+"\n")
                f.write("   ".join(map(str, np.round(kappaOD.real.reshape((9,1)).flatten(),decimals=8) ))+"\n")
                f.write("   ".join(map(str, np.round(kappaF.real.reshape((9,1)).flatten(),decimals=8) ))+"\n")
                f.close()
                os.system("touch minikappa")   
                if True:
                    print ("Diagonal part of thermal conductivity: ")
                    print (kappaD.real)
                    print ("Off-diagonal part of thermal conductivity: ")
                    print (kappaOD.real)
                    print ("Full thermal conductivity: ")
                    print (kappaF.real)

        
                
