import copy
import os
import numpy as np
from itertools import permutations
import random
import itertools
import math
import sys

global debug

debug =  False

class class_poscar:

    """
    To do:
    1. Enable autoupdate of basic poscar properties

    Properties:
    To read, modify, and write VASP POSCAR
    1. Assume POSCAR format following VASP5.x
    2. Assume POTCAR format following POTCAR-X
    3. Manybody cluster analysis
    """

    def __init__(self, pathfile, vasp4spe = None):
        #read file from pathfile --> pos_path
        self.pos_path = "/".join(pathfile.split("/")[0:-1])+"/"
        with open(pathfile,"r") as file:
            self.input=file.readlines()
        if vasp4spe != None:
            self.input.insert(5, vasp4spe+"\n")
        self.input = list(filter(None, self.input))
        self.input_org=copy.deepcopy(self.input)
        #extract basic poscar information
        self.spe = self.input[5].split()
        self.cspe = list(map(int,self.input[6].split()))
        # unsorted formula
        self.formula=""
        for i in range(len(self.spe)):
            self.formula+=self.spe[i]+str(self.cspe[i])
        # sorted formula
        self.formula_order=""
        sorted_indices = [index for index, value in sorted(enumerate(self.spe), key=lambda x: x[1])]
        for idx_tmp in sorted_indices:
            self.formula_order+=self.spe[idx_tmp]+str(self.cspe[idx_tmp])
        self.spe_all = []
        for i in range(len(self.spe)):
            self.spe_all += [self.spe[i]]*self.cspe[i]
        from atominfo import class_atom
        self.mass = [ class_atom.amass[el] for el in self.spe_all]
        self.natom = sum(self.cspe)
        self.cord = self.input[7][0].upper()
        self.abc = "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z".split()
        self.spe_all = []
        self.abc_all = []
        for i in range(len(self.spe)):
            self.spe_all+=[self.spe[i]]*self.cspe[i]
            self.abc_all+=[self.abc[i]]*self.cspe[i] 
        self.name = self.input[0]
        self.scale = float(self.input[1])
        self.latvec = []
        for i in range(2,5):
            self.latvec.append(list(map(float,self.input[i].split()[0:3])))
        [self.vol, self.rlatvec] = class_poscar.recip_vol(self.latvec)            
        self.atompos = []
        for i in range(8,8+self.natom):
            self.atompos.append(list(map(float,self.input[i].split()[0:3])))
        #output poscar --> full copy of the input file --> most operations on self.output
        self.output={}
        self.output['name'] = self.name
        self.output['scale'] = self.scale
        self.output['latvec'] = self.latvec
        self.output['spe'] = self.spe
        self.output['cspe'] = self.cspe
        self.output['spe_all'] = []
        self.output['spe_type'] = []
        for idx, el in enumerate(self.output['spe']):
            self.output['spe_all'] += [ el ] * self.output['cspe'][idx]
        for idx, el in enumerate(self.output['cspe']):
            spe_type_tmp = []
            for i in range(self.output['cspe'][idx]):
                spe_type_tmp.append(idx)
            self.output['spe_type']+=spe_type_tmp
        self.output['natom'] = self.natom
        self.output['cord'] = self.cord
        self.output['atompos'] = self.atompos[0:self.natom]
        [self.output['vol'], self.output['rlatvec']] = class_poscar.recip_vol(self.output['latvec'])
        self.primcell = copy.deepcopy(self.output)
        

    #------------------------------------------------------------------
    def get_qmesh(self, density = 2.5*24):
        ngridx = int (density / np.linalg.norm(self.latvec[0]))
        ngridy = int (density / np.linalg.norm(self.latvec[1]))
        ngridz = int (density / np.linalg.norm(self.latvec[2]))
        return [ngridx, ngridy, ngridz]
    
    
    def dump_cfg(self, pathout = "./", filename = 'mtp.cfg'):
        class_poscar.dir_to_car(self, reverse=False)
        str_mtp = "BEGIN_CFG" + "\n"
        str_mtp += "Size" + "\n"        
        str_mtp += f"{self.natom}" + "\n"
        str_mtp += "Supercell" + "\n"
        # lattice vector
        for i in range(3):
            str_mtp += "  ".join(map(str, self.latvec[i])) + "\n"
        # atoms
        str_mtp += "AtomData:  id type       cartes_x      cartes_y      cartes_z           fx          fy          fz" + "\n"
        for i in range(self.natom):
            str_mtp += f"{i+1}   {self.output['spe_type'][i]}   " + "    ".join(map(str, self.output['atompos'][i])) + \
                "   0   0   0" + "\n"
        # energy
        str_mtp += "Energy" + "\n"
        str_mtp += "0.0" + "\n"
        str_mtp += "PlusStress:  xx          yy          zz          yz          xz          xy" + "\n"
        str_mtp += "0 0 0 0 0 0" + "\n"
        str_mtp += "Feature   EFS_by       VASP" + "\n"
        str_mtp += "END_CFG" + "\n"
        
        
        with open(os.path.join(pathout, filename), 'w') as file:
            file.write(str_mtp)

    
    def dump_xyz(self, pathout = "./", pathname = 'model.xyz'):
        str_lat_vec = ""
        for i in range(3):
            for j in range(3):
                str_lat_vec += str(np.round(self.latvec[i][j],12))+" "
        self.dir_to_car()
        str_out = ""
        str_out += str(self.natom) + "\n"
        str_out += "pbc=\"T T T\" "
        str_out += f"lattice=\"{str_lat_vec[:-1]}\" "
        str_out += "properties=species:S:1:pos:R:3" + "\n"
        for i in range(self.natom):
            pos_list = self.output['atompos'][i]
            pos_arry = np.round(np.array(pos_list), 12).tolist()
            pos_str = " ".join(map(str, pos_arry))
            str_out += self.output['spe_all'][i] + " " + pos_str + "\n"
        if False:
            print (str_out)
        with open(pathout+pathname, "w") as file:
            file.write(str_out)

    
    def dump_control(self, pathout = "./", pathname = 'CONTROL',
                     dict_info={
                         "onlyharmonic": "FALSE",
                         "ngrid": [4, 4, 4],
                         "scell": [1, 1, 1]
                     }):
        control={}
        #allocations
        control['nelements'] = len(self.spe)
        control['natoms'] = self.natom
        if "ngrid" not in dict_info.keys():
            control['ngrid'] = [11,11,11]
        else:
            control['ngrid'] = dict_info['ngrid']            
        control['norientations']=0
        #crystal
        latconst = 10.0
        control['lfactor'] = latconst*self.scale/10.0
        control['lat1'] = np.array(self.latvec[0])/latconst
        control['lat2'] = np.array(self.latvec[1])/latconst
        control['lat3'] = np.array(self.latvec[2])/latconst
        tmp1 = []
        for i in range(len(self.spe)):
            tmp1 = tmp1+["\""+self.spe[i]+"\""]
        #
        control['elements'] = tmp1
        tmp1 = []
        for i in range(len(self.cspe)):
            j = i+1
            tmp1 = tmp1+[j]*self.cspe[i]
        #
        control['types'] = tmp1
        for i in range(self.natom):
            control['pos'+str(i+1)] = np.array(self.atompos[i])
        #
        if 'scell' not in dict_info.keys():
            control['scell'] = [4,4,4]
        else:
            control['scell'] = dict_info['scell']
        #parameters
        control['T'] = 300
        if "scalebroad" not in dict_info.keys():
            control['scalebroad'] = 0.1
        else:
            control['scalebroad'] = dict_info['scalebroad']
        #flags
        if "onlyharmonic" not in dict_info.keys():
            control['onlyharmonic'] = "FALSE"
        else:
            control['onlyharmonic'] = dict_info["onlyharmonic"]
        if "convergence" not in  dict_info.keys():
            control['convergence'] = "FALSE"
        else:
            control['convergence'] = dict_info["convergence"]
        control['isotopes'] = "FALSE"
        control['nonanalytic'] = "FALSE"
        control['nanowires'] = "FALSE"
        #extra
        control['natom'] = self.natom
        ##write the file
        class_poscar.write_control(control, os.path.join(pathout, pathname))


    def write_control(control, file):
        fout=open(file, "w")        
        natoms=control["natoms"]
        space="        "
        #write allocations
        fout.write("&allocations\n")
        fout.write(space+"nelements="+str(control["nelements"])+",\n")
        fout.write(space+"natoms="+str(control["natoms"])+",\n")
        fout.write(space+"ngrid(:)="+" ".join(map(str, control["ngrid"]))+"\n")
        fout.write(space+"norientations="+str(control["norientations"])+"\n")
        fout.write("&end\n")
        #write crystal
        fout.write("&crystal\n")
        fout.write(space+"lfactor="+str(control["lfactor"])+"\n")
        for i in range(1,4):
            fout.write(space+"lattvec(:,"+str(i)+")="+\
                       " ".join(map(str, control["lat"+str(i)]))+",\n")
        fout.write(space+"elements="+", ".join(control["elements"])+",\n")
        fout.write(space+"types="+" ".join(map(str, control["types"]))+"\n")
        for i in range(1, natoms+1):
            fout.write(space+"positions(:,"+str(i)+")="+\
                       " ".join(map(str, control["pos"+str(i)]))+",\n")
        fout.write(space+"scell(:)="+" ".join(map(str, control["scell"]))+"\n")
        fout.write("&end\n")
        #write parameters
        fout.write("&parameters\n")
        fout.write(space+"T="+str(control["T"])+"\n")
        fout.write(space+"scalebroad="+str(control["scalebroad"])+"\n")
        fout.write("&end\n")        
        #write flags
        fout.write("&flags\n")
        fout.write(space+"onlyharmonic="+"."+str(control["onlyharmonic"])+"."+"\n")
        fout.write(space+"convergence="+"."+str(control["convergence"])+"."+"\n")
        fout.write(space+"isotopes="+"."+str(control["isotopes"])+"."+"\n")
        fout.write(space+"nonanalytic="+"."+str(control["nonanalytic"])+"."+"\n")
        fout.write(space+"nanowires="+"."+str(control["nanowires"])+"."+"\n")
        fout.write("&end\n")
        fout.close()


    def dump_latin(self, pathout = "./", scale = 10.0, return_str = False):
        strout = ""
        for i in range(3):
            latvec_tmp = list(np.array(self.latvec[i])/scale)
            strout += "  ".join(map(str, latvec_tmp))+"\n"
        strout += "1 0 0\n"
        strout += "0 1 0\n"
        strout += "0 0 1\n"
        for i in range(self.natom):
            strout += "  ".join(map(str, self.atompos[i])) + " " + self.spe_all[i] + "," + self.abc_all[i] +"\n"
        if not return_str:
            with open(os.path.join(pathout, "lat.in"), "w") as file:
                file.write(strout)
        else:
            return strout


    def print_info(self):
        import numpy as np
        print ("Atom species: "+" ".join(self.spe))
        print ("Atom species count: "+" ".join(map(str, self.cspe)))
        print ("Atom species all: "+" ".join(self.spe_all))
        print ("Numnber of total atoms: "+str(self.natom))
        print ("Cartesian or Direct: "+self.cord)        
        print ("Lattice vectors:")
        print (np.array(self.latvec))
        print ("Atomic positions:")
        print (np.array(self.atompos))
            

    def print_poscar_org(self):
        #print ("-------> Original POSCAR")
        for line in self.input:
            print (line, end='')


    def dir_to_car(self, reverse=False):
        if not reverse:
            if self.output['cord'] == "D":
                self.output['cord'] = "C"
                for i in range(self.output['natom']):                
                    self.output['atompos'][i] = np.matmul(
                        np.array(self.output['atompos'][i]),
                        np.array(self.output['latvec']).tolist()
                    )
        else:
            if self.output['cord'] == "C":
                self.output['cord'] = "D" 
                for i in range(self.output['natom']):
                    self.output['atompos'][i] = np.matmul(
                        np.array(self.output['atompos'][i]),
                        np.linalg.inv(np.array(self.output['latvec'])).tolist()
                    )
                    
#--------------------resitrict use-------------------
    def read_my_eng(self, path_file):
        '''
        only for ff-data
        '''
        with open(path_file, "r") as file:
            lines  = file.readlines()
        self.energy = float(lines[0].strip())

    def read_my_for(self, path_file):
        with open(path_file, "r") as file:
            lines = file.readlines()
        self.forces = []
        for line in lines:
            forces_tmp = list(map(float,line.split()))
            self.forces.append(forces_tmp)            
                    

    def get_disp_atom_random_from_scell(self,
                                        disp_range = [0.01, 0.05],
                                        nconfig = 4,
                                        ave_natom = 100,
                                        out_path = ""):
        # local random number
        random.seed(13579)
        # update self.output
        self.get_scell(ave_natom = ave_natom)
        for i in range(nconfig):
            pos_output = copy.deepcopy(self.scell)
            class_poscar.disp_atom_random(self, pos_output, disp_range = disp_range)
            if out_path == "":
                self.write_poscar_dp(pos_output, self.pos_path + "/POSCAR-scell-disp-"+str(i+1))
            else:
                self.write_poscar_dp(pos_output, out_path + "/POSCAR-scell-disp-"+str(i+1))            


    def get_disp_atom_random_from_cell(self,
                                       disp_range = [0.01, 0.05],
                                       nconfig = 4,
                                       out_path_name = ""):
        
        random.seed(13579)
        for i in range(nconfig):
            pos_output = copy.deepcopy(self.output)
            class_poscar.disp_atom_random(self, pos_output, disp_range = disp_range)
            if out_path_name == "":
                self.write_poscar_dp(pos_output, self.pos_path + "/POSCAR-scell-disp-"+str(i+1))
            else:
                self.write_poscar_dp(pos_output, f"{out_path_name}-{i}")
                

    def disp_atom_random(self, pos_output, disp_range=[0.1, 0.1]):
        """
        #define random number outside the loop to ensure true random
        #import random
        #random.seed(13579)
        #random displacements with plus/minus sign
        """
        disp_tmp = []
        low_end = disp_range[0]
        high_end = disp_range[1]
        rand_list = [-1, 0, 1]
        for i in range(pos_output['natom']):
            disp_tmp.append([np.random.uniform(low_end, high_end)*random.choice(rand_list),
                             np.random.uniform(low_end, high_end)*random.choice(rand_list),
                             np.random.uniform(low_end, high_end)*random.choice(rand_list),
                         ])
        #print (disp_tmp)
        self.add_disp(disp_tmp, pos_output)
        

    def add_disp(self, disp_tmp, pos_output):
        """
        usage: add displacements perturbations to atomic positions
        """
        if pos_output['cord'] == "C":
            pos_output['atompos'] = np.array(disp_tmp) + np.array(pos_output['atompos'])
        else:
            pos_output['atompos'] = np.matmul(
                np.array(disp_tmp),
                np.linalg.inv(np.array(pos_output['latvec']))
            ) + np.array(pos_output['atompos'])
        
        pos_output['atompos'] = pos_output['atompos'].tolist()
            

    def remove_vac(self):
        """
        usage: remove "X" vacancy atoms from structures generated by icet
        self.output={}
        self.output['name'] = self.name
        self.output['scale'] = self.scale
        self.output['latvec'] = self.latvec
        self.output['spe'] = []
        self.output['cspe'] = []
        """        
        #todo --> or just del the lists?
        self.output['spe'] = []
        self.output['cspe'] = []
        self.output['cord'] = []
        self.output['atompos'] = []
        for idx, item in enumerate(self.spe):
            if "X" not in self.spe:
                nvac = 0
            if item != "X":
                self.output['spe'].append(item)
                self.output['cspe'].append(self.cspe[idx])
            else:
                nvac = self.cspe[idx]
        self.output['cord'] = self.cord
        self.output['atompos'] = self.atompos[0:self.natom-nvac]
        self.output['natom'] = len(self.output['atompos'])
                

    def poscar_to_str(self):
        """
        usage: write POSCAR in VASP5.x format
        """
        strout = ""
        strout += self.output['name']
        strout += str(self.output['scale'])+"\n"
        for i in range(3):
            strout += " ".join(['{:16.12f}'.format(x) for x in self.output['latvec'][i]])+"\n"
        strout += " ".join(self.output['spe'])+"\n"
        strout += " ".join(map(str, self.output['cspe']))+"\n"
        strout += self.output['cord']+"\n"
        if sum(self.output['cspe']) == len(self.output['atompos']):
            for i in range(sum(self.output['cspe'])):
                strout += " ".join(['{:16.12f}'.format(x) for x in self.output['atompos'][i]])+"\n"
        else:
            raise Exception("Number of atoms != Number of positions")        
        return strout


    def poscar_to_str_dp(self, pos_output):
        """                                                                                                  
        Write POSCAR in VASP5.x format                                                                         
        Using deep copied poscar output from self.output
        """
        strout = ""
        strout += pos_output['name']
        strout += str(pos_output['scale'])+"\n"
        for i in range(3):
            strout += " ".join(['{:16.12f}'.format(x) for x in pos_output['latvec'][i]])+"\n"
        strout += " ".join(pos_output['spe'])+"\n"
        strout += " ".join(map(str, pos_output['cspe']))+"\n"
        strout += pos_output['cord']+"\n"
        if sum(pos_output['cspe']) == len(pos_output['atompos']):
            for i in range(sum(pos_output['cspe'])):
                strout += " ".join(['{:16.12f}'.format(x) for x in pos_output['atompos'][i]])+"\n"
        else:
            raise Exception("Number of atoms != Number of positions")
        return strout


    def write_poscar(self, file_out):
        strout = self.poscar_to_str()
        with open(file_out,"w") as file:
            file.write(strout
)

    def write_poscar_dp(self, pos_output, file_out):
        strout = self.poscar_to_str_dp(pos_output)
        with open(file_out,"w") as file:
            file.write(strout)

    
    def print_poscar(self):
        strout = self.poscar_to_str()
        print ("------>Printing out POSCAR")
        print (strout)

            
    def fix_potcar(self, pathin_potcar, pathout_potcar):
        str_cmd = "cat "
        for spe in self.output['spe']:
            str_cmd += pathin_potcar+"POTCAR-"+spe+"  "
        str_cmd += " > " + pathout_potcar
        os.system(str_cmd)


    def fix_incar(self, pathin_incar, pathout_incar):
        info_mujl={
            "Mn":[5.0, 3.9, 0.0, 2],
            "Ni":[5.0, 6.4, 0.0, 2],
            "Co":[5.0, 4.9, 0.0, 2],
        }
        with open(pathin_incar,'r') as file:
            incar_lines = file.readlines()
        for count, line in enumerate(incar_lines):
            if "MAGMOM" in line:
                magmom_str = "MAGMOM = "
                ldauu_str = "LDAUU = "
                ldauj_str = "LDAUJ = "
                ldaul_str = "LDAUL = "                
                for idx_spe, spe in enumerate(self.output['spe']):
                    if spe in info_mujl.keys():
                        magmom_str += str(self.output['cspe'][idx_spe])+"*"+'{:6.4f}'.format(info_mujl[spe][0]) + "   "
                        ldauu_str += '{:2.1f}'.format(info_mujl[spe][1]) + "   "
                        ldauj_str += '{:2.1f}'.format(info_mujl[spe][2]) + "   "
                        ldaul_str += str(int(info_mujl[spe][3])) + "   "
                    else:
                        magmom_str += str(self.output['cspe'][idx_spe])+"*"+'{:6.4f}'.format(0) + "   "
                        ldauu_str += '{:2.1f}'.format(0.0) + "   "
                        ldauj_str += '{:2.1f}'.format(0.0) + "   "
                        ldaul_str += "-1" + "   "
                #print (magmom_str)
                #print (ldauu_str)
                #print (ldauj_str)
                #print (ldaul_str)
                incar_lines[count]=magmom_str+"\n"
            if "LDAUU" in line:
                incar_lines[count]=ldauu_str+"\n"
            if "LDAUJ" in line:
                incar_lines[count]=ldauj_str+"\n"
            if "LDAUL" in line:
                incar_lines[count]=ldaul_str+"\n"

        with open(pathout_incar,'w') as file:
            for line in incar_lines:
                file.write(line)


    def write_cfg(self,file_out):
        pass

    
    def ini_cell(self):
        """
        usage: prepare cell structure for symmetry analysis based on output[dict]
        input: poscar
        output: cell spglib style
        """
        lattice =  self.output['latvec']
        positions = self.output['atompos']
        numbers = []
        for idx, item in enumerate(self.output['cspe']):
            for dup in range(item):
                numbers.append(idx+1)
        cell = (lattice, positions, numbers)
        return cell


    def ini_cell_prim(self, symprec = 1e-4, read_prim = ""):
        """
        usage: prepare primitive cell structure for symmetry analysis
        warning: same symmetry for primcell ans supercell
        """
        # warning: yixia --> not excute the get_prim function
        #self.get_prim(symprec, read_prim)
        lattice =  self.primcell['latvec']
        positions = self.primcell['atompos']
        numbers = []
        for idx, item in enumerate(self.primcell['cspe']):
            for dup in range(item):
                numbers.append(idx+1)
        cell = (lattice, positions, numbers)
        return cell

        
    @staticmethod
    def load_spg():
        try:
            import spglib as spg
        except ImportError:
            from pyspglib import spglib as spg


    def refine_cell(self, symprec = 1e-4, to_prim = False, posname="POSCAR-sym"):
        """
        usage: Standardize and symmetrize crystal 
        alternative:
        lattice, scaled_positions, numbers =
        spg.refine_cell(self.ini_cell(), symprec) 
        warning: only checked for direct coordinations
        """
        try:
            import spglib as spg
        except ImportError:
            from pyspglib import spglib as spg
        self.get_sym(symprec)
        lattice, scaled_positions, numbers = \
        spg.standardize_cell(self.ini_cell(),
                             to_primitive = to_prim,
                             no_idealize = False,
                             symprec =  symprec)
        #print (numbers)
        self.output['latvec'] = lattice.tolist()
        copy_output = copy.deepcopy(self.output)
        self.output['spe'] = []
        self.output['cspe'] = []
        spe_tmp = [ copy_output['spe'][i-1] for i in numbers ]
        #print (spe_tmp)
        for el in spe_tmp:
            if el not in self.output['spe']:
                self.output['spe'].append(el)
        for el in self.output['spe']:
            self.output['cspe'].append(spe_tmp.count(el))
        self.output['natom'] = len(spe_tmp)
        self.output['spe_all'] = []
        for i in range(len(self.output['cspe'])):
            self.output['spe_all'] += [self.output['spe'][i]]*self.output['cspe'][i]
        self.output['atompos'] = []
        for el in self.output['spe']:
            for idx, el_in in enumerate(spe_tmp):
                if el_in == el:
                    self.output['atompos'].append(scaled_positions[idx].tolist())
        if self.output['cord'] != "D":
            print ("------>Warning: assume direct coordination")
        self.write_poscar(self.pos_path+f"/{posname}")


    def loadspglib(func):
        '''
        usage: decorator for loading spglib
        '''
        import functools
        @functools.wraps(func)
        def wrapper_loadspglib(*args, **kwargs):
            try:
                import spglib as spg
            except ImportError:
                from pyspglib import spglib as spg
            value = func(*args, **kwargs)
            #print(f"spglib loaded!")
            return value
        return wrapper_loadspglib


    @staticmethod
    def recip_vol(latvec_in):
        '''
        usage: return volume and reciprocal latvec
        '''
        latvec = np.array(latvec_in)
        vol = np.dot(latvec[:,0], np.cross(latvec[:,1], latvec[:,2]))
        rlatvec = [ (2 * math.pi / vol * np.cross(latvec[:,1], latvec[:,2])).tolist(),\
                     (2 * math.pi / vol * np.cross(latvec[:,2], latvec[:,0])).tolist(),\
        (2 * math.pi / vol * np.cross(latvec[:,0], latvec[:,1])).tolist()]
        return [vol, rlatvec]

    def get_scell_dim(self, ave_natom = 100):
        n1n2n3 = (ave_natom / self.output['natom'])
        [l1, l2, l3] = [np.linalg.norm(vec) for vec in self.output['latvec']]
        if n1n2n3 > 1:
            n1 = (n1n2n3*l2*l3/l1**2)**(1/3)
            n2 = l1 * n1 / l2
            n3 = l1 * n1 / l3
            mesh = [n1, n2, n3]
            mesh  = [ 1 if ni < 1 else int(ni//1) for ni in mesh]
        else:
            mesh = [1, 1, 1]
        return mesh


    def get_scell(self, dim_scell = "", #np.diag(np.full(3,1)),
                  ave_natom = 100, symprec = 1e-4, conv = True):
        """
        usage: creating supercell from dim_scell: 3x3 integer matrix with constraint of max_natom atoms
        input: pos_input: self.primcell using get_prim
        output: self.scell
        assumption: dim_scell should be diagonal 3x3 matrix
        """
        # create conventional cell
        if conv:
            self.refine_cell(symprec = symprec)
        # assume diagonal supercell
        [self.output['vol'], self.output['rlatvec']] = class_poscar.recip_vol(self.output['latvec'])
        n1n2n3 = (ave_natom / self.output['natom'])
        [l1, l2, l3] = [np.linalg.norm(vec) for vec in self.output['latvec']]
        if n1n2n3 > 1:
            n1 = (n1n2n3*l2*l3/l1**2)**(1/3)
            n2 = l1 * n1 / l2
            n3 = l1 * n1 / l3
            mesh = [n1, n2, n3]
            mesh  = [ 1 if ni < 1 else int(ni//1) for ni in mesh]
        else:
            mesh = [1, 1, 1]
        # record supercell dimension
        self.sc_mesh = np.diag(np.array(mesh))
        if dim_scell == "":
            latp_list = class_poscar.latp_from_scmat(np.diag(np.array(mesh)))
        else:
            latp_list = class_poscar.latp_from_scmat(dim_scell)
            mesh = list(np.diagonal(dim_scell))
        # total number of lattice point
        n1n2n3 = mesh[0]*mesh[1]*mesh[2]
        if False:
            print (mesh)
            print (latp_list)        
        # create self.scell
        self.scell = {}
        self.scell['name'] = " ".join(map(str, mesh)) + "-" + self.output['name']
        self.scell['scale'] = self.output['scale']
        self.scell['latvec'] = np.matmul(np.diag(np.array(mesh)), self.output['latvec']).tolist()
        self.scell['spe'] = self.output['spe']
        self.scell['cspe'] = [i*n1n2n3 for i in self.output['cspe']]
        self.scell['spe_all'] = []
        for idx, el in enumerate(self.output['spe']):
            self.scell['spe_all'] += [ el ] * self.output['cspe'][idx] * n1n2n3
        self.scell['natom'] = self.output['natom'] * n1n2n3
        self.scell['cord'] = self.output['cord'] # assume direct coordinate
        self.scell['atompos'] = []
        for pos in self.output['atompos']:
            for latp in latp_list:
                inv_mat = np.linalg.inv(np.diag(np.array(mesh)))
                pos_tmp = np.matmul((np.array(pos) + np.array(latp)), inv_mat).tolist()
                self.scell['atompos'].append(pos_tmp)
        [self.scell['vol'], self.scell['rlatvec']] = class_poscar.recip_vol(self.scell['latvec'])        
        # print info
        if False:
            print (self.scell['atompos'])
            print (self.output['atompos'])        
        

    def get_prim(self, symprec = 1e-4, read_prim = "", is_to_prim = True, is_no_idealize = False):
        '''
        usage: extract primitive cell from poscar
        input: poscar
        output: primitive cell
        assumption: None
        '''
        from element.atominfo import class_atom 
        if read_prim != "":
            with open(read_prim, "r") as file:
                lines =  file.readlines()                
                self.primcell['spe'] = lines[5].split()
                self.primcell['scale'] = float(lines[1])
                self.primcell['cspe'] = list(map(int,lines[6].split()))
                self.primcell['spe_all'] = []
                for i in range(len(self.primcell['spe'])):
                    self.primcell['spe_all'] += [self.primcell['spe'][i]]*self.primcell['cspe'][i]
                self.primcell['natom'] = sum(self.primcell['cspe'])
                self.primcell['latvec'] = []
                self.primcell['atompos'] = []
                for i in range(2,5):
                    self.primcell['latvec'].append(list(map(float,lines[i].split()[0:3])))
                for i in range(8,8+self.primcell['natom']):
                    self.primcell['atompos'].append(list(map(float,lines[i].split()[0:3])))
                [self.primcell['vol'], self.primcell['rlatvec']] = class_poscar.recip_vol(self.primcell['latvec'])
                self.primcell['mass'] = [ class_atom.amass[el] for el in self.primcell['spe_all']]
                # debug
                if False:
                    print ("-->"*20)
                    #print (self.primcell['spe'])
                    #print (self.primcell['cspe'])
                    print (self.primcell['spe_all'])
                    print (self.primcell['mass'])                
        else:        
            try:
                import spglib as spg
            except ImportError:
                from pyspglib import spglib as spg
            dataset = self.get_sym(symprec)
            if True:
                # new version of spglib > 1.8.x
                lattice, scaled_positions, numbers = \
                    spg.standardize_cell(self.ini_cell(),
                                         to_primitive = is_to_prim,
                                         no_idealize = is_no_idealize,
                                         symprec =  symprec)
            else:
                # old version deprecated
                lattice, scaled_positions, numbers = \
                    spg.find_primitive(self.ini_cell(),
                                       symprec=symprec,
                                       angle_tolerance=-1.0)
            #print (lattice)
            #print (numbers)
            #print (numbers)
            self.primcell['latvec'] = lattice.tolist()
            self.primcell['spe'] = []
            self.primcell['cspe'] = []
            self.primcell['scale'] = 1.0 # assume spglib-generated primitive cell
            spe_tmp = [ self.output['spe'][i-1] for i in numbers ]
            for el in spe_tmp:
                if el not in self.primcell['spe']:
                    self.primcell['spe'].append(el)
            for el in self.primcell['spe']:
                self.primcell['cspe'].append(spe_tmp.count(el))
            self.primcell['natom'] = len(spe_tmp)
            self.primcell['atompos'] = []
            for el in self.primcell['spe']:
                for idx, el_in in enumerate(spe_tmp):
                    if el_in == el:
                        self.primcell['atompos'].append(scaled_positions[idx].tolist())
            [self.primcell['vol'], self.primcell['rlatvec']] = class_poscar.recip_vol(self.primcell['latvec'])
            for i in range(len(self.primcell['spe'])):
                self.primcell['spe_all'] +=	[self.primcell['spe'][i]]*self.primcell['cspe'][i]
            self.primcell['mass'] = [ class_atom.amass[el] for el in self.primcell['spe_all']]
            # transform from C to D using scaled pos from spglib
            self.primcell['cord'] = "D"
            # dump primitive cell structure
            if is_to_prim:
                self.write_poscar_dp(self.primcell, self.pos_path+"POSCAR-prim")
            else:
                self.write_poscar_dp(self.primcell, self.pos_path+"POSCAR-conv")
                

    def get_sym_str(self, symprec = 1e-5):
        dataset = self.get_sym(symprec = 1e-5)
        rotat = dataset['translations']
        transl = dataset['rotations']
        ndigit = 6
        str_out = ""
        str_out += str(len(rotat)) + "\n"
        for i in range(len(rotat)):
            for row in rotat[i]:
                list_tmp = np.around(np.array(row), ndigit).tolist()
                str_out += " ".join(map(str, list_tmp)) + "\n"
            str_out += "\n"
            list_tmp = np.around(np.array(transl[i]), ndigit).tolist()
            str_out += " ".join(map(str, list_tmp)) + "\n"
            str_out += "\n"
        self.sym_str = str_out
    
    
    def get_sym(self, symprec = 1e-5):
        """
        usage: symmetry analysis of POSCAR through spglib      
        input: poscar
        output: dataset
        assumption: None
        """
        try:
            import spglib as spg
        except ImportError:
            from pyspglib import spglib as spg        
        cell = self.ini_cell()                                
        spacegroup = spg.get_spacegroup(cell, symprec)
        #symmetry = spg.get_symmetry(cell, symprec)
        dataset = spg.get_symmetry_dataset(cell, symprec, angle_tolerance=-1.0, hall_number=0)
        if False:
            print ("info: Space group for original poscar : " + spacegroup)
            print ("info: Point group for original poscar: " + dataset['pointgroup'])
            print ("info: equivalent_atoms")
            print (dataset['equivalent_atoms'])
        #return 
        return dataset


    def get_sym_prim(self, symprec = 1e-5):
        '''
        usage: get symmetry operators for the primitive cell
        '''
        try:
            import spglib as spg
        except ImportError:
            from pyspglib import spglib as spg
        cell = self.ini_cell_prim(symprec)
        spacegroup = spg.get_spacegroup(cell, symprec)
        dataset = spg.get_symmetry_dataset(cell, symprec, angle_tolerance=-1.0, hall_number=0)
        print ("info: Space group for primitive cell: " + spacegroup)
        print ("info: Point group for primitive cell: " + dataset['pointgroup'])
        print ("info: Number of rotational operators: " + str(len(dataset['rotations'])))        
        # extend the symmetry operations for cartesian and reciprocal space
        self.symset = dataset
        self.symset['crotations']=[]
        self.symset['qrotations']=[]
        self.symset['ctranslations']=[]
        latvec = np.array(self.primcell['latvec'])
        t_latvec = np.transpose(latvec)
        mmt_latvec = np.matmul(latvec, t_latvec)
        # cartesian rotation and q-space rotation
        for rot in dataset['rotations']:
            crot = np.matmul(t_latvec, np.matmul(rot, np.linalg.inv(t_latvec)))
            qrot = np.matmul(mmt_latvec, np.matmul(rot, np.linalg.inv(mmt_latvec)))
            self.symset['crotations'].append(np.around(crot,12))
            self.symset['qrotations'].append(qrot)
        # cartesian translation
        for tra in dataset['translations']:
            ctra = np.matmul(tra, latvec)
            self.symset['ctranslations'].append(ctra)
        # enforce time reversal symmetry
        self.symset['qrotations_tr'] = copy.deepcopy(self.symset['qrotations'])
        self.symset['crotations_tr'] = copy.deepcopy(self.symset['crotations'])
        for i in range(len(self.symset['qrotations'])):
            self.symset['qrotations_tr'].append(-1 * self.symset['qrotations'][i])
            self.symset['crotations_tr'].append(-1 * self.symset['crotations'][i])
        # round symmetry operators in reciprocal space
        for idx, item in enumerate(self.symset['qrotations_tr']):
            self.symset['qrotations_tr'][idx] = np.around(self.symset['qrotations_tr'][idx], \
                                                          int(abs(math.log(symprec))) )
        # delete duplicated operators
        [ self.symset['qrotations_tr_nodup'], self.symset['symop_idx'] ]\
            = self.get_unique_symop(self.symset['qrotations_tr'])
        if debug:
            print ("Before and after deleting duplicated symmetry operators: ")
            print (len(self.symset['qrotations_tr']))
            print (len(self.symset['qrotations_tr_nodup']))
            print ("Index for symmetry operators: ")
            print (self.symset['symop_idx'])
        # return symmetry dataset
        self.symdataset	= dataset
        return dataset
    

    @staticmethod
    def get_unique_symop(symop_list):
        symop_uniq = []
        symop_idx = []
        threshold = 1e-4
        for idx, symop in enumerate(symop_list):
            if symop_uniq != []:
                diff_list = [ np.linalg.norm(item-symop) for item in symop_uniq]                
                if min(diff_list) > threshold:
                    symop_uniq.append(symop)
                    symop_idx.append(idx)
                else:
                    pass
            else:
                symop_uniq.append(symop)
                symop_idx.append(idx)
        return [symop_uniq, symop_idx]

    @staticmethod
    def dump_cluster(unique_pair_latp_image, primcell):
        pass
        
    @staticmethod
    def dump_sym(rotat, transl, pathout, return_str = True):
        ndigit = 6
        str_out = ""
        str_out += str(len(rotat)) + "\n"
        for i in range(len(rotat)):
            for row in rotat[i]:
                list_tmp = np.around(np.array(row), ndigit).tolist()
                str_out += " ".join(map(str, list_tmp)) + "\n"
            str_out += "\n"
            list_tmp = np.around(np.array(transl[i]), ndigit).tolist()
            str_out += " ".join(map(str, list_tmp)) + "\n"
            str_out += "\n"
        if not return_str:
            with open(os.path.join(pathout, "sym.out"), "w") as file:
                file.write(str_out)
        else:
            return str_out

    @staticmethod
    def filter_scmat(unique_pair, scmat, primcell, image = 1):
        '''
        cluster example
        [[[[0, 0, 0], 0], [[0, 0, 0], 0]], [[[0, 0, 0], 0], [[-1, 0, 0], 0]]]
        return list of kept clusters
        '''
        idx_list = []
        for idx, pair in enumerate(unique_pair):
            pos1 = np.array(pair[0][0])+np.array(primcell['atompos'][ int(pair[0][1]) ])
            pos2 = np.array(pair[1][0])+np.array(primcell['atompos'][ int(pair[1][1]) ])
            radius = np.linalg.norm( np.matmul((pos1-pos2), np.array(primcell['latvec'])) )
            radius_list = []
            for i in range(-1*image, image+1): # control image range
                for j in range(-1*image, image+1):
                    for k in range(-1*image, image+1):
                        pos2_tmp = pos2 +\
                            np.array(scmat)[0]*i +\
                            np.array(scmat)[1]*j+\
                            np.array(scmat)[2]*k
                        radius_tmp = np.linalg.norm( np.matmul((pos1-pos2_tmp), np.array(primcell['latvec'])) )
                        radius_list.append(radius_tmp)
            radius_min = min(radius_list)
            if radius_min >= radius:
            #if not (radius_min < radius and abs(radius_min-radius)>1e-8):
                idx_list.append(idx)
        # summary
        return idx_list


    def get_unique_2body(
            self,
            scmat,
            map_list,
            latp_list,
            map_latp_list,
            primcell,
            symprecin = 1e-4,
            cutoff = 10,
            periodicity = True,
            pathout = "./"):
        '''
        usage: return unique 2body interactions including improper clusters
        input: scamt --> 3x3 list for supercell transformation matrix
               map_list --> [[idx_atom_supercell, idx_atom_primcell]]
               latp_list --> [ lattice point for the idx_atom_supercell]
               map_latp_list --> [ lattice point for the idx_atom_supercell, idx_atom_primcell]
               primcell --> define in the class
        assumption: assume symmetry operator has identity operation
        '''
        unique_latp_list = list(map(list, sorted(set(map(tuple, latp_list)), reverse = False)))
        print (f"info: unique_latp_list: {unique_latp_list}; length: {len(unique_latp_list)}")
        #print (len(unique_latp_list))
        pair_map_unqCl = np.zeros((primcell['natom'], len(unique_latp_list), primcell['natom']))
        pair_map_symOp = np.zeros((primcell['natom'], len(unique_latp_list), primcell['natom']))        
        
        dataset = self.get_sym_prim(symprecin)
        class_poscar.dump_sym(dataset['rotations'], dataset['translations'], pathout)
        scancell = max([abs(np.amax(np.array(scmat))), abs(np.amin(np.array(scmat)))]) + 3
        grid_atoms = (2*scancell+1)**3*primcell['natom']
        pair_map_idx = np.zeros(( primcell['natom'], grid_atoms ))
        
        unique_pair_latp =[]
        unique_pair_idx_scell = []
        unique_count = 0
        if debug:
            print ("debug: Rotations: "+str(len(dataset['rotations'])))
            print (dataset['rotations'])
            print ("debug: Translations: "+str(len(dataset['rotations'])))
            print (dataset['translations'])
        for idx_p, pos_p in enumerate(primcell['atompos']):
            for idx_s, lat_s in enumerate(latp_list):
                pos_tmp = np.array(primcell['atompos'][map_list[idx_s][1]])
                pair_pos = np.array([pos_p, (np.array(lat_s)+pos_tmp).tolist()])
                [pair_pos_min, dist_min] = self.find_min_dis_pos(
                    pair_pos[0], pair_pos[1], scmat, primcell['latvec'], image = 0) #yixia fix image            
                # throw away clusters with large cutoff distance
                if dist_min > cutoff:
                    continue
                # yes periodic image
                if True:
                    # with periodic image for min distance based on atomic position convention
                    pair_latp = [self.pos_to_latp(pos, primcell['atompos'], symprecin) for pos in pair_pos_min]
                else:                    
                    # no periodic image 
                    pair_latp = [self.pos_to_latp(pos, primcell['atompos'], symprecin) for pos in pair_pos]
                # set index for the large supercell
                #pair_idx = self.latp_to_idx(pair_latp, primcell['natom'], scancell)
                latp_haha = map_latp_list[idx_s][0]
                idp_haha = map_latp_list[idx_s][1]
                if pair_map_unqCl[idx_p][unique_latp_list.index(latp_haha)][idp_haha] == 0:
                    unique_count += 1
                    unique_pair_latp.append(pair_latp)
                    unique_pair_idx_scell.append([map_latp_list.index([[0,0,0],idx_p]), idx_s, dist_min])
                    '''
                    print ("*********************")
                    print ([map_latp_list.index([[0,0,0],idx_p]), idx_s, dist_min])
                    print (pair_latp)
                    print ("*********************")
                    '''
                    # new-----------------------------
                    latp_haha = map_latp_list[idx_s][0]
                    idp_haha = map_latp_list[idx_s][1]
                    if latp_haha in unique_latp_list:
                        pair_map_unqCl[idx_p][unique_latp_list.index(latp_haha)][idp_haha] = unique_count
                    latp_haha = list((-1*np.array(map_latp_list[idx_s][0])).astype(int))
                    idp_haha = map_latp_list[idx_s][1]
                    if latp_haha in unique_latp_list:
                        pair_map_unqCl[idp_haha][unique_latp_list.index(latp_haha)][idx_p] = unique_count
                    # new-----------------------------
                    # rotate positions
                    pair_pos_rot = [ (np.matmul(rmat, np.array(pair_pos_min).T).T).tolist()
                                     for rmat in dataset['rotations']]
                    # translate positions
                    for id_rt  in range(len(pair_pos_rot)):
                        pair_pos_rot[id_rt][0] = (np.array(pair_pos_rot[id_rt][0])+dataset['translations'][id_rt])
                        pair_pos_rot[id_rt][1] = (np.array(pair_pos_rot[id_rt][1])+dataset['translations'][id_rt])
                    # loop over rotated and translated pair    
                    for id_pair, pair in enumerate(pair_pos_rot):
                        pair_latp_rot = [ self.pos_to_latp(pos, primcell['atompos'], symprecin) for pos in pair]
                        pair_idx_rot = self.latp_to_idx(pair_latp_rot, primcell['natom'], scancell)
                        # new-----------------------------
                        latp_haha = list((np.array(pair_latp_rot[1][0])-np.array(pair_latp_rot[0][0])).astype(int))
                        idx_1 = pair_latp_rot[0][1]
                        idx_2 = pair_latp_rot[1][1]
                        #print (f"idx_1: {idx_1}; idx_2: {idx_2}; latp_haha: {latp_haha}")
                        if latp_haha in unique_latp_list:
                            pair_map_unqCl[idx_1][unique_latp_list.index(latp_haha)][idx_2] = unique_count
                        latp_haha = list((np.array(pair_latp_rot[0][0])-np.array(pair_latp_rot[1][0])).astype(int))
                        idx_1 = pair_latp_rot[1][1]
                        idx_2 = pair_latp_rot[0][1]
                        #print (f"idx_1: {idx_1}; idx_2: {idx_2}; latp_haha: {latp_haha}")
                        if latp_haha in unique_latp_list:
                            pair_map_unqCl[idx_1][unique_latp_list.index(latp_haha)][idx_2] = unique_count
                        # new-----------------------------
        # summary
        if False:
            print (f"pair_map_unqCl: {pair_map_unqCl}")
        # enforce periodic boundary condition
        idx_list = class_poscar.filter_scmat(unique_pair_latp, scmat, primcell, image = 3)
        unique_pair_idx_scell_image = [unique_pair_idx_scell[idx] for idx in idx_list]
        unique_pair_latp_image = [unique_pair_latp[idx] for idx in idx_list]
        print (f"info: withtout considering periodic image: {len(unique_pair_idx_scell)}")
        #print (len(unique_pair_idx_scell))
        print (f"info: with considering periodic image: {len(unique_pair_idx_scell_image)}")
        #print (len(unique_pair_idx_scell_image))
        # to do list
        #class_poscar.dump_cluster(unique_pair_latp_image, primcell)
        print (f"info: after filting with supercell: {len(idx_list)}")
        return unique_pair_idx_scell_image, unique_pair_latp_image

        
    def get_unique_2body_back(
            self,
            scmat,
            map_list,
            latp_list,
            map_latp_list,
            primcell,
            symprecin = 1e-4,
            cutoff = 15,
            periodicity = True):
        '''
        usage: return unique 2body interactions including improper clusters
        input: scamt --> 3x3 list for supercell transformation matrix
               map_list --> [[idx_atom_supercell, idx_atom_primcell]]
               latp_list --> [ lattice point for the idx_atom_supercell]
               map_latp_list --> [ lattice point for the idx_atom_supercell, idx_atom_primcell]
               primcell --> define in the class
        assumption: assume symmetry operator has identity operation
        '''
        unique_latp_list = list(map(list, sorted(set(map(tuple, latp_list)), reverse = False)))
        print (f"unique_latp_list: {unique_latp_list}")
        print (len(unique_latp_list))
        pair_map_unqCl = np.zeros((primcell['natom'], len(unique_latp_list), primcell['natom']))
        pair_map_symOp = np.zeros((primcell['natom'], len(unique_latp_list), primcell['natom']))
        
        
        dataset = self.get_sym_prim(symprecin)
        scancell = max([abs(np.amax(np.array(scmat))), abs(np.amin(np.array(scmat)))]) + 3
        grid_atoms = (2*scancell+1)**3*primcell['natom']
        pair_map_idx = np.zeros(( primcell['natom'], grid_atoms ))
        
        unique_pair_latp =[]
        unique_pair_idx_scell = []
        unique_count = 0
        if debug:
            print ("debug: Rotations: "+str(len(dataset['rotations'])))
            print (dataset['rotations'])
            print ("debug: Translations: "+str(len(dataset['rotations'])))
            print (dataset['translations'])
        for idx_p, pos_p in enumerate(primcell['atompos']):
            for idx_s, lat_s in enumerate(latp_list):
                pos_tmp = np.array(primcell['atompos'][map_list[idx_s][1]])
                pair_pos = np.array([pos_p, (np.array(lat_s)+pos_tmp).tolist()])
                [pair_pos_min, dist_min] = self.find_min_dis_pos(
                    pair_pos[0], pair_pos[1], scmat, primcell['latvec'])
                # throw away clusters with large cutoff distance
                if dist_min > cutoff:
                    continue
                # yes periodic image
                if True:
                    # with periodic image for min distance based on atomic position convention
                    pair_latp = [self.pos_to_latp(pos, primcell['atompos'], symprecin) for pos in pair_pos_min]
                else:                    
                    # no periodic image 
                    pair_latp = [self.pos_to_latp(pos, primcell['atompos'], symprecin) for pos in pair_pos]
                # set index for the large supercell
                pair_idx = self.latp_to_idx(pair_latp, primcell['natom'], scancell)
                if debug:
                    print ("debug: check_latp")
                    check_latp = [[[0, 0, 0], 0], [[0, 0, -1], 1]]
                    if pair_latp == check_latp:
                        print ("--"*20+"I find it!")
                if (pair_map_idx[pair_idx[0][0]][pair_idx[0][1]] == 0) and \
                   (pair_map_idx[pair_idx[1][0]][pair_idx[1][1]] == 0):
                    unique_count += 1
                    if debug:
                        print (f"debug: unique_count: {unique_count}")
                        print (f"debug: pair_lat: {pair_latp}")
                    unique_pair_latp.append(pair_latp)
                    unique_pair_idx_scell.append([map_latp_list.index([[0,0,0],idx_p]), idx_s, dist_min])
                    pair_map_idx[ pair_idx[0][0] ][ pair_idx[0][1] ] = unique_count 
                    pair_map_idx[ pair_idx[1][0] ][ pair_idx[1][1] ] = unique_count
                    """
                    # new-----------------------------
                    latp_haha = map_latp_list[idx_s][0]
                    idp_haha = map_latp_list[idx_s][1]
                    if latp_haha in unique_latp_list:
                        pair_map_unqCl[idx_p][unique_latp_list.index(latp_haha)][idp_haha] = unique_count
                    latp_haha = list((-1*np.array(map_latp_list[idx_s][0])).astype(int))
                    idp_haha = map_latp_list[idx_s][1]
                    if latp_haha in unique_latp_list:
                        pair_map_unqCl[idp_haha][unique_latp_list.index(latp_haha)][idx_p] = unique_count
                    # new-----------------------------
                    """
                    # rotate positions
                    pair_pos_rot = [ (np.matmul(rmat, np.array(pair_pos_min).T).T).tolist()
                                     for rmat in dataset['rotations']]
                    # translate positions
                    for id_rt  in range(len(pair_pos_rot)):
                        pair_pos_rot[id_rt][0] = (np.array(pair_pos_rot[id_rt][0])+dataset['translations'][id_rt])
                        pair_pos_rot[id_rt][1] = (np.array(pair_pos_rot[id_rt][1])+dataset['translations'][id_rt])
                    # loop over rotated and translated pair    
                    for id_pair, pair in enumerate(pair_pos_rot):
                        if debug:
                            print (f"debug: pair_pos_rot: {pair}")
                        pair_latp_rot = [ self.pos_to_latp(pos, primcell['atompos'], symprecin) for pos in pair]
                        print (pair_latp_rot)
                        pair_idx_rot = self.latp_to_idx(pair_latp_rot, primcell['natom'], scancell)
                        pair_map_idx[ pair_idx_rot[0][0] ][ pair_idx_rot[0][1] ] = unique_count 
                        pair_map_idx[ pair_idx_rot[1][0] ][ pair_idx_rot[1][1] ] = unique_count
                        """
                        # new-----------------------------
                        latp_haha = list((np.array(pair_latp_rot[1][0])-np.array(pair_latp_rot[0][0])).astype(int))
                        idx_1 = pair_latp_rot[0][1]
                        idx_2 = pair_latp_rot[1][1]
                        print (f"idx_1: {idx_1}; idx_2: {idx_2}; latp_haha: {latp_haha}")
                        if latp_haha in unique_latp_list:
                            pair_map_unqCl[idx_1][unique_latp_list.index(latp_haha)][idx_2] = unique_count
                        latp_haha = list((np.array(pair_latp_rot[0][0])-np.array(pair_latp_rot[1][0])).astype(int))
                        idx_1 = pair_latp_rot[1][1]
                        idx_2 = pair_latp_rot[0][1]
                        print (f"idx_1: {idx_1}; idx_2: {idx_2}; latp_haha: {latp_haha}")
                        if latp_haha in unique_latp_list:
                            pair_map_unqCl[idx_1][unique_latp_list.index(latp_haha)][idx_2] = unique_count
                        # new-----------------------------                                                
                        """
                        if False:
                            if pair_latp == check_latp:
                                print ("--"*20+"I find it during counting!")
                                print (f"unique_count: {unique_count}")
                                print (pair_latp_rot)
                                print (pair_idx_rot)
        print (pair_map_unqCl)
        if debug:
            print ("debug: --->max index in pair_map_idx")
            print (np.amax(pair_map_idx))
            print ("debug: --->unique_count")
            print (unique_count)
            print ("debug: --->unique_pair_latp")
            print (unique_pair_latp)
            print ("debug: --->sorted unique_pair_idx_scell")
            print (sorted(unique_pair_idx_scell, key=lambda x:x[2]))
            print ("debug: --->unsorted unique_pair_idx_scell")
            print (unique_pair_idx_scell)
            print ("debug: check onsite cluster: ")
            
            for check_latp in unique_pair_latp:
                print (check_latp)
                pair_idx_rot = self.latp_to_idx(check_latp, primcell['natom'], scancell)
                print (pair_idx_rot)
                idx_map = pair_map_idx[pair_idx_rot[0][0]][pair_idx_rot[0][1]]
                print (idx_map)
                idx_map = pair_map_idx[pair_idx_rot[1][0]][pair_idx_rot[1][1]]
                print (idx_map)
                print (unique_pair_latp[int(idx_map-1)])
                print (unique_pair_idx_scell[int(idx_map-1)])
                print (unique_pair_idx_scell[unique_pair_latp.index(check_latp)])
                print ("\n")
            print (f"scancell: {scancell}")
            #print (pair_map_idx.tolist())
            if False:
                test_latp = []
                for check_latp in unique_pair_latp:
                    if check_latp not in test_latp:
                        test_latp.append(check_latp)
                print (f"length of unique: {len(unique_pair_latp)}")
                print (f"length of unique: {len(test_latp)}")
                print (np.sum(pair_map_idx))
        # use default value of True
        periodicity = False
        unique_pair_idx_scell_image = []
        unique_pair_latp_image = []
        if periodicity:
            for check_latp in unique_pair_latp:
                pair_idx_rot = self.latp_to_idx(check_latp, primcell['natom'], scancell)
                idx_map = pair_map_idx[ pair_idx_rot[1][0] ][ pair_idx_rot[1][1] ]
                if unique_pair_latp[int(idx_map-1)] not in unique_pair_latp_image:
                    unique_pair_latp_image.append(unique_pair_latp[int(idx_map-1)])
                    unique_pair_idx_scell_image.append(unique_pair_idx_scell[int(idx_map-1)])
        else:
            unique_pair_idx_scell_image = unique_pair_idx_scell
            unique_pair_latp_image = unique_pair_latp
        print ("withtout considering periodic image:")
        print (len(unique_pair_idx_scell))
        print ("with considering periodic image:")
        print (len(unique_pair_idx_scell_image))

        return unique_pair_idx_scell_image, unique_pair_latp_image


    def pair_sym_from_radius(self, radius = 5.0, symprec = 1e-4):
        '''
        usage: generate symmetrically distinct pair of atoms 
        according to radius --> ineratomic distance
        '''
        pass


    def pair_sym_from_scmat(self, scmatin = [[2,0,0],[0,2,0],[0,0,2]], symprec = 1e-4):
        '''                                                                                       
        usaeg: generate symmetrically distinct pair of atoms                                   
        input: 1. 3x3 matrix for supercell cell with 2x2x2 as the default                         
        output: symmetrically distinct pair of atoms including onsite pair                     
        assumptions:                                                                              
        1. the original lattice point lies at [0,0,0]                                             
        2. the lattice points are based on the extracted primitive cell                            
        '''
        self.get_prim(symprec)
        # to do: centering lattice point
        sel_latp = self.latp_from_scmat(scmatin)
        map_list = []
        map_latp_list = []
        count_atom = 0
        for iatom in range(self.primcell['natom']):
            for latp in sel_latp:
                count_atom += 1
                map_latp_list.append([latp,iatom])
                map_list.append([count_atom, iatom])
        if debug:
            print (f"debug: map_list: {map_list}")
            print (f"debug: sel_latp: {sel_latp}")
        latp_list = sel_latp * self.primcell['natom']
        self.unique_pair_idx_scell, self.unique_pair_latp = \
        self.get_unique_2body(
            scmat = scmatin,
            map_list = map_list,
            latp_list = latp_list,
            map_latp_list = map_latp_list,
            primcell = self.primcell,
            symprecin = symprec,
            periodicity = True
        )
        #
        if debug:
            print ("debug: unique_pair_idx_scell")
            print (self.unique_pair_idx_scell)
            print ("debug: unique_pair_latp")
            print (self.unique_pair_latp)


    def get_csld_cluster_from_scell(self, cutoff_dict = {'2':10.0}, return_only_uniqueC = True):
        # implement return pair cluster
        # default to contrain the 1body terms in 2body interactions
        # cutoff_dict = {'2':4.0, '3':4.0, '4':0}
        uniqueC = []
        if '2' in cutoff_dict.keys():
            cutoff = cutoff_dict['2']
            self.pair_sym_from_scell(symprec = 1e-4, cutoff = cutoff)
            # sorting cluster
            idx_sort = np.argsort(np.array(self.unique_pair_idx_scell)[:,-1])
            unique_pair_idx = [self.unique_pair_idx_scell[idx][-1] for idx in idx_sort]
            unique_pair_latp = [self.unique_pair_latp[idx] for idx in idx_sort]
            # 1body clusters
            for idx, pair in enumerate(unique_pair_latp):
                if unique_pair_idx[idx] < 1e-6:
                    id_atom1 = pair[0][1]
                    uniqueC.append([self.primcell['atompos'][id_atom1]])
            # 2body clusters
            for	idx, pair in enumerate(unique_pair_latp):
                id_atom1 = pair[0][1]
                latp_atom2 = pair[1][0]
                id_atom2 = pair[1][1]
                pos_atom1 = self.primcell['atompos'][id_atom1]
                pos_atom2 = (np.array(latp_atom2) + np.array(self.primcell['atompos'][id_atom2])).tolist()
                uniqueC.append([pos_atom1, pos_atom2])
        # return
        if return_only_uniqueC:
            return uniqueC
        else:
            return [uniqueC, unique_pair_latp]


    def get_csld_cluster_from_scell_phdb(self, cutoff_dict = {'2':10.0}):
        uniqueC = []
        dict_pair = {}
        if '2' in cutoff_dict.keys():
            cutoff = cutoff_dict['2']
            self.pair_sym_from_scell(symprec = 1e-4, cutoff = cutoff)
            # sorting cluster                                                                             
            idx_sort = np.argsort(np.array(self.unique_pair_idx_scell)[:,-1])
            unique_pair_idx = [self.unique_pair_idx_scell[idx][-1] for idx in idx_sort]
            unique_pair_latp = [self.unique_pair_latp[idx] for idx in idx_sort]
            '''
            # 1body clusters
            for idx, pair in enumerate(unique_pair_latp):
                if unique_pair_idx[idx] < 1e-6:
                    id_atom1 = pair[0][1]
                    uniqueC.append([self.primcell['atompos'][id_atom1]])
            '''
            dict_pair['pair_idxscell'] = []
            dict_pair['pair_spe'] = []
            dict_pair['pair_latp'] = []
            dict_pair['pair_pos'] = []
            dict_pair['pair_dist'] = []

            # 2body clusters                                                                                 
            for idx, pair in enumerate(unique_pair_latp):
                id_atom1 = pair[0][1]
                latp_atom2 = pair[1][0]
                id_atom2 = pair[1][1]
                pos_atom1 = self.primcell['atompos'][id_atom1]
                pos_atom2 = (np.array(latp_atom2) + np.array(self.primcell['atompos'][id_atom2])).tolist()
                id_atom1_scell = self.unique_pair_idx_scell[idx_sort[idx]][0]
                id_atom2_scell = self.unique_pair_idx_scell[idx_sort[idx]][1]
                distance = self.unique_pair_idx_scell[idx_sort[idx]][-1]
                uniqueC.append([
                    [id_atom1_scell, id_atom2_scell],
                    [self.spe_all[id_atom1_scell], self.spe_all[id_atom2_scell] ],
                    [[[0,0,0],id_atom1], [latp_atom2, id_atom2]],
                    [pos_atom1, pos_atom2],
                    distance
                ])
                
                dict_pair['pair_idxscell'].append( [id_atom1_scell, id_atom2_scell] )
                dict_pair['pair_spe'].append( [self.spe_all[id_atom1_scell], self.spe_all[id_atom2_scell] ] )
                dict_pair['pair_latp'].append( [[[0,0,0],id_atom1], [latp_atom2, id_atom2]] )
                dict_pair['pair_pos'].append( [pos_atom1, pos_atom2] )
                dict_pair['pair_dist'].append( distance )
                
        return uniqueC, dict_pair


    
    @staticmethod
    def get_radius_cluster(lpt_cluster, primcell):
        '''
        [[latp1, atom1], [latp2, atom2], [latp3, atom3]]
        '''
        radius = 0
        if len(lpt_cluster) == 1:
            radius = 0.0
            return radius
        else:
            pos_list = []
            for atom in lpt_cluster:
                id_atom =  atom[1]
                pos_list.append( (np.array(atom[0])+ np.array(primcell['atompos'][id_atom])).tolist() )
            poscart_array = np.matmul(np.array(pos_list), np.array(primcell['latvec']))
            dis_list = []
            len_pos = len(pos_list)
            for i in range(len_pos):
                for j in range(i+1, len_pos):
                    dis_list.append( np.linalg.norm(poscart_array[i]-poscart_array[j]) )
            return max(dis_list)
        

    @staticmethod
    def get_bond_cluster(lpt_cluster, primcell, image = 2, cutoff = 4.0):
        #latp_list = []
        bond_cluster = []
        for i in range(-1*image, image+1):
            for j in range(-1*image, image+1):
                for k in range(-1*image, image+1):
                    #latp_list.append([i,j,k])
                    for l in range(primcell['natom']):
                        new_cluster = lpt_cluster + [[[i,j,k],l]]
                        radius_tmp = class_poscar.get_radius_cluster(new_cluster, primcell)
                        if radius_tmp < cutoff:
                            bond_cluster.append(new_cluster)
                            if debug:
                                print (radius_tmp)
        return bond_cluster


    def get_3body_cluster_notworking(self, cutoff = 4.0):
        debug_local = True
        try:
            self.unique_pair_latp
        except NameError:
            self.pair_sym_from_scell(self, cutoff = cutoff)
        idx_sort = np.argsort(np.array(self.unique_pair_idx_scell)[:,-1])
        
        # generate all lattice points within the cutoff
        latp_cutoff = []
        dim_latp = np.ndarray.max(np.absolute(np.array(self.latp_list))) + 1
        scancell = dim_latp
        grid_atoms = (2*dim_latp+1)**3*self.primcell['natom']    

        # start index of 3body clusters
        index_3body = np.zeros(( self.primcell['natom'], grid_atoms, grid_atoms ))
        pnatom = self.primcell['natom']
        self.unique_3body_cluster = []        
        for idx in idx_sort: 
            pair_latp = self.unique_pair_latp[idx]
            if class_poscar.get_radius_cluster(pair_latp, self.primcell) < cutoff:
                trip_latp = class_poscar.get_bond_cluster(pair_latp,
                                                          self.primcell,
                                                          image = dim_latp,
                                                          cutoff = cutoff)
                for trip in trip_latp:
                    ind_1 = trip[0][1]
                    ind_2 = self.latp_to_num_new(trip[1], scancell, pnatom)
                    ind_3 = self.latp_to_num_new(trip[2], scancell, pnatom)
                    if index_3body[ind_1, ind_2, ind_3] == 0:
                        self.unique_3body_cluster.append(trip)
                        for trip_sub in self.get_perm_sym(trip):
                            ind_1 = trip_sub[0][1]
                            ind_2 = self.latp_to_num_new(trip[1], scancell, pnatom)
                            ind_3 = self.latp_to_num_new(trip[2], scancell, pnatom)
                            index_3body[ind_1, ind_2, ind_3] = 1
        #summary
        print (f"Number of unique 3body clusters: {len(self.unique_3body_cluster)}")



    def get_3body_cluster(self, cutoff = 4.0):
        debug_local = True
        try:
            self.unique_pair_latp
        except AttributeError:
            self.pair_sym_from_scell(self, cutoff = cutoff)
        #sort cluster
        idx_sort = np.argsort(np.array(self.unique_pair_idx_scell)[:,-1])
        
        if debug:
            print ("debug: clusters with radius sorted")
            print ([class_poscar.get_radius_cluster(self.unique_pair_latp[idx], self.primcell) for idx in idx_sort])
            print ("debug: indexed clusters with extra bond")
            print (class_poscar.get_bond_cluster(self.unique_pair_latp[0], self.primcell, image = 2, cutoff = 5.0))
            print (f"self.latp_list {self.latp_list}")
            print ( f"Max latp index: {np.ndarray.max(np.absolute(np.array(self.latp_list)))}" )
            
        # generate all lattice points within the cutoff
        latp_cutoff = []
        dim_latp = np.ndarray.max(np.absolute(np.array(self.latp_list))) + 1
        
        for m in range(self.primcell['natom']):
            for i in range(-1*dim_latp, dim_latp+1):
                for j in range(-1*dim_latp, dim_latp+1):
                    for k in range(-1*dim_latp, dim_latp+1):
                        for n in range(self.primcell['natom']):
                            pair_cluster = [ [[0,0,0],m], [[i,j,k],n] ]
                            if class_poscar.get_radius_cluster(pair_cluster, self.primcell) <= cutoff:
                                latp_cutoff.append([[i,j,k],n])
                            else:
                                continue
        
        # summary
        if debug_local:
            print (f"Number of lattice point within cutoff: {len(latp_cutoff)}")
            #print (f"latp_cutoff: {latp_cutoff}")
        # start index of 3body clusters
        index_3body = np.zeros(( self.primcell['natom'], len(latp_cutoff), len(latp_cutoff) ))
        self.unique_3body_cluster = []
        for idx in idx_sort: 
            pair_latp = self.unique_pair_latp[idx]
            if class_poscar.get_radius_cluster(pair_latp, self.primcell) <= cutoff:
                trip_latp = class_poscar.get_bond_cluster(pair_latp,
                                                          self.primcell,
                                                          image = dim_latp,
                                                          cutoff = cutoff)
                for trip in trip_latp:
                    ind_1 = trip[0][1]
                    ind_2 = latp_cutoff.index(trip[1]) 
                    ind_3 = latp_cutoff.index(trip[2])
                    if index_3body[ind_1, ind_2, ind_3] == 0:
                        self.unique_3body_cluster.append(trip)
                        for trip_sub in self.get_perm_sym(trip):
                            if debug:
                                print (trip_sub)
                                print (class_poscar.get_radius_cluster(trip_sub, self.primcell))
                            ind_1 = trip_sub[0][1]
                            ind_2 = latp_cutoff.index(trip_sub[1]) 
                            ind_3 = latp_cutoff.index(trip_sub[2])
                            index_3body[ind_1, ind_2, ind_3] = len(self.unique_3body_cluster)
        #summary
        print (f"Number of unique 3body clusters: {len(self.unique_3body_cluster)}")
        for trip in self.unique_3body_cluster:
            print(trip)
        

    def get_4body_cluster(self, cutoff = 3.0):
        debug_local = True
        try:
            self.unique_3body_cluster
        except AttributeError:
            self.get_3body_cluster(cutoff = cutoff)
            
        # generate all lattice points within the cutoff
        latp_cutoff = []
        dim_latp = np.ndarray.max(np.absolute(np.array(self.latp_list)))
        for m in range(self.primcell['natom']):
            for i in range(-1*dim_latp, dim_latp+1):
                for j in range(-1*dim_latp, dim_latp+1):
                    for k in range(-1*dim_latp, dim_latp+1):
                        for n in range(self.primcell['natom']):
                            pair_cluster = [ [[0,0,0],m], [[i,j,k],n] ]
                            if class_poscar.get_radius_cluster(pair_cluster, self.primcell) <= cutoff:
                                latp_cutoff.append([[i,j,k],n])
                            else:
                                continue
        
        # start index of 4body clusters
        index_4body = np.zeros(( self.primcell['natom'], len(latp_cutoff), len(latp_cutoff), len(latp_cutoff) ))
        self.unique_4body_cluster = []
        for idx in range(len(self.unique_3body_cluster)):
            trip_latp = self.unique_3body_cluster[idx]
            if class_poscar.get_radius_cluster(trip_latp, self.primcell) <= cutoff:
                quad_latp = class_poscar.get_bond_cluster(trip_latp,
                                                          self.primcell,
                                                          image = dim_latp, # timing
                                                          cutoff = cutoff)
                for quad in quad_latp:
                    ind_1 = quad[0][1]
                    ind_2 = latp_cutoff.index(quad[1]) 
                    ind_3 = latp_cutoff.index(quad[2])
                    ind_4 = latp_cutoff.index(quad[3])
                    if index_4body[ind_1, ind_2, ind_3, ind_4] == 0:
                        self.unique_4body_cluster.append(quad)
                        for quad_sub in self.get_perm_sym(quad):
                            ind_1 = quad[0][1]
                            ind_2 = latp_cutoff.index(quad_sub[1])
                            ind_3 = latp_cutoff.index(quad_sub[2])
                            ind_4 = latp_cutoff.index(quad_sub[3])
                            index_4body[ind_1, ind_2, ind_3, ind_4] = len(self.unique_4body_cluster)
        #summary
        print (f"Number of unique 4body clusters: {len(self.unique_4body_cluster)}")
        for quad in self.unique_4body_cluster:
            print (quad)
        
                
    @staticmethod
    def move_latp_origin(clus_in):
        '''
        move the first atom onto lattice point [0,0,0]
        '''
        clus_out = []
        for latp in clus_in:
            latp_lattice = (np.int_(latp[0])-np.int_(clus_in[0][0])).tolist()
            latp_atom = latp[1]
            clus_out.append([latp_lattice, latp_atom])
        return clus_out
    

    @staticmethod
    def equiv_by_trans(clus_1, clus_2, return_true_false = True):
        # generate all permutations
        clus_2_perm_list = []
        opera_perm_list = []
        all_opera_perm = list(permutations(list(range(len(clus_2)))))
        for idx, clus in enumerate(permutations(clus_2)):
            clus_tmp = class_poscar.move_latp_origin(clus)
            if clus_tmp not in clus_2_perm_list:
                clus_2_perm_list.append(clus_tmp)
                opera_perm_list.append(all_opera_perm[idx])
        # check if euqal
        tag_output = False
        tag_idx = -1000 # assume a very big but not valid number
        for idx, clus_tmp in enumerate(clus_2_perm_list):
            if clus_tmp == clus_1:
                tag_output = True
                tag_idx = idx
                break
        if return_true_false:
            return tag_output
        else:
            return [tag_output, opera_perm_list[tag_idx]]
        
        '''
        # old style
        if any( clus_tmp == clus_1 for clus_tmp in clus_2_perm_list ):
            if return_true_false:
                return True
        else:
            if return_true_false:
                return False
        '''


    def get_orbit_iso(self, clus_in, symprecin = 1e-4):
        '''
        get orbit clusters and the mapping symmetries
        to do list: permutation symmetry net yet enforced yet
        '''
        pos_list = []
        clus_all_rot = []
        # convert clus to latp
        for latp in clus_in:
            pos = (np.array(latp[0]) + np.array(self.primcell['atompos'][latp[1]])).tolist()
            pos_list.append(pos)
        # rotate all points
        pos_list_rot = [ (np.matmul(rmat, np.array(pos_list).T).T).tolist()
                         for rmat in self.symdataset['rotations']]
        # translate all points
        for id_rt in range(len(pos_list_rot)):
            for i in range(len(pos_list_rot[id_rt])):
                pos_list_rot[id_rt][i] = ( np.array(pos_list_rot[id_rt][i]) +
                                           np.array(self.symdataset['translations'][id_rt]) ).tolist()
        # convert latp to clus
        for item in pos_list_rot:
            clus_tmp = [ self.pos_to_latp(pos, self.primcell['atompos'], symprecin) for pos in item ]
            clus_all_rot.append(class_poscar.move_latp_origin(clus_tmp))
            
        # construct orbit and isotropy
        orbit_clus_list = []
        orbit_sym_list = []
        iso_sym_list = []
        for idx, clus in enumerate(clus_all_rot):
            # for orbit
            if any( class_poscar.equiv_by_trans(clus_tmp, clus) for clus_tmp in orbit_clus_list):
                pass
            else:
                orbit_clus_list.append(clus)
                orbit_sym_list.append([self.symset['crotations'][idx], self.symdataset['rotations'][idx]])
            # for isotropy
            if class_poscar.equiv_by_trans(clus_in, clus):
                iso_sym_list.append([self.symset['crotations'][idx], self.symdataset['rotations'][idx]])
        if True:
            print ("Number of all rotated clusters : ", len(clus_all_rot))
            print (clus_all_rot)
            print ("Number of cluters in orbit:  ", len(orbit_clus_list))
            print (orbit_clus_list)
            #print (orbit_sym_list)
            print ("Number of isotroy symmeties:  ", len(iso_sym_list))
            #print (iso_sym_list])
    
                                
    def get_perm_sym(self, clus_in, symprecin = 1e-4):
        '''
        return all clusters with permutations and space group symmetry operation
        '''
        debug_local = False
        clus_out_list = []
        
        # generate all permutations
        clus_perm_list = []
        for clus in permutations(clus_in):
            clus_tmp = class_poscar.move_latp_origin(clus)
            if clus_tmp not in clus_perm_list:
                clus_perm_list.append(clus_tmp)
                
        # generate all symmetrically equivalent ones
        for clus in clus_perm_list:
            pos_list = []
            for latp in clus:
                pos = (np.array(latp[0]) + np.array(self.primcell['atompos'][latp[1]])).tolist()
                pos_list.append(pos)
            pos_list_rot = [ (np.matmul(rmat, np.array(pos_list).T).T).tolist()
                             for rmat in self.symdataset['rotations']]
            for id_rt in range(len(pos_list_rot)):
                for i in range(len(pos_list_rot[id_rt])):
                    pos_list_rot[id_rt][i] = ( np.array(pos_list_rot[id_rt][i]) +
                                                np.array(self.symdataset['translations'][id_rt]) ).tolist()
            if debug_local:
                print (f"pos_lsit: {pos_list}")
                print (f"pos_list_rot[0]: {pos_list_rot[0]}")
            for item in pos_list_rot:
                clus_tmp = [ self.pos_to_latp(pos, self.primcell['atompos'], symprecin) for pos in item ]
                clus_out_list.append(class_poscar.move_latp_origin(clus_tmp))

        #summary
        return clus_out_list


    def get_perm_sym_id(self, clus_in, symprecin = 1e-4):
        '''                                                                                                        
        return all clusters with permutations and space group symmetry operation with sym ids
        '''
        debug_local = False
        clus_out_list = []
        idx_perm = list(permutations(list(range(len(clus_in)))))
        sym_operation = []

        # generate all permutations (just all permutations)
        clus_perm_list = []
        for clus in permutations(clus_in):
            clus_tmp = class_poscar.move_latp_origin(clus)
            if clus_tmp not in clus_perm_list:
                clus_perm_list.append(clus_tmp)                

        # generate all symmetrically equivalent ones
        for idx, clus in enumerate(clus_perm_list):
            pos_list = []
            for latp in clus:
                pos = (np.array(latp[0]) + np.array(self.primcell['atompos'][latp[1]])).tolist()
                pos_list.append(pos)
            pos_list_rot = [ (np.matmul(rmat, np.array(pos_list).T).T).tolist()
                             for rmat in self.symdataset['rotations']]
            # record symmetry operator
            if False:
                # in direct coordination
                sym_operation += [ [idx_perm[idx], rmat]  for rmat in self.symdataset['rotations']]
            if False:
                # in cartesian coordination
                sym_operation += [ [idx_perm[idx], rmat]  for rmat in self.symset['crotations']]
            if True:
                # both cartesian and direct coordinations
                sym_operation += [ [idx_perm[idx], [rmat, self.symdataset['rotations'][idx_peach]]]
                                   for idx_peach, rmat in enumerate(self.symset['crotations'])]
            
            for id_rt in range(len(pos_list_rot)):
                for i in range(len(pos_list_rot[id_rt])):
                    pos_list_rot[id_rt][i] = ( np.array(pos_list_rot[id_rt][i]) +
                                                np.array(self.symdataset['translations'][id_rt]) ).tolist()
            if debug_local:
                print (f"pos_lsit: {pos_list}")
                print (f"pos_list_rot[0]: {pos_list_rot[0]}")
            for item in pos_list_rot:
                clus_tmp = [ self.pos_to_latp(pos, self.primcell['atompos'], symprecin) for pos in item ]
                clus_out_list.append(class_poscar.move_latp_origin(clus_tmp))
                
        if debug_local:
            print ("symmetry operators: ", sym_operation)
            print ("number of symmetry operators: ", len(sym_operation))
            
        #filter with unique clusters
        unique_clus_out_list = []
        unique_sym_out_list = []
        for clus in clus_out_list:
            if not clus in unique_clus_out_list:
                unique_clus_out_list.append(clus)
                unique_sym_out_list.append(sym_operation[clus_out_list.index(clus)])
                
        #summary
        return [unique_clus_out_list, unique_sym_out_list]
    
                    
                                            
    def pair_sym_from_scell(self, symprec = 1e-4, cutoff = 10.0):
        '''
        properties of supercell
        usage: extract symmetrically distinct pairs of atoms
        assumption: the original lattice point is centered at [0,0,0] point        
        '''
        
        #roundint = int(-1*math.log(symprec,10))
        self.map_to_prim(symprec)
        self.map_latp_list = [ [latp, self.map_list[i][1] ] for i, latp in enumerate(self.latp_list)]
        
        # check if cell centered at [0,0,0]
        if self.latp_list.count([0,0,0]) != self.primcell['natom']:
            print("Warning: [0,0,0] is not centerd, exit!")
            print (f"scmat: {self.scmat}")
            print (self.latp_list)
            print (len(self.latp_list))
            sys.exit()
        if debug:
            print (f"scmat: {self.scmat}")
            print (f"map_list: {self.map_list}")
            print (f"latp_list: {self.latp_list}")
            print (f"map_latp_list: {self.map_latp_list}")
        self.unique_pair_idx_scell, self.unique_pair_latp = \
        self.get_unique_2body(
            scmat = self.scmat, 
            map_list = self.map_list, 
            latp_list = self.latp_list, 
            map_latp_list = self.map_latp_list, 
            primcell = self.primcell,
            cutoff =  cutoff,
            symprecin = symprec,
            periodicity = True
        )
        if debug:
            print ("debug: unique_pair_idx_scell")
            print (np.array(self.unique_pair_idx_scell))
            idx_sort = np.argsort(np.array(self.unique_pair_idx_scell)[:,-1])
            print ([self.unique_pair_idx_scell[idx][-1] for idx in idx_sort])
            print ("debug: unique_pair_latp")
            print ([self.unique_pair_latp[idx] for idx in idx_sort])
            print ("debug get_radius_cluster")
            print ([class_poscar.get_radius_cluster(self.unique_pair_latp[idx], self.primcell) for idx in idx_sort])
            print (class_poscar.get_bond_cluster(self.unique_pair_latp[0], self.primcell, image = 2, cutoff = 5.0))
        
        '''
        # duplicate code for self.get_unique_2body
        for idx_p, pos_p in enumerate(self.primcell['atompos']):
            for idx_s, lat_s in enumerate(self.latp_list):
                pos_tmp = np.array(self.primcell['atompos'][self.map_list[idx_s][1]])
                pair_pos = np.array([pos_p, (np.array(lat_s)+pos_tmp).tolist()])
                [pair_pos_min, dist_min] = self.find_min_dis_pos(pair_pos[0], pair_pos[1], \
                                                                     self.scmat, self.primcell['latvec'])
                pair_latp = [self.pos_to_latp(pos, self.primcell['atompos'], symprec) for pos in pair_pos_min]
                pair_idx = self.latp_to_idx(pair_latp, self.primcell['natom'], scancell)
                if (pair_map_idx[pair_idx[0][0]][pair_idx[0][1]] == 0) and \
                   (pair_map_idx[pair_idx[1][0]][pair_idx[1][1]] == 0):
                    unique_count += 1
                    unique_pair_latp.append(pair_latp)
                    unique_pair_idx_scell.append([self.map_latp_list.index([[0,0,0],idx_p]), idx_s, dist_min])
                    pair_map_idx[pair_idx[0][0]][pair_idx[0][1]] = unique_count
                    pair_map_idx[pair_idx[1][0]][pair_idx[1][1]] = unique_count                        
                    if debug:
                        print ("pair_latp")
                        print (pair_latp)
                        print ("pair_idx")
                        print (pair_idx)
                        print ("dist_min")
                        print (dist_min)
                        print ("\n")
                    # generate symmetrically equivalent pairs of atoms
                    pair_pos_rot = [ (np.matmul(rmat, pair_pos.T).T).tolist() for rmat in dataset['rotations']]
                    for pair in pair_pos_rot:
                        pair_latp_rot = [ self.pos_to_latp(pos, self.primcell['atompos'], symprec) for pos in pair]
                        pair_idx_rot = self.latp_to_idx(pair_latp_rot, self.primcell['natom'], scancell)
                        pair_map_idx[pair_idx_rot[0][0]][pair_idx_rot[0][1]] = unique_count
                        pair_map_idx[pair_idx_rot[1][0]][pair_idx_rot[1][1]] = unique_count                               
        if debug:
            print ("max index in pair_map_idx")
            print (np.amax(pair_map_idx))
            print ("unique_count")
            print (unique_count)
            print ("unique_pair_latp")
            print (unique_pair_latp)
            print ("sorted unique_pair_idx_scell")
            print (sorted(unique_pair_idx_scell, key=lambda x:x[2]))
        return unique_pair_idx_scell, unique_pair_latp
        '''


    @staticmethod
    def latp_to_idx(pair_latp, pnatom, scancell):
        '''
        usage: return pair of indexes       
        assumption:         
        to do: generalize this function to arbitrary multi-body clusters
        '''
        pair_idx = []
        latp1 = (np.array([0,0,0])+scancell).astype(int).tolist()
        latp2 = (np.array(pair_latp[1][0])-np.array(pair_latp[0][0])+scancell).astype(int).tolist()
        atom1 =pair_latp[0][1]
        atom2 =pair_latp[1][1]
        idx1 = atom1 #poscar.latp_to_num(latp1,scancell)*pnatom + atom1
        idx2 = class_poscar.latp_to_num(latp2,scancell)*pnatom + atom2
        pair_idx.append([idx1, idx2])

        latp1 = (np.array([0,0,0])+scancell).astype(int).tolist()
        latp2 = (np.array(pair_latp[0][0])-np.array(pair_latp[1][0])+scancell).astype(int).tolist()
        atom1 =pair_latp[1][1]
        atom2 =pair_latp[0][1]
        idx1 = atom1 #poscar.latp_to_num(latp1,scancell)*pnatom + atom1
        idx2 = class_poscar.latp_to_num(latp2,scancell)*pnatom + atom2
        pair_idx.append([idx1, idx2])
        return pair_idx

    
    @staticmethod
    def latp_to_num(latp, scancell):
        '''
        usage: return index of the lattice point 
        '''
        return latp[0]+latp[1]*(2*scancell+1)+latp[2]*(2*scancell+1)**2

    @staticmethod
    def latp_to_num_new(latp_in,scancell,pnatom):
        [latp, id_atom] = latp_in
        return (latp[0]+latp[1]*(2*scancell+1)+latp[2]*(2*scancell+1)**2)*pnatom + id_atom


#    def round_atompos_prim(self, atompos_prim_in, symprec):
#        roundint = int(-1*math.log(symprec,10))
#        atompos_latp = [np.array(list(map(math.floor, apple))).astype(int) for apple in atompos_prim_in]
#        return atompos_prim_in - atompos_latp
    

    @staticmethod
    def pos_to_latp(posatom, atompos_prim, symprec = 1e-4):
        '''
        usage: position to lattice point + atomic index
        assumption: based on primitive lattice vector
        '''
        
        def round_atompos_prim(atompos_prim_in, symprec):
            # round the atompos_prim --> wathch out new lattice point! 
            roundint = int(-1*math.log(symprec,10))
            atompos_prim_in = np.around(np.array(atompos_prim_in), roundint)
            atompos_latp = [np.array(list(map(math.floor, apple))).astype(int) for apple in atompos_prim_in]
            #print (atompos_latp)
            return (atompos_prim_in - atompos_latp )
        
        posatom_round = np.array([ round(num, int(-1*math.log(symprec, 10))) for num in posatom])
        latp = np.array(list(map(math.floor, posatom_round))).astype(int)
        pos = posatom_round - latp
        # round the atompos_prim --> wathch out new lattice point!   
        atompos_prim = round_atompos_prim(np.array(atompos_prim), symprec)
        
        pos_diff = atompos_prim - pos
        pos_diff_norm = list(map(np.linalg.norm, pos_diff))
        if min(pos_diff_norm) < symprec *10:
            idx = pos_diff_norm.index(min(pos_diff_norm))
        else:
            print ("Warning: atomic positions not match, exit!")
            if False:
                print (f"debug: posatom_round: {posatom_round}")
                print (f"debug: latp: {latp}")
                print (f"debug: pos: {pos}")
                print (f"debug: pos_diff_norm: {pos_diff_norm}")
            sys.exit()
        return [latp.tolist(), idx]


    @staticmethod
    def find_min_latp(latp, scmat, image = 1):
        # !!!extreme attention!!!
        # always assume the distance between the origin [0,0,0] and latp
        dist_min = 10**3
        latp_out = latp
        for ix in range(-1*image,image+1):
            for iy in range(-1*image,image+1):
                for iz in range(-1*image,image+1):
                    latp_diff = np.array(latp) + ix*np.array(scmat)[0,:] + \
                        iy*np.array(scmat)[1,:] + \
                        iz*np.array(scmat)[2,:]
                    dist_latp = np.linalg.norm(latp_diff)
                    if dist_latp < dist_min:
                        dist_min = dist_latp
                        latp_out = latp_diff.tolist()
        return latp_out
    

    @staticmethod
    def find_min_dis_pos(pos1, pos2, scmat, latvec, image = 1):
        '''
        usage: return image position with minimum distance for pairs
        '''
        # check if periodic boundary condition enforce equivalence
        #scmat = [[10,0,0],[0,10,0],[0,0,10]]    
        dist_min = 10**2
        for ix in range(-1*image,image+1):
            for iy in range(-1*image,image+1):
                for iz in range(-1*image,image+1):
                    # atomic position convention
                    pos_diff = np.array(pos2) - np.array(pos1)
                    # lattice vector convention
                    #pos_diff = np.rint(pos2) - np.rint(pos1)
                    pos_diff += ix*np.array(scmat)[0,:] + \
                                iy*np.array(scmat)[1,:] + \
                                iz*np.array(scmat)[2,:]
                    dist_tmp = np.linalg.norm(
                        np.matmul(pos_diff, np.array(latvec))
                    )
                    if dist_tmp < dist_min:
                        dist_min = dist_tmp
                        pair_pos = [pos1.tolist(), (pos_diff + pos1).tolist()]
        return [pair_pos, dist_min]


    def get_pair_orbit_noinv(self, symprec = 1e-4):
        '''
        1. generate all up-diagonal pairs of atoms within a supercell
        2. elliminate all the translationally equivalent clusters
        3. no reflection/inversion symmetry is applied
        output:
        self.orbit_noinv_list: list of clusters in orbit
        self.orbit_idx_map: list of index of cluster in orbit for lower triangle part of scell atom matrix
        '''
        if not hasattr(self, 'latp_list'):
            self.map_to_prim(symprec)
            print ("Regenerate the mapping")
        else:
            print ("Mapping already exists")

        debug_local = False
        
        self.orbit_noinv_list = []
        # -1 sometimes could create some confusion
        self.orbit_idx_map = np.array([-1]*self.natom*self.natom).reshape(self.natom, self.natom)
        orbit_count = 0
        
        # loop over the elements in the lower triangle part of a matrix
        for iatom in range(self.natom):
        #for iatom in range(20): # debug purpose
            for jatom in range(iatom+1):
                latp_iatom = self.latp_list[iatom]
                latp_jatom = self.latp_list[jatom]            
                type_iatom = self.map_list[iatom][1]
                type_jatom = self.map_list[jatom][1]
                pos_iatom = self.primcell['atompos'][type_iatom]
                pos_jatom = self.primcell['atompos'][type_jatom]
                latp_move_jatom = (np.int_(latp_jatom)-np.int_(latp_iatom)).tolist()
                #clus_pair = [ [[0,0,0], type_iatom], [latp_move_jatom, type_jatom]]
                pos1 = np.array(latp_iatom)+np.array(pos_iatom)
                pos2 = np.array(latp_jatom)+np.array(pos_jatom)
                if debug_local:
                    print ("-"*20)
                    print ("pair atompos before PBC mapping: ", [pos1.tolist(), pos2.tolist()])
                [pair_pos, dist_min] = class_poscar.find_min_dis_pos(pos1, pos2,
                                                                     self.scmat,
                                                                     self.primcell['latvec'],
                                                                     image = 1)
                
                clus_in = [self.pos_to_latp(pos, self.primcell['atompos'], symprec) for pos in pair_pos]
                clus_pair = self.move_latp_origin(clus_in)
                if debug_local:
                    print ("pair atompos after PBC mapping: ", clus_in, dist_min)
                    print ("pair atompos after PBC mapping and setting origin: ", clus_pair)
                if clus_pair not in self.orbit_noinv_list:
                    orbit_count += 1
                    self.orbit_idx_map[iatom, jatom] = orbit_count - 1
                    self.orbit_noinv_list.append(clus_pair)
                else:
                    self.orbit_idx_map[iatom, jatom] = self.orbit_noinv_list.index(clus_pair)
        # summary
        if debug_local:
            print (self.orbit_idx_map)
            print (f"Number of clusters in orbitals without inversion: {len(self.orbit_noinv_list)}")

        # map to unique clusters
        _ = self.get_sym_prim(symprec)
        
        # test output symmetry operations
        if False:
            all_sym_clus = self.get_perm_sym_id(self.orbit_noinv_list[10])
            print ("sym clusters: ", all_sym_clus)
            print ("total number: ", len(all_sym_clus))
            sys.exit()
        
        '''
        unique_idx_map: index for each cluster in orbit to unique cluster
        unique_list:  list of unique cluster
        '''
        self.unique_idx_map = [-1] * len(self.orbit_noinv_list)
        self.unique_sym_map = [[] for i in range(len(self.orbit_noinv_list))]
        
        for idx, clus in enumerate(self.orbit_noinv_list):
            [all_sym_clus, all_sym_operation] = self.get_perm_sym_id(clus)
            
            for idx_sym, clus_tmp in enumerate(all_sym_clus):
                if clus_tmp in self.orbit_noinv_list:
                    idx_tmp = self.orbit_noinv_list.index(clus_tmp)
                    if self.unique_idx_map[idx_tmp] == -1:
                        self.unique_idx_map[idx_tmp] = idx
                        self.unique_sym_map[idx_tmp] = all_sym_operation[idx_sym]
                    #print ("index of cluster in orbit: ", self.orbit_noinv_list.index(clus_tmp))
            if debug_local:
                print ("selected cluster: ", clus)
                print ("Symmetry generated clusters: ", all_sym_clus)
                print ("Number of symmetry generated clusters: ", len(all_sym_clus))
        
        # collect unique clusters
        self.unique_list = [self.orbit_noinv_list[idx] for idx in list(set(self.unique_idx_map))]

        # raise error if any element in self.unique_idx_map == -1 --> not mapped
        if any(item == -1 for item in self.unique_idx_map):
            raise ValueError("Clusters in orbit not completed maped, exit!")
        
        #summary
        if debug_local:
            idx_test = 100
            print ("list of orbit clusters: ", self.orbit_noinv_list[idx_test])
            #print ("list of unique clusters: ", self.unique_list[self.unique_idx_map[idx_test]])
            print ("list of unique clusters: ", self.orbit_noinv_list[self.unique_idx_map[idx_test]])
            #print ("number of unique clusters: ", len(self.unique_list))
            print ("unique_idx_map: ", self.unique_idx_map[idx_test])
            print ("unique_sym_map: ", self.unique_sym_map[idx_test])
            print ("unique_idx_map: ", self.unique_idx_map)
            
        
    
    def map_to_prim(self, symprec = 1e-4, read_prim = ""):
        '''
        usage: build a map between supercell and primtive cell
        input: poscar
        output: map from primitive to supercell: (lattice point, atomic index)
        assumption: assume both cell are in direct coordination
        warning: test for unregular scmat
        todo: raise error message
        '''
        
        self.get_prim(symprec, read_prim, is_no_idealize = True)
        #self.get_prim(symprec, read_prim="./POSCAR-prim-yixia", is_no_idealize = True)
        latvec_prim = self.primcell['latvec']
        latvec_scell = self.output['latvec']
        if debug:
            print ("debug: "+"-"*80+"map_to_prim")
            print ("debug: latvec_prim")
            print (latvec_prim)
            print ("debug: latvec_scell")
            print (latvec_scell)
            
        # round the atompos_prim --> wathch out new lattice point!
        roundint = int(-1*math.log(symprec,10))
        atompos_prim = np.around(np.array(self.primcell['atompos']),roundint)
        atompos_latp = [np.array(list(map(math.floor, apple))).astype(int) for apple in atompos_prim]
        atompos_prim = atompos_prim - atompos_latp
        
        scmat = np.matmul( np.array(self.output['latvec']),\
                           np.linalg.inv(np.array(self.primcell['latvec'])) )
        scmat = np.rint(scmat).astype(int).tolist()
        sel_latp = self.latp_from_scmat(scmat)
        if debug:
            print ("debug: scmat")
            print (scmat)
            print ("debug: sel_latp")
            print (sel_latp)            
            print (len(sel_latp))
            print (len(atompos_prim))
        
        pos_scell = self.output['atompos']
        #pos from scell transformed in pcell
        pos_sprim = np.matmul(
        np.array(pos_scell),        
        np.matmul(
            np.array(latvec_scell),
            np.linalg.inv(np.array(latvec_prim))
        )
        )
        # define map from supercell to primitive cell
        self.map_list = []
        self.latp_list = []
        self.scmat = scmat
        for count, posatom in enumerate(pos_sprim):         
            posatom_round = np.array([ round(num, int(-1*math.log(symprec, 10))) for num in posatom])
            latp = np.array(list(map(math.floor, posatom_round)))
            pos = posatom_round - latp
            pos_diff = atompos_prim - pos
            pos_diff_norm = list(map(np.linalg.norm, pos_diff))
            if debug:
                print ("debug:"+"-"*40)
                print (f"debug: count: {count}")
                print (f"debug: posatom: {posatom}")
                print (f"debug: posatom_round: {posatom_round}")
                print (f"debug: pos: {pos}")
                print (f"debug: latp: {latp}")
                print (f"debug: min(pos_diff_norm): {min(pos_diff_norm)}")
                #print (atompos_prim)
                #print (atompos_latp)
            if min(pos_diff_norm) < symprec *10:
                idx = pos_diff_norm.index(min(pos_diff_norm))
            else:
                print ("Warning: atomic positions not match, exit!")
                print (pos_diff_norm)
                sys.exit()
            
            self.map_list.append([count, idx])
            # yixia 
            if True:
                # assume periodic image based on lattice vector convention
                self.latp_list.append(self.find_min_latp(latp.tolist(), self.scmat))
            else:
                # assume no periodic image
                self.latp_list.append(latp.tolist())
            
        if debug:
            print ('debug: map_list')
            print (self.map_list)
            print ('debug: latp_list')
            print (self.latp_list)
            print ('debug: count [0,0,0]')
            print (self.latp_list.count([0,0,0]))
            for idx, latp in enumerate(self.latp_list):
                if self.latp_list[idx] == [0,0,0]:
                    print ([self.latp_list[idx], self.map_list[idx][1]])
        # sorting lattice point
        set_from_scmat = sorted(set(map(tuple, sel_latp)), reverse = False)
        set_from_calc = sorted(set(map(tuple, self.latp_list)), reverse = False)
        if set_from_scmat ==  set_from_calc and \
           len(set_from_calc) == int(self.output['natom'] / self.primcell['natom']):
            pass
        else:
            print ("Warning: two lattice point are different!")
            if debug:
                print ("--"*40)
                print ("debug: set_from_scmat")
                print (set_from_scmat)
                print ("debug: set_from_calc")
                print (set_from_calc)
                print ("--"*40)
                print ("debug: count [0 0 0]")
                print (self.latp_list.count([0,0,0]))


    @staticmethod
    def map_prim_sup(pposcar, sposcar):
        pass


    @staticmethod
    def order_latp_old(latp_in):
        max_num = np.amax(np.array(latp_in))
        min_num = np.amin(np.array(latp_in))
        div_num = (max_num - min_num) + 1
        id_list = [ np.dot(np.array(latp), np.array([0, div_num, div_num**2]))for latp in latp_in ]
        sort_index = np.argsort(id_list)
        latp_out = [ latp_in[idx] for idx in sort_index ]
        return latp_out


    @staticmethod
    def latp_from_scmat(scmat):
        '''
        usage: generate lattice porint from supercell matrix
        input: matrix for scell
        output: list of lattice point
        assumption: hard-coded set for 3D space                    
        '''
        dim = 3
        latp_set = [[1], [2], [3], [1, 2], [1, 3], [2, 3], [1, 2, 3]]
        # generate all lattice translations according to scell
        tot_set = []
        for i_set in latp_set:
            tmp_latp = [0]*dim
            for idx in i_set:
                #print (np.array(scmat[idx-1]))
                #print (np.array(tmp_latp))
                tmp_latp = np.array(tmp_latp) + np.array(scmat[idx-1])
                tmp_latp = list(map(int, tmp_latp.tolist()))
            tot_set.append(tmp_latp)
        tot_set.append([0]*dim)
        tot_set = np.array(tot_set).T
        tot_set = tot_set.tolist()
        # generate outer-box for lattice vectors
        out_set = []
        for i_set in tot_set:
            out_set.append(range(min(i_set), max(i_set)+1))
        # generate all possible combinations
        all_set = []
        for i in itertools.product(out_set[0], out_set[1], out_set[2]):
            all_set.append(list(i))
        # select lattice point according to distance
        invscmat = np.linalg.inv(np.array(scmat).T)
        sel_latp = []
        for iset in all_set:
            distance =  np.dot(invscmat, np.array(iset).T)
            if min(distance) >= 0.0 and max(distance) < 1.0:
                sel_latp.append(iset)
        # return lattice point
        if int(round(abs(np.linalg.det(np.array(scmat))),1)) == len(sel_latp):
            #if using static method within calss then refer to class name
            return class_poscar.order_latp(sel_latp)
        else:
            return class_poscar.order_latp(sel_latp)  


    @staticmethod
    def order_latp(latp_in):
        max_num = np.amax(np.array(latp_in))
        min_num = np.amin(np.array(latp_in))
        div_num = (max_num - min_num) + 1
        id_list = [ np.dot(np.array(latp), np.array([0, div_num, div_num**2]))for latp in latp_in ]
        sort_index = np.argsort(id_list)
        latp_out = [ latp_in[idx] for idx in sort_index ]
        return latp_out    
    
        
    def vacancy_single(self, elm_list, symprec = 1e-4):
        '''
        usage: Creating single vacancy for symmetrically equivalent atoms
        input: poscar, element list, and symprec
        output: poscar with vacancy written on disk
        assumption: 
        '''
        dataset = self.get_sym(symprec)
        eqv_list = dataset['equivalent_atoms']
        print (eqv_list)
        count = 0
        str_map = ""
        for elm in elm_list:
            print ("Creating single vacancy for: " + " ".join(elm_list))
            idx_elm = self.output['spe'].index(elm)
            sub_eqv_list = [] # append list
            for idx, spe_atom in enumerate(self.output['spe_all']):
                if spe_atom == elm:
                    if eqv_list[idx] not in sub_eqv_list:                        
                        sub_eqv_list.append(int(eqv_list[idx]))
            for idx in sub_eqv_list:                
                pos_output = copy.deepcopy(self.output)
                del pos_output['atompos'][idx]
                del pos_output['spe_all'][idx]
                pos_output['cspe'][idx_elm] -= 1
                if pos_output['cspe'][idx_elm] == 0:
                    del pos_output['spe'][idx_elm]
                    del pos_output['cspe'][idx_elm]
                self.write_poscar_dp(pos_output, "POSCAR-vac-" + elm + str(idx))

                if True:
                    count += 1
                    self.write_poscar_dp(pos_output, "icet-POSCAR-" + str(count))
                    str_map += "POSCAR-vac-" + elm + str(idx) + ": " + "icet-POSCAR-" + str(count) + "\n"
        with open("str_map.dat", "w") as file:
            file.write(str_map)
                        

    @staticmethod
    def image_chop(listin):
        '''
        untested
        '''
        listout = []
        for pos in listin:
            while pos < 0.0 or pos >= 1.0:
                if pos >= 1.0:
                    pos -= 1.0
                if pos < 0.0:
                    pos += 1.0
            listout.append(pos)            
        return listout

        
    @staticmethod
    def find_dist(pos1, pos2, latvec, image = 1):
        '''
        input: list 
        output: minimum distance between pos1 and pos2
                in consideration of periodic images
        assumption: pos1 are in directional coordination
        '''
        dist_min = 10**2
        for ix in range(-1*image,image+1):
            for iy in range(-1*image,image+1):
                for iz in range(-1*image,image+1):
                    pos_diff = np.array(pos2) - np.array(pos1)
                    pos_diff += np.array([ix,iy,iz])
                    dist_tmp = np.linalg.norm(
                        np.matmul(pos_diff,np.array(latvec))
                    )
                    if dist_tmp < dist_min:
                        dist_min = dist_tmp
        return dist_min
        
        
    def get_pair(self, el_pair, bader = False):
        '''
        input: pair of elements ['O','O']
        input: 
        '''
        #if bader:
        #    sys.path.append('/home/yxb4830/script/trans-cfg//vasp_io/')
        #    import bader_chg
        #convert coordinate from C to D
        if self.output['cord'] == "C":
            self.dir_to_car(reverse = True)
        pos_pair1 = []
        pos_pair2 = []
        for idx, el in enumerate(self.output['spe_all']):
            if el == el_pair[0]:
                pos_pair1.append(self.output['atompos'][idx])
            if el == el_pair[1]:
                pos_pair2.append(self.output['atompos'][idx])
        print (pos_pair1)
        #print (pos_pair2)
        pair_name = []
        pair_dist = []
        pair_info = {}
        for iatom, ipos in enumerate(pos_pair1):
            for jatom, jpos in enumerate(pos_pair2):
                dist_name = el_pair[0]+"_"+str(iatom+1)+"-"+el_pair[1]+"_"+str(jatom+1)
                dist_name_dup = el_pair[1]+"_"+str(jatom+1)+"-"+el_pair[0]+"_"+str(iatom+1)
                
                dist_min = self.find_dist(ipos, jpos, self.output['latvec'], image = 2)
                if dist_min > 1**-2:
                    pair_dist.append(dist_min)
                    pair_name.append(dist_name)
                    if dist_name not in pair_info and dist_name_dup not in pair_info:
                        if not bader: 
                            pair_info[dist_name] = [dist_min]
                        else:
                            sys.path.append('/home/yxb4830/script/trans-cfg//vasp_io/')
                            import bader_chg
                            bader_vasp = bader_chg.baderchg("./ACF.dat", self.output['spe_all'])
                            idone = self.output['atompos'].index(ipos)
                            idtwo = self.output['atompos'].index(jpos)
                            bader_chg = (bader_vasp.chg_array[idone]+bader_vasp.chg_array[idtwo])/2
                            pair_info[dist_name] = [dist_min, bader_chg]
        pair_info_sort = dict(sorted(pair_info.items(), key = lambda item: item[1]))
        print (pair_info_sort)
        print (len(pair_info_sort))
        with open("./pair-"+el_pair[0]+"-"+el_pair[1]+".dat","w") as file:
            for key in pair_info_sort:
                #file.write(key+"  "+'{:16.12f}'.format(pair_info_sort[key])+"\n")
                strout = "  ".join(map(str,pair_info_sort[key]))
                file.write(key+"  "+strout + "\n")
                
        

#-------------------------------------------------------------
#-------------------------------------------------------------
#-------------------------------------------------------------
    
if __name__ == "__main__":

    test_flag = {
        "remove_vac" : False,
        "fix_potcar": False,
        "fix_incar": False,
        "write_poscar": False
    }
    
    #initialize the random number
    random.seed(13579)
    debug = False

    for i in range(1):

        #initialize the object
        #struct = class_poscar("./POSCAR-prim")
        #struct.print_info()
        #struct.dir_to_car(reverse = False)
        #struct.disp_atom_random(disp_range = [0.01, 0.05])
        #struct.get_sym(symprec = 1e-4)
        #struct.get_sym_prim(symprec = 1e-4)
        #struct.print_poscar()    
        #struct.print_poscar_org()                                                                          
        # create single vacancy
        #struct.vacancy_single(['O'])
        
        # geneate lattice point from give scell
        #tmat = [-4, 4, 4, 4, -4, 4, 4, 4, -4]
        #struct.latp_from_scmat(np.reshape(tmat,(3,3)))

        #struct.refine_cell(symprec = 1e-4, to_prim = False)
        #struct.get_prim(symprec = 1e-4)
        #struct.map_to_prim(symprec = 1e-4)
        
        # extract symmetrically distinctive pair of atoms from scell
        struct = class_poscar("./SPOSCAR") 
        struct.pair_sym_from_scell(symprec = 1e-4)
        #struct.pair_sym_from_scmat(symprec = 1e-4)
        #tmat = [2,0,0,0,2,0,0,0,2]
        #struct.pair_sym_from_scmat(scmatin = np.reshape(tmat,(3,3)).tolist(), symprec = 1e-4 ) 

        # statistical analysis of pair of elemental atoms
        #struct = class_poscar("./POSCAR")  
        #struct.get_pair(['O','O'], bader = True)

        
        if test_flag["remove_vac"]:
            struct.remove_vac()
        
        if test_flag["write_poscar"]:
            struct.write_poscar("./POSCAR-"+str(i))
    
        if test_flag["fix_potcar"]:
            pathin_potcar = "/projects/p31102/Li-rich/O2-stack/vasp_input/"
            struct.fix_potcar(pathin_potcar, "./POTCAR-"+str(i))
        
        if test_flag["fix_incar"]:
            pathin_incar = pathin_potcar+"INCAR-rel"
            struct.fix_incar(pathin_incar,"./INCAR-"+str(i))
