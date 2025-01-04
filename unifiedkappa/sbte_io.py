import numpy as np
import sys
import os

def update_control_file(input_file, output_file, allocations, crystal, parameters, flags):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    updated_lines = []
    current_section = None

    def update_line(line, dictionary):
        key, value = line.strip().split('=', 1)
        key = key.strip()
        if key in dictionary:
            return f"        {key}={dictionary[key]}\n"
        else:
            return line

    for line in lines:
        if line.startswith('&'):
            current_section = line.strip()[1:]
            updated_lines.append(line)
            #print (current_section)
        elif current_section is not None:
            if current_section == 'allocations':
                #print ("update allocations")
                updated_lines.append(update_line(line, allocations))
            elif current_section == 'crystal':
                updated_lines.append(update_line(line, crystal))
            elif current_section == 'parameters':
                updated_lines.append(update_line(line, parameters))
            elif current_section == 'flags':
                updated_lines.append(update_line(line, flags))
            else:
                updated_lines.append(line)
        else:
            updated_lines.append(line)

    with open(output_file, 'w') as file:
        file.writelines(updated_lines)

class class_shengbte:
    """
    Class for handling ShengBTE input and output
    """
    def __init__(self, path_dir = "",
                 is_kappa = True,
                 is_phase = True,
                 is_rates3rd = True,
                 override_error = False):
        self.path_dir = path_dir
        self.values = {}
        self.errormsg = True
        if self.path_dir != "":
            if not override_error:
                self.get_errormsg("BTE.out")
            self.get_control("CONTROL")
            if is_kappa and self.values['onlyharmonic'] == False:
                self.get_kappa()
            if is_phase and self.values['onlyharmonic'] == False:
                self.get_phase()
            if is_rates3rd and self.values['onlyharmonic'] == False:
                self.get_rates3rd()
            
    def get_errormsg(self, fname):
        path_fname = os.path.join(self.path_dir, fname)
        print(f"info: {path_fname}")
        if os.path.isfile(path_fname):
            with open(path_fname, "r") as file:
                str_file = file.read()
            if "normal exit" in str_file:
                self.errormsg = False
            if "Info: normal exit" in str_file:
                self.errormsg = False
            if "Info: onlyharmonic=.true., stopping here" in str_file:
                self.errormsg = False
            #else:
            #    self.errormsg = True
            #    print ("found error")
        else:
            self.errormsg = True
        
        if self.errormsg == True:
            print ("info: Error: corrupted calculations ----> exit!")
            sys.exit()
        else:
            print ("info: Normal ---> continue!")
            
        
    def get_control(self, fname):
        try:
            with open(os.path.join(self.path_dir,fname), "r") as file:
                lines = file.readlines()
        except IOError:
            print ("CONTROL file does not exist!")
        self.values['natom'] = 0
        self.values['fourthrate'] = False
        self.values['onlyharmonic'] = True
        # 
        for idx, line in enumerate(lines):
            if "positions" in line:
                self.values['natom'] += 1
            if "T" in  line.split("=")[0] and "T_" not in  line.split("=")[0] :
                self.values['temp'] = [int(line.split("=")[-1].strip())]
            if "T_min" in line.split("=")[0]:
                self.values['t_min'] = int(line.split("=")[-1].strip())
            if "T_max" in line.split("=")[0]:
                self.values['t_max'] = int(line.split("=")[-1].strip())
            if "T_step" in line.split("=")[0]:
                self.values['t_step'] = int(line.split("=")[-1].strip())
            if "fourthrate" in line.split("=")[0] and ".TRUE." in line.split("=")[1]:
                self.values['fourthrate'] = True
            if "onlyharmonic" in line.split("=")[0] and ".FALSE." in line.split("=")[1]:
                self.values['onlyharmonic'] = False
        if 't_step' in self.values.keys():
            self.values['temp'] = []
            tmin = self.values['t_min']
            tmax = self.values['t_max']
            tstep = self.values['t_step']
            for i in range(int((tmax-tmin)/tstep)):
                self.values['temp'].append(tmin+i*tstep)
            self.values['temp'].append(tmax)
            

    def get_qptall(self):
        fname_fl = os.path.join(self.path_dir,"BTE.qpoints_full")
        fname_ir = os.path.join(self.path_dir,"BTE.qpoints")
        try:
            #------------
            self.values['qptfull'] = []
            with open(fname_fl, "r") as file:
                lines = file.readlines()
            for line in lines:
                self.values['qptfull'].append(int(line.split()[1]))
            #------------
            self.values['qptir'] = []
            with open(fname_ir, "r") as file:
                lines = file.readlines()
            for line in lines:
                self.values['qptir'].append(list(map(float, line.split()[3:])))
        except IOError:
            print ("BTE.qpoints_full file does not exist!")
            print ("BTE.qpoints file does not exist!")
            

    def get_rates3rd(self):
        self.values['rates3rd_ir'] = {}
        self.values['rates3rd_fl'] = {}
        class_shengbte.get_qptall(self)
        for temp in self.values['temp']:
            fname = os.path.join(self.path_dir, "T"+str(temp)+"K/"+"BTE.w")
            try:                
                data_tmp = np.loadtxt(fname)
                natom = self.values['natom']
                nqpt_ir = len(self.values['qptir'])
                nqpt_fl = len(self.values['qptfull'])
                if False:
                    print (nqpt_ir)
                    print (nqpt_fl)
                self.values['rates3rd_ir'][str(temp)] = np.zeros((nqpt_ir, 3*natom))
                self.values['rates3rd_fl'][str(temp)] = np.zeros((nqpt_fl, 3*natom))
                for iq in range(nqpt_ir):
                    for ib in range(3*natom):
                        self.values['rates3rd_ir'][str(temp)][iq][ib] = data_tmp[ib*nqpt_ir+iq, 1]
                for iq in range(nqpt_fl):
                    index = self.values['qptfull'][iq] - 1
                    self.values['rates3rd_fl'][str(temp)][iq] = self.values['rates3rd_ir'][str(temp)][index]
                if False:
                    print (self.values['rates3rd_fl'][str(temp)][:,0])
                    print (len(self.values['rates3rd_fl'][str(temp)][:,1]))
            except IOError:
                print ("BTE.w file does not exist!")                            
            
            
    def get_kappa(self):
        # three phonon kappa
        self.values['kappa'] = {}
        for temp in self.values['temp']:
            fname = os.path.join(self.path_dir, "T"+str(temp)+"K/"+"BTE.kappa_tensor")
            print (fname)
            try:
                with open(fname, "r") as file:
                    lines = file.readlines()
                # first item is RTA
                # second item is IRTA
                self.values['kappa'][str(temp)] = \
                    [np.array(list(map(float, lines[0].split()[1:]))).reshape((3,3)),\
                     np.array(list(map(float, lines[-1].split()[1:]))).reshape((3,3))]
            except IOError:
                print ("BTE.kappa_tensor file does not exist!")
        # four phonon kappa
        if self.values['fourthrate'] == True:
            self.values['kappa_4th'] = {}
            for temp in self.values['temp']:
                fname = os.path.join(self.path_dir, "T"+str(temp)+"K/"+"BTE.kappa_tensor_4th")
                try:
                    with open(fname, "r") as file:
                        lines = file.readlines()
                        self.values['kappa_4th'][str(temp)] = \
                            [np.array(list(map(float, lines[0].split()[1:]))).reshape((3,3))]
                except IOError:
                    print ("BTE.kappa_tensor_4th file does not exist!")


    def get_phase(self):
        self.values['phase'] = {}
        self.values['phase']['wp3'] = []
        self.values['phase']['wp4'] = []
        for temp in self.values['temp']:
            file_wp3 = os.path.join(self.path_dir, "T"+str(temp)+"K" + "/BTE.WP3")
            file_wp4 = os.path.join(self.path_dir, "T"+str(temp)+"K" + "/BTE.WP4")
            print (file_wp3)
            if os.path.isfile(file_wp3):
                self.values['phase']['wp3'].append(np.loadtxt(file_wp3, dtype=float))
            if os.path.isfile(file_wp4):
                self.values['phase']['wp4'].append(np.loadtxt(file_wp4, dtype=float))
        


if __name__ == '__main__':
    # basic extraction
    pass
            
            
    
