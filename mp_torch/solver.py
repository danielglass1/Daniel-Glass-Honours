import torch
from .get_matrices import get_matrices  #needs a dot cause in same folder?
from .torch_bessel import jv, h1v
import time
default_type=torch.float32
default_ctype=torch.complex64
torch.set_default_dtype(default_type)

class solver:
    def __init__(self,cylinders,inc_k_0, inc_phi, jacket_a, jacket_n_real, jacket_n_imag, modify_trunc=1,solve_neumann=False,neumann_depth=3):
        
        #Globals
        self.inc_k_0=inc_k_0
        self.inc_theta=inc_phi
        self.inc_k_z=inc_k_0*torch.cos(inc_phi)

        #functions
        eps=1e-12                        
        def wiscombe_trunc(k_perp,a): 
            x=torch.real(k_perp.detach()*a.detach())
            wiscombe=(x+2+4*x**(1/3))*modify_trunc
            return torch.ceil(wiscombe).to(torch.int)
        
        def generate_k_perp(n):
            inc_k=n*inc_k_0
            return torch.sqrt(inc_k*inc_k-self.inc_k_z*self.inc_k_z)
        
        self.inc_k_perp=torch.sqrt(inc_k_0**2-self.inc_k_z**2+eps)

        #JACKET
        self.jacket_a=jacket_a
        self.jacket_n=jacket_n_real+1j*jacket_n_imag
        self.jacket_k_perp=generate_k_perp(self.jacket_n)

        self.jacket_trunc=wiscombe_trunc(self.inc_k_perp,self.jacket_a)
        self.jacket_terms=2*self.jacket_trunc+1 

        #CYLINDER
        self.cyl_pos=cylinders[:,0:2]
        self.cyl_a=cylinders[:,2]
        self.cyl_n=cylinders[:,3]+1j*cylinders[:,4]
        row,col=cylinders.shape
        self.cyl_count=row
        cyl_k=self.cyl_n*inc_k_0
        self.cyl_k_perp=generate_k_perp(self.cyl_n)

        self.cyl_trunc=wiscombe_trunc(self.jacket_k_perp,self.cyl_a)
        self.cyl_terms=2*self.cyl_trunc+1
        self.cyl_terms_total=torch.sum(self.cyl_terms)

        cyl_arg=torch.atan2(self.cyl_pos[:,1]+eps,self.cyl_pos[:,0]+eps)
        cyl_mod=torch.linalg.norm(self.cyl_pos,dim=1)
        cyl_pos_diff=self.cyl_pos[None,:,:]-self.cyl_pos[:,None,:]    
        cyl_mod_diff=torch.linalg.norm(cyl_pos_diff,axis=2) #distance between any pair of cylinders
        cyl_arg_diff=torch.atan2(cyl_pos_diff[...,1]+eps,cyl_pos_diff[...,0]+eps) #arg between any pair of cylinders

        #CYLINDER TO ORIGIN 
        self.H_oc=torch.zeros((2*self.jacket_terms, 2*self.cyl_terms_total),dtype=default_ctype)
        vn=torch.arange(-self.jacket_trunc,self.jacket_trunc+1)[:,None]
        col_offset=0
        for i in range(self.cyl_count):
            vm=torch.arange(-self.cyl_trunc[i],self.cyl_trunc[i]+1)[None,:] #could do this at the start and have a list of vns...
            H_oc_block=jv(vn-vm,self.jacket_k_perp*cyl_mod[i])*torch.exp(-1j*(vn-vm)*cyl_arg[i])
            self.H_oc[:self.jacket_terms,col_offset:col_offset+self.cyl_terms[i]]=H_oc_block
            col_offset+=self.cyl_terms[i]
        self.H_oc[self.jacket_terms:,self.cyl_terms_total:]=self.H_oc[:self.jacket_terms,:self.cyl_terms_total]

        #ORIGIN TO CYLINDER
        self.H_co=torch.zeros((2*self.cyl_terms_total,2*self.jacket_terms),dtype=default_ctype)
        vm=torch.arange(-self.jacket_trunc,self.jacket_trunc+1)[None,:]
        row_offset=0
        for i in range(self.cyl_count):
            vn=torch.arange(-self.cyl_trunc[i],self.cyl_trunc[i]+1)[:,None] 
            H_co_block=jv(vm-vn,self.jacket_k_perp*cyl_mod[i])*torch.exp(1j*(vm-vn)*cyl_arg[i])
            self.H_co[row_offset:row_offset+self.cyl_terms[i],:self.jacket_terms]=H_co_block
            row_offset+=self.cyl_terms[i]
        self.H_co[self.cyl_terms_total:,self.jacket_terms:]=self.H_co[:self.cyl_terms_total,:self.jacket_terms]

        # CYLIDNER TO CYLINDER
        self.H_cc=torch.zeros((2*self.cyl_terms_total, 2*self.cyl_terms_total),dtype=default_ctype)
        row_id=0
        for d in range(self.cyl_count):
                col_id=0
                n = 2*self.cyl_trunc[d]+1
                vn=torch.arange(-self.cyl_trunc[d],self.cyl_trunc[d]+1)[:,None]
                for c in range(self.cyl_count):
                    m = 2*self.cyl_trunc[c]+1       
                    if d!=c:
                        vm=torch.arange(-self.cyl_trunc[c],self.cyl_trunc[c]+1)[None,:]
                        H_cc_block=h1v(vn-vm,self.jacket_k_perp*(cyl_mod_diff[d,c]))*torch.exp(-1j*(vn-vm)*cyl_arg_diff[d,c])
                        self.H_cc[row_id:row_id+n,col_id:col_id+m]=H_cc_block
                    col_id+=m
                row_id += n
        self.H_cc[self.cyl_terms_total:,self.cyl_terms_total:]=self.H_cc[:self.cyl_terms_total,:self.cyl_terms_total]

        ### SCATTERING MATRICES
        self.c_R_out=torch.zeros((2*self.cyl_terms_total, 2*self.cyl_terms_total),dtype=default_ctype)
        self.c_T_in=torch.zeros((2*self.cyl_terms_total, 2*self.cyl_terms_total),dtype=default_ctype)
        offset=0
        for i in range(self.cyl_count):   #jacket?, trunc,    n_o,         n_i,     a,       k_z,    k_0):
            R_EE,R_EK,R_KE,R_KK,T_EE,T_EK,T_KE,T_KK = get_matrices(False,self.cyl_trunc[i],self.jacket_n,   
                                                                   self.cyl_n[i],  self.cyl_a[i],self.inc_k_z,inc_k_0, 
                                                                   default_type,default_ctype)
            ct=self.cyl_terms[i]
            self.c_R_out.diagonal(0)[offset:offset+ct] = R_EE
            self.c_R_out[:self.cyl_terms_total, self.cyl_terms_total:].diagonal(0)[offset:offset+ct] = R_EK
            self.c_R_out[self.cyl_terms_total:,:self.cyl_terms_total].diagonal(0)[offset:offset+ct] = R_KE
            self.c_R_out.diagonal(0)[offset+self.cyl_terms_total:offset+self.cyl_terms_total+ct] = R_KK

            self.c_T_in.diagonal(0)[offset:offset+ct] = T_EE
            self.c_T_in[:self.cyl_terms_total, self.cyl_terms_total:].diagonal(0)[offset:offset+ct] = T_EK
            self.c_T_in[self.cyl_terms_total:,:self.cyl_terms_total].diagonal(0)[offset:offset+ct] = T_KE
            self.c_T_in.diagonal(0)[offset+self.cyl_terms_total:offset+self.cyl_terms_total+ct] = T_KK

            offset+=self.cyl_terms[i]
        
        # Jacket Scattering Matrices         build matrix?, trunc,    n_o,  n_i,     a,      k_z,    k_0):
        self.j_R_out,self.j_R_in,self.j_T_out,self.j_T_in=get_matrices(True,self.jacket_trunc,1.0,self.jacket_n,
                                                             jacket_a,self.inc_k_z,inc_k_0,default_type,default_ctype)
    
        
        I=torch.eye((2*self.cyl_terms_total),dtype=default_ctype)

        self.step1= self.H_co @ self.j_T_in
        # solve
        self.M=self.c_R_out @ (self.H_cc + self.H_co @ self.j_R_in @ self.H_oc)
        solve_input=self.c_R_out @ self.H_co @ self.j_T_in
        if solve_neumann==True:
            P=I+self.M
            for _ in range(neumann_depth):
                P=I+self.M@P   
            self.cyl_B_matrix= P @ solve_input
        else:
            B_LU,B_pivots=torch.linalg.lu_factor(I-self.M)
            self.cyl_B_matrix=torch.linalg.lu_solve(B_LU,B_pivots,solve_input)

        self.jacket_B_matrix=self.j_T_out @ self.H_oc @ self.cyl_B_matrix + self.j_R_out #critical #S_0

        # #trace estimation
        # A= self.j_T_out @ self.H_oc
        # A_herm=A.conj().T
        # C= self.c_R_out @ self.H_co @ self.j_T_in
        # C_herm=C.conj().T
        # D= self.j_R_out
        # D_herm=D.conj() #transpose unneccesary - diagonal
        # D_diag=torch.diag(D) 
        # im_D_diag=torch.imag(D_diag)
        # re_D_diag=torch.real(D_diag)
        # D_diag_sq=torch.sum(im_D_diag*im_D_diag+re_D_diag*re_D_diag)






        

        

    

       
        
        