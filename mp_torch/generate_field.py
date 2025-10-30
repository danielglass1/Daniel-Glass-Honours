import torch
from mp_torch.torch_bessel import jv, h1v
torch.set_default_dtype(torch.float32)
import matplotlib
matplotlib.use('Agg') #for writing to file only

def generate_field(solver,inc_magnitude,inc_delta,inc_theta,xy_range,npts):
    with torch.no_grad():
        inc_k_perp_vec=solver.inc_k_perp*torch.tensor([torch.cos(inc_theta),torch.sin(inc_theta)])

        inc_z_magnitude=inc_magnitude*torch.sin(solver.inc_theta)
        inc_Ez_magnitude=inc_z_magnitude*torch.sin(inc_delta)
        inc_Kz_magnitude=inc_z_magnitude*torch.cos(inc_delta)
        ### SOLVE FOR COEFFECIENTS #############################################################
        # Source Field - Jacobi Anger Expansion
        jacket_trunc_vec=torch.arange(-solver.jacket_trunc,solver.jacket_trunc+1)
        A_E0=inc_Ez_magnitude*torch.exp(1j*jacket_trunc_vec*(torch.pi/2-inc_theta))
        A_K0=inc_Kz_magnitude*torch.exp(1j*jacket_trunc_vec*(torch.pi/2-inc_theta))
        A_0=torch.cat([A_E0,A_K0])

        B_0=solver.jacket_B_matrix @ A_0
        B_E=(B_0[0:solver.jacket_trunc*2+1])[:,None,None]
        B_K=(B_0[solver.jacket_trunc*2+1:])[:,None,None]

        jacket_A_matrix = solver.j_R_in @ solver.H_oc @ solver.cyl_B_matrix + solver.j_T_in 
        cyl_A_matrix = solver.c_T_in @ (solver.H_co @ jacket_A_matrix + solver.H_cc @ solver.cyl_B_matrix)

        ### FIELD GENERATION #################################################################
        x=torch.linspace(-xy_range,xy_range,npts)
        y=torch.linspace(-xy_range,xy_range,npts)
        X, Y = torch.meshgrid(x, y,indexing='xy') #indexing='ij' transposes in the image
        ORDER=jacket_trunc_vec[:,None,None]            
        R  = torch.sqrt((X)**2 + (Y)**2)[None,:,:]  
        THETA=torch.arctan2(Y,X)[None,:,:]
        THETA_TERM=torch.exp(1j*(ORDER * THETA))

        ### SOURCE FIELDS ######################################################################
        # exact plane wave
        outside_E_field=inc_Ez_magnitude*torch.exp(1j*(inc_k_perp_vec[0]*X+inc_k_perp_vec[1]*Y))
        outside_K_field=inc_Kz_magnitude*torch.exp(1j*(inc_k_perp_vec[0]*X+inc_k_perp_vec[1]*Y))

        ### JACKET BASIS FIELDS ###################################################
        # Radiating field 
        SHARED_TERM=h1v(ORDER, solver.inc_k_perp * R)*THETA_TERM
        outside_E_field+=torch.sum(B_E*SHARED_TERM,dim=0)
        outside_K_field+=torch.sum(B_K*SHARED_TERM,dim=0)

        # Internal field
        jacket_A=jacket_A_matrix @ A_0
        Jacket_A_E=(jacket_A[:solver.jacket_terms])[:,None,None]
        jacket_A_K=(jacket_A[solver.jacket_terms:])[:,None,None]
        SHARED_TERM=jv(ORDER,solver.jacket_k_perp*R)*THETA_TERM
        jacket_E_field=torch.sum(Jacket_A_E*SHARED_TERM,axis=0)
        jacket_K_field=torch.sum(jacket_A_K*SHARED_TERM,axis=0)

        ### CYLINDER BASIS FIELDS ############################
        # Radiating
        cyl_B=solver.cyl_B_matrix @ A_0  
        # Internal  
        cyl_A=cyl_A_matrix @ A_0 

        #broadcasting
        X_cyl=X[None,None,:,:]-solver.cyl_pos[:,0][:,None,None,None] #cyl, order, X, Y
        Y_cyl=Y[None,None,:,:]-solver.cyl_pos[:,1][:,None,None,None]
        R_cyl=torch.sqrt((X_cyl)**2+(Y_cyl)**2)
        THETA_cyl=torch.arctan2(Y_cyl,X_cyl)
        cyl_E_fields=torch.zeros(solver.cyl_count,npts,npts,dtype=torch.complex64)
        cyl_K_fields=torch.zeros(solver.cyl_count,npts,npts,dtype=torch.complex64)
        offset=0
        for i in range(solver.cyl_count):
            ORDER=torch.arange(-solver.cyl_trunc[i],solver.cyl_trunc[i]+1)[:,None,None]
            THETA_TERM_cyl=torch.exp(1j*ORDER*(THETA_cyl[i]))

            E_B_coeffs=(cyl_B[offset:offset+solver.cyl_terms[i]])[:,None,None]
            K_B_coeffs=(cyl_B[solver.cyl_terms_total+offset:solver.cyl_terms_total+offset+solver.cyl_terms[i]])[:,None,None]
            SHARED_TERM=h1v(ORDER,solver.jacket_k_perp*R_cyl[i])*THETA_TERM_cyl
            jacket_E_field+=torch.sum(E_B_coeffs*SHARED_TERM,axis=0)
            jacket_K_field+=torch.sum(K_B_coeffs*SHARED_TERM,axis=0)
            
            E_A_coeffs=(cyl_A[offset:offset+solver.cyl_terms[i]])[:,None,None]
            K_A_coeffs=(cyl_A[solver.cyl_terms_total+offset:solver.cyl_terms_total+offset+solver.cyl_terms[i]])[:,None,None]
            SHARED_TERM=jv(ORDER,solver.cyl_k_perp[i]*R_cyl[i])*THETA_TERM_cyl
            cyl_E_fields[i,:,:]=torch.sum(E_A_coeffs*SHARED_TERM,axis=0)
            cyl_K_fields[i,:,:]=torch.sum(K_A_coeffs*SHARED_TERM,axis=0)
            offset+=solver.cyl_terms[i]

        ### FIELD MASKING #############################################################################
        # Jacket 
        mask=torch.sqrt((X)**2 + (Y)**2)<solver.jacket_a
        E_field=torch.where(mask,jacket_E_field,outside_E_field).to(dtype=torch.complex64)
        K_field=torch.where(mask,jacket_K_field,outside_K_field).to(dtype=torch.complex64)

        # Cylinder
        for i in range(solver.cyl_count):
            mask = R_cyl[i,0] < solver.cyl_a[i]
            E_field[mask]=cyl_E_fields[i,:,:][mask]
            K_field[mask]=cyl_K_fields[i,:,:][mask]

        return E_field,K_field