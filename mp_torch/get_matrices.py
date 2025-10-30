import torch
from .torch_bessel import jv, jvp, h1v, h1vp

def get_matrices(jacket,trunc,n_o,n_i,a,k_z,k_0,default_type,default_ctype):
    torch.set_default_dtype(default_type)
    eps=1e-12
    order_vec=torch.arange(-trunc,trunc+1)
    k_perp_i=torch.sqrt((n_i*k_0)**2-k_z**2+eps)
    k_perp_o=torch.sqrt((n_o*k_0)**2-k_z**2+eps)
    
    H_o,HP_o=h1v(order_vec,k_perp_o*a),h1vp(order_vec,k_perp_o*a)
    H_i,HP_i=h1v(order_vec,k_perp_i*a),h1vp(order_vec,k_perp_i*a)
    J_o,JP_o=jv(order_vec,k_perp_o*a),jvp(order_vec,k_perp_o*a)
    J_i,JP_i=jv(order_vec,k_perp_i*a),jvp(order_vec,k_perp_i*a)

    X_JJ_e=k_perp_o*n_i**2*JP_i*J_o - k_perp_i*n_o**2*J_i*JP_o
    X_JJ_m=k_perp_o       *JP_i*J_o - k_perp_i       *J_i*JP_o
    X_HH_e=k_perp_o*n_i**2*HP_i*H_o - k_perp_i*n_o**2*H_i*HP_o
    X_HH_m=k_perp_o       *HP_i*H_o - k_perp_i       *H_i*HP_o
    X_JH_e=k_perp_o*n_i**2*JP_i*H_o - k_perp_i*n_o**2*J_i*HP_o
    X_JH_m=k_perp_o       *JP_i*H_o - k_perp_i       *J_i*HP_o

    C=(k_perp_i**2-k_perp_o**2)*(k_z*1j*order_vec)/(a*k_0*k_perp_i*k_perp_o)
    delta=(C*J_i*H_o)**2+X_JH_e*X_JH_m
    denominator=1/(delta)
    W_o=(2j)/(torch.pi*k_perp_o*a)
    W_i=(2j)/(torch.pi*k_perp_i*a)

    #Reflection on outside of cylinder
    R_out_diag=-(C*J_i)**2*H_o*J_o
    R_out_EE=denominator*(R_out_diag-X_JH_m*X_JJ_e)
    R_out_KK=denominator*(R_out_diag-X_JH_e*X_JJ_m)
    R_out_cross=denominator*k_perp_i*C*W_o*J_i**2
    R_out_EK=-R_out_cross
    R_out_KE=R_out_cross*n_o**2

    #transmission into cylinder
    T_in_diag=-denominator*k_perp_i*W_o #should this have a minus?
    T_in_EE=T_in_diag*n_o**2*X_JH_m
    T_in_KK=T_in_diag      *X_JH_e

    T_in_cross=T_in_diag*C*J_i*H_o
    T_in_EK=T_in_cross
    T_in_KE=-T_in_cross*n_o**2

    if jacket:
        #Reflection on inside of cylinder
        R_in_diag=-(C*H_o)**2*H_i*J_i
        R_in_EE=denominator*(R_in_diag-X_JH_m*X_HH_e)
        R_in_KK=denominator*(R_in_diag-X_JH_e*X_HH_m)
        R_in_cross=denominator*k_perp_o*C*W_i*H_o**2
        R_in_EK=-R_in_cross
        R_in_KE=R_in_cross*n_i**2

        #transmission out of cylinder
        T_out_diag=-denominator*k_perp_o*W_i
        T_out_EE=T_out_diag*n_i**2*X_JH_m
        T_out_KK=T_out_diag       *X_JH_e

        T_out_cross=T_out_diag*C*J_i*H_o
        T_out_EK=T_out_cross
        T_out_KE=-T_out_cross*n_i**2

        vec_size=2*trunc+1
        matrix_size=2*vec_size

        def fill_blocks(EE, EK, KE, KK):
            M = torch.zeros(matrix_size, matrix_size,dtype=default_ctype)
            M.diagonal(0)[:vec_size] = EE
            M[:vec_size, vec_size:].diagonal(0)[:] = EK
            M[vec_size:, :vec_size].diagonal(0)[:] = KE
            M.diagonal(0)[vec_size:] = KK
            return M

        R_out = fill_blocks(R_out_EE, R_out_EK, R_out_KE, R_out_KK)
        R_in  = fill_blocks(R_in_EE,  R_in_EK,  R_in_KE,  R_in_KK)
        T_out = fill_blocks(T_out_EE, T_out_EK, T_out_KE, T_out_KK)
        T_in  = fill_blocks(T_in_EE,  T_in_EK,  T_in_KE,  T_in_KK)

        return R_out,R_in,T_out,T_in
    
    else:
        return R_out_EE,R_out_EK,R_out_KE,R_out_KK, T_in_EE,T_in_EK,T_in_KE,T_in_KK

    


