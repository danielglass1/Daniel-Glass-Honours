import torch
import matplotlib.pyplot as plt
from mp_torch.solver import solver
from mp_torch.forward import forward
import matplotlib as mpl
mpl.use('Agg') #for writing to file only

eps=1e-12
trials=30
pack_circles_max=4
prescreen_stall_max=2
stall_max=3

preview=True

### INCIDENT FIELD VAR ####################################################################
inc_phi = torch.tensor(0.5*torch.pi)       
k_count=1
kmin=2*torch.pi/0.3
kmax=2*torch.pi/2.5
k0_range=torch.linspace(kmin,kmax,k_count)

inc_k_0 = k0_range               
inc_k_perp=inc_k_0*torch.sin(inc_phi)

### CONSTANTS ###########################################################################
#fiber
jacket_n_real = torch.tensor(1.5)
jacket_n_imag= torch.tensor(0)
jacket_n=jacket_n_real+1j*jacket_n_imag
jacket_a = torch.tensor(5)
learning_rate=jacket_a*0.02

#holes
cyls_n_real=torch.tensor(1.0)
cyls_n_imag=0

### PREVIEW STRUCTURE ##########################################################################
def preview_strucure(var_input,title,file_name):
    if preview:
        var_input=var_input.detach()
        num_cyls=int(len(var_input)/3)
        raw_pos_x,raw_pos_y,raw_rad=torch.split(var_input,num_cyls)
        #fix params
        pos_x=torch.cat([softplus(raw_pos_x[:1]),raw_pos_x[1:]])
        pos_y=torch.cat([torch.zeros(1),raw_pos_y[1:]])
        rad=softplus(raw_rad)

        rad=softplus(rad) #avoid negative radius
        fig, ax = plt.subplots()
        for i in range(num_cyls):
            circle = plt.Circle((pos_x[i], pos_y[i]), rad[i], edgecolor='blue', facecolor='none', linewidth=0.5)
            ax.add_patch(circle)

        jacket = plt.Circle((0, 0), jacket_a, edgecolor='blue', facecolor='none', linewidth=0.5)
        ax.add_patch(jacket)

        ax.set_aspect('equal')
        lim=jacket_a*1.2
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        # ax.grid(True, linestyle='--')
        plt.title(f"Q_f = {title.item()*100:.3f}%, n holes={num_cyls}")
        plt.savefig(file_name)
        plt.close()

### OBJECTIVE #############################################################################
softplus=torch.nn.Softplus(beta=25)
def objective(var_input,trunc_modify,intersect_penalty):
    num_cyls=int(len(var_input)/3) 
    raw_pos_x,raw_pos_y,raw_rad=torch.split(var_input,num_cyls)
    #fix params
    pos_x=torch.cat([softplus(raw_pos_x[:1]),raw_pos_x[1:]])
    pos_y=torch.cat([torch.zeros(1),raw_pos_y[1:]])
    rad=softplus(raw_rad)+1e-7

    
    # Jacket Intersect Penalty
    pos_r=torch.sqrt(pos_x**2+pos_y**2+eps)
    jacket_boundary=((pos_r+rad)/jacket_a)-1 #less than 0 when in boundary
    jacket_penalty=torch.sum(softplus(jacket_boundary))
    

    pos_x_diff=pos_x[None,:]-pos_x[:,None]    
    pos_y_diff=pos_y[None,:]-pos_y[:,None] 
    pos_diff_sq=pos_x_diff*pos_x_diff+pos_y_diff*pos_y_diff
    rad_total=rad[None,:]+rad[:,None]
    cyl_boundary_normalised=1-((pos_diff_sq)/(rad_total*rad_total))
    cylinder_penalty=softplus(cyl_boundary_normalised) # penalised for negative distances/intersect otherwise = 0
    cylinder_penalty=torch.sum(torch.triu(cylinder_penalty,diagonal=1))
    penalty=intersect_penalty*(jacket_penalty+cylinder_penalty)

    # Multipole solver
    # X,Y,R,N_real,N_imag
    opt_cylinders=torch.zeros(num_cyls,5)
    opt_cylinders[:,0]=pos_x
    opt_cylinders[:,1]=pos_y
    opt_cylinders[:,2]=rad
    opt_cylinders[:,3]=cyls_n_real
    opt_cylinders[:,4]=cyls_n_imag
    forward_q=torch.zeros(k_count)
    
                        #cylinders, inc_k_0, inc_theta,   jacket_a,jacket_n_real, jacket_n_imag, trunc mod)
    for i in range(k_count):
        solver1=solver(opt_cylinders, inc_k_0[i], inc_phi, jacket_a, jacket_n_real,jacket_n_imag,trunc_modify,False,1)
        S_0=solver1.jacket_B_matrix
        forward_q[i]=forward(S_0,inc_k_perp[i],jacket_a)
    
    forward_q_avg=torch.sum(forward_q)/k_count
    loss=forward_q_avg+penalty
    return loss
    
### PACK CIRCLES ###############################################################
def sample_radii(n, center, lo, hi, kappa): #kappa>0 larger kappa tighter spread
    p = (center - lo) / (hi - lo)
    alpha = p * kappa
    beta  = (1 - p) * kappa
    dist = torch.distributions.Beta(alpha, beta)
    y = dist.sample((n,))
    return lo + (hi - lo) * y  # in [lo, hi]

def pack_circles(max_circles, container_radius,  max_attempts=5000):
    #generate radii
    approx_center=torch.sqrt(0.5/torch.tensor(max_circles))*container_radius
    test_radii=sample_radii(max_circles,approx_center,0,jacket_a,10)
    # Place larger circles first 
    radii_sorted,_ = torch.sort(test_radii)
    placed_x=torch.zeros(max_circles)
    placed_y=torch.zeros(max_circles)
    placed_r=torch.zeros(max_circles)    
    for i in range(max_circles):
        for _ in range(max_attempts):
            # polar sampling
            R_bound=container_radius-radii_sorted[i]            
            test_theta = 2*torch.pi * torch.rand(1)          
            test_r = R_bound * torch.sqrt(torch.rand(1))
            test_x = test_r*torch.cos(test_theta)
            test_y = test_r*torch.sin(test_theta)

            if i==0:
                placed_x[0]=torch.abs(test_x)
                placed_y[0]=0
                placed_r[0]=radii_sorted[i]
                num_placed=1
                break

            dx=(placed_x[:i] - test_x)
            dy=(placed_y[:i] - test_y)
            distance_sq = dx * dx + dy * dy
            total_rad=placed_r[:i]+radii_sorted[i]
            total_rad_sq=total_rad*total_rad
            diff=distance_sq-total_rad_sq

            if torch.min(diff)>=0:
                placed_x[i]=test_x
                placed_y[i]=test_y
                placed_r[i]=radii_sorted[i]
                num_placed+=1
                break
    
    return torch.cat([placed_x[:num_placed],placed_y[:num_placed],placed_r[:num_placed]])

### LOOP ###################################################################################
absolute_min_loss=1e12
precision=1e-3

    
        
jacket_penalty_weight=1

for trial_num in range(trials):
    ### initial guess/screening ############################################################################
    screen_min_loss=1e9
    stall=0
    with torch.no_grad():
        for screen_num in range(10000):
            screen_input=pack_circles(pack_circles_max,jacket_a,2000)
            screen_objective=objective(screen_input,1,0)
            
            stall=stall+1
            print(stall)
            if screen_min_loss>screen_objective+precision: 
                preview_strucure(screen_input,screen_objective,"simulations/structure_preview")
                screen_min_loss=screen_objective.clone()
                trial_input=screen_input.clone() #need to clone or they just become the same tensor
                stall=0
                print(screen_min_loss)
            if stall>prescreen_stall_max:break 
    print("screening minimum = ", screen_min_loss)

    ### Gradient Optimisation #############################################################################
    x = torch.nn.Parameter(trial_input)
    opt = torch.optim.AdamW([x], lr=learning_rate)

    torch.autograd.set_detect_anomaly(True) #epic for finding where NaNs happen - typically gradients of atans and sqrts

    trial_min_loss=1e12
    stall=0

    for step in range(400):
        opt.zero_grad()
        loss = objective(x,1.0,jacket_penalty_weight) #input, truncation modifier, intersection penalty
        loss.backward()
        preview_strucure(x,loss,"simulations/structure_preview") 
        stall=stall+1
        if trial_min_loss>loss+precision:
            trial_min_loss=loss.clone()
            trial_min_input=x.clone()
            stall=0
        if stall>stall_max:break 

        opt.step()

    trial_min_loss=objective(trial_min_input,1,0)
    num_cyls=int(len(trial_min_input)/3) 

    id=str(trial_min_loss.item())
    id=id.replace(".", "p")
    id=id.replace("-", "")
    file_name=id
    torch.save(trial_min_input.clone(),file_name+".pt")
    preview_strucure(trial_min_input,trial_min_loss,file_name)