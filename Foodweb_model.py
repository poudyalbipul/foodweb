import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp

###############################################################################
# empirical data from Li et al


# parameters taken for the Varying size ratio model with temperature for aquatic
# species
#                 \beta_x,0 \beta_x,1    \beta_x,temp
beta_base = np.array([[3.1503,  0.4600,    -0.0738], # mu_i
                 [1.6645,    0.2172,     -0.0912], # sigma_i
                 [-1.102,   -0.1533,    0.0309], # \theta_i
                 [-0.002,    -0.24*np.log(10), 0], #x_i
                 [0.05,      0,              0]]) #c_intra 


# change from f_i = m_i - mu_i
beta_base[0] = -beta_base[0]
beta_base[0,1] += 1
# change to natrual log instead of log10
beta_base[:,1] /= np.log(10)

# this allows to still access beta_base for other programs
beta = beta_base.copy()

#issue is with uiterwaal, small species don't eav fast enough'
# what dimensions do we have in mind?

def generate_species(n_spec, e = 0.45, K = 1e7, beta = beta.copy(), random = True,
                     B0 = 1e-7):
    species_id = {
        "m_i": 10**(-np.random.normal(0,2,n_spec)),
        "e": e, "K": K, "c_intra": beta[4,0]}
    species_id["m_i"] = np.sort(species_id["m_i"])
    if not random:
        species_id["m_i"] = 10**np.linspace(-6,4, n_spec)
    if B0:
        species_id["m_i"][0] = B0
    species_id["random_state"] = np.random.randint(2**30-1)
    
    return species_id

traits = ["f_i", "sig_i", "theta_i"]
def change_temperature(species_id, T = 20, beta = beta.copy()):

    delta_T = T - 20 # Li et al used temperatures relative to 20°C
    for i, trait in enumerate(traits):
        species_id[trait] = beta[i, 0] + beta[i,1]*np.log(species_id["m_i"]) + beta[i, 2]*delta_T
        
    # transformation of the traits
    # adjust that body mass is given in natural log
    species_id["f_i"] = np.exp(np.log(10)*species_id["f_i"])
    # adjust that body mass is given in natural log
    species_id["sig_i"] = np.log(10)*np.sqrt(np.exp(species_id["sig_i"]))
    species_id["theta_i"] = np.exp(species_id["theta_i"])
    species_id["theta_i"] = species_id["theta_i"]/(1+species_id["theta_i"])
    
    species_id["f_i"][0] = 1e-50
    
    return species_id

# for plotting and testing
n_spec = 101
species_id_base = generate_species(n_spec)
species_id_base["m_i"] = 10**np.linspace(-12,8, n_spec)
species_id_base = change_temperature(species_id_base)

def compute_predation_prob(species_id):
    log_mi = np.log(species_id["m_i"])
    log_fi = np.log(species_id["f_i"])
    
    # preference matrix
    s_ji = (species_id["theta_i"]
            *np.exp(-(log_mi[:,np.newaxis] - log_fi)**2/(2*species_id["sig_i"]**2)))
    return s_ji

def compute_links(species_id):
    rng_interact = np.random.RandomState(species_id["random_state"])
    
    s_ji = compute_predation_prob(species_id)
    treshhold = rng_interact.uniform(0.05,1, s_ji.shape)
    
    return 1.0*(s_ji>treshhold)

if __name__ == "__main__":
    s_ji = compute_predation_prob(species_id_base)
    extent = np.percentile(np.log10(species_id_base["m_i"]), [0,100,0,100])
    fig, ax = plt.subplots(2,2, figsize = (9,9))
    fig.suptitle("Basic interactions")
    cmap = ax[0,0].imshow(s_ji, extent = extent, origin = "lower")
    fig.colorbar(cmap, ax = ax[0,0])
    cmap = ax[0,0].contour(s_ji, extent = extent, origin = "lower",
                           cmap = "RdBu", levels = np.append([0.05],np.arange(0.1, 0.7, 0.1)))
    ax[0,0].plot(extent, extent, 'r')
    fig.colorbar(cmap, ax = ax[0,0])
    ax[0,0].set_xlabel("Predator bodysize [log]")
    ax[0,0].set_ylabel("Prey bodysize [log]")
    ax[0,0].set_title("Predation probability")

###############################################################################
# empirical data from uiterwaal et al.

# 3D predation space
forage_attack = {
    "intercept": -3.3, # intercept of regression
    "temp_2": -0.01, # quadratic effect of temperature
    "temp": 0.32, # linear effect of temperature
    "log_pred": 0.54, # effect of predator mass
    "log_prey": 0.05, # effect of prey mass
    "T_ref": 26.7} # reference temperature for fit

def compute_attack_rate(species_id, T = 20, forage_attack=forage_attack.copy()):
    # basic attack rate
    log_mi = np.log(species_id["m_i"])
    
    interm = {
        "temp_2": (T-forage_attack["T_ref"])**2, "temp": T-forage_attack["T_ref"],
        "log_pred": log_mi, "log_prey": log_mi[:,np.newaxis]}
    
    attack_rate = forage_attack["intercept"]
    for key in interm.keys():
        attack_rate = attack_rate + interm[key]*forage_attack[key]
    
    # convert from log space to normal space
    attack_rate = np.exp(attack_rate)
    # convert from attack rate/individuum to attak rate per kg
    attack_rate /= species_id["m_i"][:,np.newaxis]
    
    return attack_rate

# plot basic result of uiterwaal
if __name__ == "__main__":
    attack_rate = compute_attack_rate(species_id_base)
    cmap = ax[0,1].imshow(np.log(attack_rate),
               origin = "lower", extent = extent)
    fig.colorbar(cmap, ax = ax[0,1])
    ax[0,1].set_xlabel("Predator bodysize [log]")
    ax[0,1].set_ylabel("Prey bodysize [log]")
    ax[0,1].set_title("Attack rate")



###############################################################################
# combine Li and Uiterwaal to create actual food web model

def compute_LV_param(species_id, T = 20, beta = beta, forage_attack = forage_attack.copy()):

    # get predation matrix
    links_ji = compute_links(species_id)
    
    # regression obtained from Li et al.
    attack_rate_ji = compute_attack_rate(species_id, T,forage_attack)
    
    # compute predation strength
    pred_ji = links_ji*attack_rate_ji
    
    # Lotka_Volterra interaction rate
    a_ij = (-(species_id["e"]*pred_ji).T # predation
            + (pred_ji) # consumption
            + np.eye(len(species_id["sig_i"]))*beta[4,0]) # self lim
    
    a_ij[0,0] = 1e0
    # cap maximal species interactions
    a_ij[np.abs(a_ij)>2] = 2*np.sign(a_ij[np.abs(a_ij)>2])
    
    mu = beta[3,0]*species_id["m_i"]**beta[3,1]
    mu[0] = species_id["K"]*a_ij[0,0]
    
    return mu, a_ij

def compute_trophic_level(species_id, present = None):

    s_ji = compute_links(species_id).T
    if not(present is None):
        s_ji = s_ji[present][:, present]
    s_ji = s_ji/np.sum(s_ji, axis = 1, keepdims=True)
    s_ji[np.isnan(s_ji)] = 0
    
    try:
        trophic_level = np.linalg.solve(np.eye(len(s_ji)) - s_ji, np.ones(len(s_ji)))
    except np.linalg.LinAlgError:
        return np.full(len(s_ji), np.nan)
    return trophic_level - 1

def plot_foodweb(species_id, surv = None, ax = plt):
    s_ij = compute_links(species_id)
    trophic_level = compute_trophic_level(species_id, surv)
    # spread species according to the trophic level
    ind = np.argsort(trophic_level)
    n_stages = 4
    n_per_stage = np.array(n_stages*[(len(ind)-1)//n_stages])
    n_per_stage[:(len(ind)-1)%n_stages] += 1
    
    x_loc = np.append(0.5, np.concatenate([np.linspace(0,1, n+2)[1:-1]
                                      for n in n_per_stage]))
    x_loc = x_loc[ind]
    ax.scatter(x_loc, trophic_level)
    for i in range(len(ind)):
        for j in range(len(ind)):
            if s_ij[i,j]:
                ax.arrow(x_loc[i], trophic_level[i],
                         x_loc[j]-x_loc[i], trophic_level[j]-trophic_level[i],
                         zorder = 0, alpha = 0.1)
    

###############################################################################
# plot results from both combined
if __name__ == "__main__":
    imu, A = compute_LV_param(species_id_base)
    cmap = ax[1,0].imshow(np.sign(A), extent = extent, cmap = "RdBu"
                          , origin = "lower", interpolation = "None")
    fig.colorbar(cmap, ax = ax[1,0])
    ax[1,0].set_xlabel("Predator bodysize [log]")
    ax[1,0].set_ylabel("Prey bodysize [log]")
    ax[1,0].set_title("Sign of interaction")
    
    cmap = ax[1,1].imshow(np.log(np.abs(A)), extent = extent, vmax = 1
                          , origin = "lower", interpolation = "None")
    fig.colorbar(cmap, ax = ax[1,1])
    ax[1,1].set_xlabel("Predator bodysize [log]")
    ax[1,1].set_ylabel("Prey bodysize [log]")
    ax[1,1].set_title("Magnitude of interaction")
    fig.tight_layout()

###############################################################################
# plot species traits and feeding preferences
if __name__ == "__main__":
    species_id = generate_species(150, random = False,
                                  B0 = 1e-8)
    species_id = change_temperature(species_id)
    mu, A = compute_LV_param(species_id)

    sigi = species_id["sig_i"][1:]
    mi = np.log10(species_id["m_i"])[1:]
    fi = np.log10(species_id["f_i"])[1:]
    thetai = species_id["theta_i"][1:]
    s_ji = compute_predation_prob(species_id)
    
    fig, ax = plt.subplots(3,2, sharex = True, figsize = (9,9))
    for a in ax.ravel():
        a.set_xlabel("Predator bodysize [log]")

    ax[0,0].plot(mi, fi, label = "f_i")
    ax[0,0].plot(mi, mi, label = "m_i")
    ax[0,0].legend()
    ax[0,0].set_ylabel("Feeding preference")

    ax[0,1].plot(mi, sigi, label = r"$\sigma_i$")
    ax[0,1].set_ylabel("Niche width, $\sigma_i$")
    ax01 = plt.twinx(ax[0,1])
    ax01.plot(mi, thetai, label = r"$\theta_i$", color = "r")
    ax[0,1].legend(loc = "center right")
    ax01.legend(loc = "center left")
    ax01.set_ylabel(r"Total predation probability, $\theta_i$")

    ax[1,0].plot(mi, s_ji[0,1:], label = "Prob. of eating B0",
                 color = "r")
    ax[1,0].axhline(np.mean(s_ji[0,1:]), label = "Mean eating B0", color = "r",
                    linestyle = "--")
    ax[1,0].plot(mi, np.mean(s_ji[:,1:], axis = 0), label = "Mean number of links",
                 color = "b")
    
    ax[1,0].axhline(np.mean(s_ji[:,1:]), label = "Mean of (mean number of links)", color = "b", ls = "--")
    ax[1,0].legend()
    ax[1,0].set_ylabel("Probability of eating B0")
    
    # compute hypothetical A if all species were linked
    s_ji_hypothetical = np.ones(s_ji.shape)
    attack_rate_ji = compute_attack_rate(species_id)
    pred_ji = s_ji_hypothetical*attack_rate_ji
    
    # Lotka_Volterra interaction rate

    a_ij = -(species_id["e"]*pred_ji).T # predation
    a_ij[a_ij<-2] = -2
    
    ax[1,1].semilogy(mi, -a_ij[1:,0], label = "Effect of B0 on species")
    ax[1,1].set_ylabel("Effect of B0 on species, $a_{i0}$")
    
    ax[2,0].plot(mi, mu[1:])
    ax[2,0].set_ylabel("Intrinsic growth rate")
    
    
    
    ax[2,1].plot(mi, mu[1:] - a_ij[1:,0]*species_id["K"])
    ax[2,1].axhline(0, color = "k")
    ax[2,1].set_ylim([min(mu[1:]), -min(mu[1:])])
    ax[2,1].set_ylabel("Invasion into empty")
    
    fig.tight_layout()
    
def LV_model(t, N, mu, A):
    return N*(mu - A.dot(N))


# Normalize the x-values to make the mean 0, min -1, and max 1
def normalize_x(x):
    x_min = np.min(x)
    x_max = np.max(x)
    x_mean = np.mean(x)
    
    # Apply the formula to normalize the x-values
    return 2 * (x - x_mean) / (x_max - x_min) 


if __name__ == "__main__":
    fig, ax = plt.subplots(3,3, sharex = True, sharey = True)
    fig2, ax2 = plt.subplots(*ax.shape, sharex = True, sharey = True)
   
    for a in ax[-1,:]:
        a.set_xlabel("Time")
    for a in ax[:,0]:
        a.set_ylabel("Densities")
    ax = ax.ravel()
    
    for a in ax2[-1,:]:
        a.set_xlabel("Bodymass")
    for a in ax2[:,0]:
        a.set_ylabel("Trophic level")
    ax2 = ax2.ravel() 
    
    for i in range(9):
        n_spec = 100
        species_id = generate_species(n_spec, random = True,
                                      B0 = 1e-6)
        species_id = change_temperature(species_id)
        mu, A = compute_LV_param(species_id, beta = beta, forage_attack= forage_attack.copy())
        
        N_0 = np.full(n_spec, 1e2)
        N_0[0] = species_id["K"]
        N_0_log = np.log(N_0)
        t_eval = np.linspace(0, 100, 101)
        sol = solve_ivp(LV_model, [0, np.inf],
                        t_eval = t_eval, y0 = N_0, method = "LSODA", args = (mu, A))
        #sol.y = np.exp(sol.y)
        rel_size = np.log(species_id["m_i"])
        rel_size = (rel_size - np.amin(rel_size))/(np.amax(rel_size)- np.amin(rel_size))
        colors = plt.cm.viridis(rel_size)
        for j in range(n_spec):
            if j == 0:
                ax[i].semilogy(sol.t, sol.y[j], 'k--')
            else:
                ax[i].semilogy(sol.t, sol.y[j], color = colors[j])
        ax[i].set_ylim([1e-1, 1e10])
    
        
        # compute trophic level of survivor
        """mod_solution = np.where(sol.y[:,-1] < 0, 0, sol.y[:,-1])
        diag_matrix = np.diag(mod_solution)
        community_matrix = -diag_matrix * A
        #print(diag_matrix)
        #print(community_matrix.shape)
        eigenvalues = np.linalg.eigvals(community_matrix).real
        neg_eigenvalues = eigenvalues[eigenvalues < 0]
        print(np.max(neg_eigenvalues))"""
        
        surv = sol.y[:,-1]>1
        length = sum(surv)
        #print(length)
        trophic_level = compute_trophic_level(species_id, surv)
        meantl = sum(trophic_level)/length
        ax2[i].loglog()
        ax2[i].scatter(species_id["m_i"][surv], trophic_level,
                       c = np.log(sol.y[surv,-1]))
    
        #"""
        
        