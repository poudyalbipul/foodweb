import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import solve_ivp

import Foodweb_model as fm

beta = fm.beta
def simulate_foodweb(beta_values, beta_index, n_spec=150, beta = beta):
    mean_length_list = []
    mean_trophic_list = []
    max_trophic_list = []
    var_trophic_list = []
    max_stabil_list = []
    biomass_resource_list = []
    total_biomass_list = []
    biomass_tl12_list = []
    biomass_tl23_list = []
    biomass_tl34_list = []
    biomass_tl4_list = []
    bioperbody_sub25_list = []
    bioperbody_2550_list = []
    bioperbody_5075_list = []
    bioperbody_up75_list = []
    
    for i in range(40):  # 9 realizations
        species_id = fm.generate_species(n_spec, random=True, B0=1e-6)
        length_list = []
        trophic_list = []
        max_list = []
        var_list = []
        stabil_list = []
        biomass_resource = []
        biomass_list = []
        biomass_12 = []
        biomass_23 = []
        biomass_34 = []
        biomass_4 = []
        bioperbody_sub25 = []
        bioperbody_2550 = []
        bioperbody_5075 = []
        bioperbody_up75 = []
        for beta_value in beta_values:
            beta[beta_index] = beta_value
            species_id = fm.change_temperature(species_id, beta=beta)
            mu, A = fm.compute_LV_param(species_id, beta=beta)            
            N_0 = np.full(n_spec, 1e2)
            N_0[0] = species_id["K"]
            N_0_log = np.log(N_0)
            t_eval = np.linspace(0, 100, 101)
            sol = solve_ivp(fm.LV_model, [0, np.inf], 
                            t_eval= t_eval, 
                            y0=N_0, method="LSODA", args=(mu, A))  
            
            survivors = sol.y[:, -1] > 1            
            biomass = sum(num for num in sol.y[1:,-1] if num > 1)
            
            ind_s = np.where(sol.y[:,-1] > 1)[0]
            A_surv = A[ind_s[:, np.newaxis], ind_s]
            eigenvalues = np.linalg.eigvals(-np.diag(sol.y[ind_s,-1])@A_surv).real
                       
            #if biomass > 10014900:
              #  print(np.sum(sol.y[:,-1]))    \             
           
            tlresource_mask = (fm.compute_trophic_level(species_id, survivors) ==  0)
           
            tlresource_biomass = np.sum(sol.y[survivors,-1][tlresource_mask]) 
         
            tlbasal_biomass_mask = ((fm.compute_trophic_level(species_id, survivors) > 0.6) &
                                 (fm.compute_trophic_level(species_id, survivors) <= 1))
            basal_biomass = np.sum(sol.y[:,-1][survivors][tlbasal_biomass_mask])    
            
            tl12_biomass_mask = ((fm.compute_trophic_level(species_id, survivors) > 1) &
                                 (fm.compute_trophic_level(species_id, survivors) <= 2))
            tl12_biomass = np.sum(sol.y[:,-1][survivors][tl12_biomass_mask])   
            
            tl23_biomass_mask = ((fm.compute_trophic_level(species_id, survivors) > 2) &
                                 (fm.compute_trophic_level(species_id, survivors) <= 3))
            tl23_biomass = np.sum(sol.y[:,-1][survivors][tl23_biomass_mask])  
            
            tl34_biomass_mask = ((fm.compute_trophic_level(species_id, survivors) > 3) &
                                 (fm.compute_trophic_level(species_id, survivors) <= 4))
            tl34_biomass = np.sum(sol.y[:,-1][survivors][tl34_biomass_mask]) 
            
            tl4_biomass_mask = (fm.compute_trophic_level(species_id, survivors) > 4)
            tl4_biomass = np.sum(sol.y[survivors,-1][tl4_biomass_mask])
            
            bio_per_body_sub25 = np.sum(sol.y[1:,-1][species_id["m_i"][1:] <= np.percentile(species_id["m_i"][1:],25)]>1)
            bio_per_body_2550 = np.sum(sol.y[(species_id["m_i"] > np.percentile(species_id["m_i"],25)) & 
                                             (species_id["m_i"] <= np.percentile(species_id["m_i"],50)) ,-1]>1)
            bio_per_body_5075 = np.sum(sol.y[(species_id["m_i"] > np.percentile(species_id["m_i"],50)) & 
                                             (species_id["m_i"] <= np.percentile(species_id["m_i"],75)) ,-1]>1)
            bio_per_body_up75 = np.sum(sol.y[species_id["m_i"] > np.percentile(species_id["m_i"], 75) ,-1]>1)
    
            
            if sum(survivors) == 1 or sum(survivors) == 150: 
                length_list.append(np.nan)
                trophic_list.append(np.nan)    
                max_list.append(np.nan)
                var_list.append(np.nan)
                stabil_list.append(np.nan)
                biomass_resource.append(np.nan)
                biomass_list.append(np.nan)
                biomass_12.append(np.nan)
                biomass_23.append(np.nan)
                biomass_34.append(np.nan)
                biomass_4.append(np.nan)
                bioperbody_sub25.append(np.nan)
                bioperbody_2550.append(np.nan)
                bioperbody_5075.append(np.nan)
                bioperbody_up75.append(np.nan)
            else:
                length_list.append(sum(survivors))            
                trophic_list.append(np.nanmedian(fm.compute_trophic_level(species_id, survivors)))         
                max_list.append(np.nanmax(fm.compute_trophic_level(species_id, survivors)))
                var_list.append(np.nanvar(fm.compute_trophic_level(species_id, survivors)))
                stabil_list.append(np.nanmax(eigenvalues))
                biomass_resource.append(tlresource_biomass/ (biomass + tlresource_biomass))
                biomass_list.append(basal_biomass/biomass)
                biomass_12.append(tl12_biomass/biomass)
                biomass_23.append(tl23_biomass/biomass)
                biomass_34.append(tl34_biomass/biomass)
                biomass_4.append(tl4_biomass/biomass)
                bioperbody_sub25.append(bio_per_body_sub25)
                bioperbody_2550.append(bio_per_body_2550)
                bioperbody_5075.append(bio_per_body_5075)
                bioperbody_up75.append(bio_per_body_up75)                                
                
                
        mean_length_list.append(length_list)
        mean_trophic_list.append(trophic_list)
        max_trophic_list.append(max_list)
        var_trophic_list.append(var_list)  
        max_stabil_list.append(stabil_list) 
        biomass_resource_list.append(biomass_resource)
        total_biomass_list.append(biomass_list)
        biomass_tl12_list.append(biomass_12)
        biomass_tl23_list.append(biomass_23)
        biomass_tl34_list.append(biomass_34)
        biomass_tl4_list.append(biomass_4)
        bioperbody_sub25_list.append(bioperbody_sub25)
        bioperbody_2550_list.append(bioperbody_2550)
        bioperbody_5075_list.append(bioperbody_5075)
        bioperbody_up75_list.append(bioperbody_up75)
    
    
    proportion_formed = 1- (np.isnan(mean_length_list).sum(axis=0)/len(mean_length_list))
   
    #print(total_biomass_list)
    
    return (np.nanmedian(mean_length_list, axis=0), 
            np.nanvar(mean_length_list, axis=0), 
            np.nanmean(mean_trophic_list, axis=0), 
            np.nanmean(max_trophic_list, axis = 0), 
            np.nanmean(var_trophic_list, axis = 0), 
            np.nanmedian(max_stabil_list, axis = 0), 
            np.nanvar(max_stabil_list, axis=0),
            proportion_formed,
            np.nanmedian(biomass_resource_list, axis = 0),
            np.nanmedian(total_biomass_list,axis = 0),
            np.nanmedian(biomass_tl12_list,axis = 0),
            np.nanmedian(biomass_tl23_list,axis = 0),
            np.nanmedian(biomass_tl34_list,axis = 0),
            np.nanmedian(biomass_tl4_list,axis = 0),
            np.nanmedian(bioperbody_sub25_list, axis =0),
            np.nanmedian(bioperbody_2550_list, axis = 0),
            np.nanmedian(bioperbody_5075_list, axis = 0),
            np.nanmedian(bioperbody_up75_list, axis = 0))

# Initialize parameters
beta_00 = np.sort(np.geomspace(beta[0, 0] / 2, 2 * beta[0, 0], 10))
beta_01 = np.sort(np.geomspace(beta[0, 1] / 2, 2 * beta[0, 1], 10))
beta_10 = np.sort(np.geomspace(beta[1, 0] / 2, 2 * beta[1, 0], 10))
beta_11 = np.sort(np.geomspace(beta[1, 1] / 2 ,2 * beta[1, 1], 10))
beta_20 = np.sort(np.geomspace(beta[2, 0] / 2, 2 * beta[2, 0], 10))
beta_21 = np.sort(np.geomspace(beta[2, 1] / 2, 2 * beta[2, 1], 10))
beta_30 = np.sort(np.geomspace(beta[3, 0] / 2, 2 * beta[3, 0], 10))
beta_31 = np.sort(np.geomspace(beta[3, 1] / 2, 2 * beta[3, 1], 10))
beta_40 = np.sort(np.geomspace(beta[4, 0] / 2, 2 * beta[4, 0], 10))

"""beta_00 = np.linspace(beta[0, 0] - 2.59, beta[0, 0] + 2.59, 10)
beta_01 = np.linspace(beta[0, 1] - 0.13667, beta[0, 1] + 0.13667, 10)
beta_10 = np.linspace(beta[1, 0] - 0.9865, beta[1, 0] + 0.9865, 10)
beta_11 = np.linspace(beta[1, 1] - 0.0001, beta[1, 1] + 0.0001, 10)
beta_20 = np.linspace(beta[2, 0] - 1.0883, beta[2, 0] + 1.0883, 10)
beta_21 = np.linspace(beta[2, 1] - 0.03114, beta[2, 1] + 0.03114, 10)
beta_30 = np.linspace(beta[3, 0] - 0, beta[3, 0] + 0, 10)
beta_31 = np.linspace(beta[3, 1] - 0, beta[3, 1] + 0, 10)"""

"""beta_00 = np.linspace(beta[0, 0] - 0.5, beta[0, 0] + 0.5, 10)
beta_01 = np.linspace(beta[0, 1] - 0.5, beta[0, 1] + 0.5, 10)
beta_10 = np.linspace(beta[1, 0] - 0.5, beta[1, 0] + 0.5, 10)
beta_11 = np.linspace(beta[1, 1] - 0.5, beta[1, 1] + 0.5, 10)
beta_20 = np.linspace(beta[2, 0] - 0.5, beta[2, 0] + 0.5, 10)
beta_21 = np.linspace(beta[2, 1] - 0.5, beta[2, 1] + 0.5, 10)
beta_30 = np.sort(np.geomspace(beta[3, 0] - 0.5, beta[3, 0] + 0.5, 10))
beta_31 = np.sort(np.linspace(beta[3, 1] - 0.5, beta[3, 1] + 0.5, 10))
beta_40 = np.linspace(0, 0.5, 20)"""

# Run simulations
"""(mean_length_00, var_length_00, mean_trophic_00, max_trophic_00, var_trophic_00, max_stabil_00, 
var_stabil_00, proportion_00, resource_biomass_00, basal_biomass_00, tl12_biomass_00, tl23_biomass_00, tl34_biomass_00,
tl4_biomass_00, bodymass_sub25_00, bodymass_2550_00, bodymass_5075_00, bodymass_up75_00) = simulate_foodweb(beta_00, beta_index=(0, 0))"""

#beta[0,0]/=  (5/3)  
beta[0,0] = -3.1503  #aquatic
#beta[0,0] = -0.5601
"""(mean_length_01, var_length_01, mean_trophic_01, max_trophic_01, var_trophic_01, max_stabil_01,
var_stabil_01, proportion_01, resource_biomass_01, basal_biomass_01, tl12_biomass_01, tl23_biomass_01, tl34_biomass_01,
tl4_biomass_01, bodymass_sub25_01, bodymass_2550_01, bodymass_5075_01, bodymass_up75_01) = simulate_foodweb(beta_01, beta_index=(0, 1))"""
#beta[0,1]/=  (5/3)   
beta[0,1] = 0.234519  

"""(mean_length_10, var_length_10, mean_trophic_10, max_trophic_10, var_trophic_10, max_stabil_10, 
var_stabil_10, proportion_10, resource_biomass_10, basal_biomass_10, tl12_biomass_10, tl23_biomass_10, tl34_biomass_10,
tl4_biomass_10, bodymass_sub25_10, bodymass_2550_10, bodymass_5075_10, bodymass_up75_10) = simulate_foodweb(beta_10, beta_index=(1, 0))"""
#beta[1,0]/=  (5/3)   
beta[1,0] = 1.6645 
#beta[1,0] = 0.6780

(mean_length_11, var_length_11, mean_trophic_11, max_trophic_11, var_trophic_11, max_stabil_11, 
var_stabil_11, proportion_11, resource_biomass_11, basal_biomass_11, tl12_biomass_11, tl23_biomass_11, tl34_biomass_11,
tl4_biomass_11, bodymass_sub25_11, bodymass_2550_11, bodymass_5075_11, bodymass_up75_11) = simulate_foodweb(beta_11, beta_index=(1, 1))
#beta[1,1]/=  (5/3)   
beta[1,1] = 0.09433

"""(mean_length_20, var_length_20, mean_trophic_20, max_trophic_20, var_trophic_20, max_stabil_20, 
var_stabil_20, proportion_20, resource_biomass_20, basal_biomass_20, tl12_biomass_20, tl23_biomass_20, tl34_biomass_20,
tl4_biomass_20, bodymass_sub25_20, bodymass_2550_20, bodymass_5075_20, bodymass_up75_20)  = simulate_foodweb(beta_20, beta_index=(2, 0))"""
#beta[2,0]/=  (5/3)   
beta[2,0] = -1.1012   #aquatic
#beta[2,0] = -0.0219   #terrestrial
"""(mean_length_21, var_length_21, mean_trophic_21, max_trophic_21, var_trophic_21, max_stabil_21, 
var_stabil_21, proportion_21, resource_biomass_21, basal_biomass_21, tl12_biomass_21, tl23_biomass_21, tl34_biomass_21,
tl4_biomass_21, bodymass_sub25_21, bodymass_2550_21, bodymass_5075_21, bodymass_up75_21) = simulate_foodweb(beta_21, beta_index=(2, 1))"""
#beta[2,1]/=  (5/3)
beta[2,1] = -0.066577

"""(mean_length_30, var_length_30, mean_trophic_30, max_trophic_30, var_trophic_30, max_stabil_30, 
var_stabil_30, proportion_30, basal_biomass_30, tl12_biomass_30, tl23_biomass_30, tl34_biomass_30,
tl4_biomass_30, bodymass_sub25_30, bodymass_2550_30, bodymass_5075_30, bodymass_up75_30) = simulate_foodweb(beta_30, beta_index=(3, 0))"""

beta[3,0] = -0.037
"""(mean_length_31, var_length_31, mean_trophic_31, max_trophic_31, var_trophic_31, max_stabil_31, 
var_stabil_31, proportion_31, basal_biomass_31, tl12_biomass_31, tl23_biomass_31, tl34_biomass_31,
tl4_biomass_31, bodymass_sub25_31, bodymass_2550_31, bodymass_5075_31, bodymass_up75_31)  = simulate_foodweb(beta_31, beta_index=(3, 1))"""
#beta[3,1]/= (5/3)
beta[3,1] = -0.04
#mean_length_40, var_length_40, mean_trophic_40, max_trophic_40, var_trophic_40, max_stabil_40, var_stabil_40 = simulate_foodweb(beta_40, beta_index=(4, 0))
beta[4,0] = 0.5
#beta[4,0] = 10**-6

   
# Plot results
fig, ax1 = plt.subplots()
#fig2,ax2 = plt.subplots()
#fig3,ax3 = plt.subplots()
#fig4, ax4 = plt.subplots()
#ax1.plot(fm.normalize_x(beta_00), mean_length_00, label="$\\alpha_{f}$")
#ax1.plot(fm.normalize_x(beta_01), mean_length_01, label="$\\beta_{f}$")
#ax1.plot(fm.normalize_x(beta_10), mean_length_10, label="$\\alpha_{\\sigma}$")
ax1.plot(fm.normalize_x(beta_11), max_stabil_11, label="$\\beta_{\\sigma}$")
#ax1.plot(fm.normalize_x(beta_20), mean_length_20, label="$\\alpha_{\\theta}$")
#ax1.plot(fm.normalize_x(beta_21), mean_length_21, label="$\\beta_{\\theta}$")
#ax1.plot(fm.normalize_x(beta_30), max_stabil_30, label="$\\alpha_{x}$")
#ax1.plot(fm.normalize_x(beta_31), max_stabil_31, label="$\\beta_{x}$")
#ax1.plot(beta_40, max_stabil_40, label="Cintra")
ax1.set_xlabel("Change from mean")
#ax1.set_ylabel("Species Richness")
ax1.set_ylabel("Stability")
#ax1.set_xlabel("Intraspecific competition ")
#ax1.set_ylim([0,150])

#plt.tight_layout()
#plt.show()

forage_attack = fm.forage_attack
def simulate_foodweb2(forage_values, forage_key, n_spec=150, forage_attack = forage_attack):
    mean_length_list = []
    mean_trophic_list = []
    max_trophic_list = []
    var_trophic_list = []
    max_stabil_list = []
    biomass_resource_list = []
    total_biomass_list = []
    biomass_tl12_list = []
    biomass_tl23_list = []
    biomass_tl34_list = []
    biomass_tl4_list = []
    bioperbody_sub25_list = []
    bioperbody_2550_list = []
    bioperbody_5075_list = []
    bioperbody_up75_list = []
    for i in range(50):  # 9 realizations
        species_id = fm.generate_species(n_spec, random=True, B0=1e-6)
        length_list = []
        trophic_list = []
        max_list = []
        var_list = []
        stabil_list = []
        biomass_resource = []
        biomass_list = []
        biomass_12 = []
        biomass_23 = []
        biomass_34 = []
        biomass_4 = []
        bioperbody_sub25 = []
        bioperbody_2550 = []
        bioperbody_5075 = []
        bioperbody_up75 = []   
        
        for forage_value in forage_values:
            forage_attack[forage_key] = forage_value
            species_id = fm.change_temperature(species_id, beta=beta)
            mu, A = fm.compute_LV_param(species_id, beta=beta, forage_attack = forage_attack)            
            N_0 = np.full(n_spec, 1e2)
            N_0[0] = species_id["K"]
            N_0_log = np.log(N_0)
            t_eval = np.linspace(0, 100, 101)
            sol = solve_ivp(fm.LV_model, [0, np.inf], 
                            t_eval= t_eval, 
                            y0=N_0, method="LSODA", args=(mu, A))            
           
            survivors = sol.y[:, -1] > 1  
            biomass = sum(num for num in sol.y[1:,-1] if num > 1)
            
            ind_s = np.where(sol.y[:,-1] > 1)[0]
            A_surv = A[ind_s[:, np.newaxis], ind_s]
            eigenvalues = np.linalg.eigvals(-np.diag(sol.y[ind_s,-1])@A_surv).real
            
            tlresource_mask = (fm.compute_trophic_level(species_id, survivors) ==  0)
            tlresource_biomass = np.sum(sol.y[survivors,-1][tlresource_mask]) 
            
            tlbasal_biomass_mask = ((fm.compute_trophic_level(species_id, survivors) > 0.8) &
                                 (fm.compute_trophic_level(species_id, survivors) <= 1))
            basal_biomass = np.sum(sol.y[:,-1][survivors][tlbasal_biomass_mask])    
            
            tl12_biomass_mask = ((fm.compute_trophic_level(species_id, survivors) > 1) &
                                 (fm.compute_trophic_level(species_id, survivors) <= 2))
            tl12_biomass = np.sum(sol.y[:,-1][survivors][tl12_biomass_mask])   
            
            tl23_biomass_mask = ((fm.compute_trophic_level(species_id, survivors) > 2) &
                                 (fm.compute_trophic_level(species_id, survivors) <= 3))
            tl23_biomass = np.sum(sol.y[:,-1][survivors][tl23_biomass_mask])  
            
            tl34_biomass_mask = ((fm.compute_trophic_level(species_id, survivors) > 3) &
                                 (fm.compute_trophic_level(species_id, survivors) <= 4))
            tl34_biomass = np.sum(sol.y[:,-1][survivors][tl34_biomass_mask]) 
            
            tl4_biomass_mask = (fm.compute_trophic_level(species_id, survivors) > 4)
            tl4_biomass = np.sum(sol.y[survivors,-1][tl4_biomass_mask])
            
            
            bio_per_body_sub25 = np.sum(sol.y[1:,-1][species_id["m_i"][1:] <= np.percentile(species_id["m_i"][1:],25)])
            bio_per_body_2550 = np.sum(sol.y[(species_id["m_i"] > np.percentile(species_id["m_i"],25)) & 
                                             (species_id["m_i"] <= np.percentile(species_id["m_i"],50)) ,-1])
            bio_per_body_5075 = np.sum(sol.y[(species_id["m_i"] > np.percentile(species_id["m_i"],50)) & 
                                             (species_id["m_i"] <= np.percentile(species_id["m_i"],75)) ,-1])
            bio_per_body_up75 = np.sum(sol.y[species_id["m_i"] > np.percentile(species_id["m_i"], 75) ,-1])
            
            if sum(survivors) == 1 or sum(survivors) == 150: 
                length_list.append(np.nan)
                trophic_list.append(np.nan)    
                max_list.append(np.nan)
                var_list.append(np.nan)
                stabil_list.append(np.nan)
                biomass_resource.append(np.nan)
                biomass_list.append(np.nan)
                biomass_12.append(np.nan)
                biomass_23.append(np.nan)
                biomass_34.append(np.nan)
                biomass_4.append(np.nan)
                bioperbody_sub25.append(np.nan)
                bioperbody_2550.append(np.nan)
                bioperbody_5075.append(np.nan)
                bioperbody_up75.append(np.nan)
            else:
                length_list.append(sum(survivors))            
                trophic_list.append(np.nanmedian(fm.compute_trophic_level(species_id, survivors)))         
                max_list.append(np.nanmax(fm.compute_trophic_level(species_id, survivors)))
                var_list.append(np.nanvar(fm.compute_trophic_level(species_id, survivors)))
                stabil_list.append(np.nanmax(eigenvalues))
                biomass_resource.append(tlresource_biomass/ (biomass + tlresource_biomass))
                biomass_list.append(basal_biomass/biomass)
                biomass_12.append(tl12_biomass/biomass)
                biomass_23.append(tl23_biomass/biomass)
                biomass_34.append(tl34_biomass/biomass)
                biomass_4.append(tl4_biomass/biomass)
                bioperbody_sub25.append(bio_per_body_sub25/biomass)
                bioperbody_2550.append(bio_per_body_2550/biomass)
                bioperbody_5075.append(bio_per_body_5075/biomass)
                bioperbody_up75.append(bio_per_body_up75/biomass)         
                
        mean_length_list.append(length_list)
        mean_trophic_list.append(trophic_list)
        max_trophic_list.append(max_list)
        var_trophic_list.append(var_list)  
        max_stabil_list.append(stabil_list) 
        biomass_resource_list.append(biomass_resource)
        total_biomass_list.append(biomass_list)
        biomass_tl12_list.append(biomass_12)
        biomass_tl23_list.append(biomass_23)
        biomass_tl34_list.append(biomass_34)
        biomass_tl4_list.append(biomass_4)
        bioperbody_sub25_list.append(bioperbody_sub25)
        bioperbody_2550_list.append(bioperbody_2550)
        bioperbody_5075_list.append(bioperbody_5075)
        bioperbody_up75_list.append(bioperbody_up75)
        
        
    proportion_formed = 1- (np.isnan(mean_length_list).sum(axis=0)/len(mean_length_list))
    
    #print(total_biomass_list)
    
    return (np.nanmedian(mean_length_list, axis=0), 
        np.nanvar(mean_length_list, axis=0), 
        np.nanmean(mean_trophic_list, axis=0), 
        np.nanmean(max_trophic_list, axis = 0), 
        np.nanmean(var_trophic_list, axis = 0), 
        np.nanmedian(max_stabil_list, axis = 0), 
        np.nanvar(max_stabil_list, axis=0),
        proportion_formed,
        np.nanmedian(biomass_resource_list,axis = 0),
        np.nanmedian(total_biomass_list,axis = 0),
        np.nanmedian(biomass_tl12_list,axis = 0),
        np.nanmedian(biomass_tl23_list,axis = 0),
        np.nanmedian(biomass_tl34_list,axis = 0),
        np.nanmedian(biomass_tl4_list,axis = 0),
        np.nanmedian(bioperbody_sub25_list, axis =0),
        np.nanmedian(bioperbody_2550_list, axis = 0),
        np.nanmedian(bioperbody_5075_list, axis = 0),
        np.nanmedian(bioperbody_up75_list, axis = 0))

            

#initialize parameters

forage_1 = np.sort(np.geomspace(forage_attack["intercept"]/2, 2 * forage_attack["intercept"], 10))
forage_4 = np.sort(np.geomspace(forage_attack["log_pred"]/2, 2 * forage_attack["log_pred"], 10))
forage_5 = np.sort(np.geomspace(forage_attack["log_prey"]/2, 2 * forage_attack["log_prey"], 10))


"""forage_1 = np.linspace(forage_attack["intercept"] - 7.47, forage_attack["intercept"] + 7.47, 10)
forage_4 = np.linspace(forage_attack["log_pred"] - 0.49,  forage_attack["log_pred"] + 0.49, 10)
forage_5 = np.linspace(forage_attack["log_prey"] - 0.0495, forage_attack["log_prey"] + 0.0495, 10)"""

"""forage_1 = np.linspace(forage_attack["intercept"] - 0.5, forage_attack["intercept"] + 0.5, 10)
forage_4 = np.linspace(forage_attack["log_pred"] - 0.5,  forage_attack["log_pred"] + 0.5, 10)
forage_5 = np.linspace(forage_attack["log_prey"] - 0.5, forage_attack["log_prey"] + 0.5, 10)"""


#Run simulations
"""(mean_length_1, var_length_1, mean_trophic_1, max_trophic_1, var_trophic_1, max_stabil_1, 
var_stabil_1, proportion_1, resource_biomass_1, basal_biomass_1, tl12_biomass_1, tl23_biomass_1, tl34_biomass_1,
tl4_biomass_1, bodymass_sub25_1, bodymass_2550_1, bodymass_5075_1, bodymass_up75_1) = simulate_foodweb2(forage_1, forage_key = "intercept")"""

forage_attack["intercept"] = -11.83

"""(mean_length_4, var_length_4, mean_trophic_4, max_trophic_4, var_trophic_4, max_stabil_4, 
var_stabil_4, proportion_4, resource_biomass_4, basal_biomass_4, tl12_biomass_4, tl23_biomass_4, tl34_biomass_4,
tl4_biomass_4, bodymass_sub25_4, bodymass_2550_4, bodymass_5075_4, bodymass_up75_4) = simulate_foodweb2(forage_4, forage_key= "log_pred" )"""
    
forage_attack["log_pred"] = 0.54

"""(mean_length_5, var_length_5, mean_trophic_5, max_trophic_5, var_trophic_5, max_stabil_5, var_stabil_5, 
proportion_5, resource_biomass_5, basal_biomass_5, tl12_biomass_5, tl23_biomass_5, tl34_biomass_5,
tl4_biomass_5, bodymass_sub25_5, bodymass_2550_5, bodymass_5075_5, bodymass_up75_5) = simulate_foodweb2(forage_5, forage_key= "log_prey" )""" 
forage_attack["log_prey"] = 0.05 

#plot
#fig,ax = plt.subplots()
#fig2,ax2 = plt.subplots()

#ax1.plot(fm.normalize_x(forage_1), max_stabil_1, color="red", label = "$\\alpha_{a}$")
#ax1.plot(fm.normalize_x(forage_4), max_stabil_4, label = "$\\beta_{ai}$")
#ax1.plot(fm.normalize_x(forage_5), max_stabil_5, label = "$\\beta_{ak}$")
#ax.set_xlabel("Change from mean")
#ax.set_ylabel("Species Richness")
#ax.legend()
#ax.grid(True)

#ax2.plot(fm.normalize_x(forage_1), mean_trophic_1, label = "$\\alpha_{a}$")
#ax2.plot(fm.normalize_x(forage_4), mean_trophic_4, label = "$\\beta_{ai}$")
#ax2.plot(fm.normalize_x(forage_5), mean_trophic_5, label = "$\\beta_{ak}$")
#ax2.set_xlabel("Change from mean")
#ax2.set_ylabel("Mean Trophic Level")   
#ax2.legend()
#ax2.grid(True)

ax1.legend()
ax1.grid(True)
#ax1.set_title("change $\\pm$ 65%")
#fig.savefig("Cintra_stabil")

"""datapm65 = pd.DataFrame({
                         "alphaf_x":fm.normalize_x(beta_00), "alphaf_y":mean_length_00, 
                         "var_length":var_length_00, "alphaf_mtl": mean_trophic_00, "alphaf_maxtl":max_trophic_00, 
                         "alphaf_vartl":var_trophic_00,"alphaf_stabil":max_stabil_00, "alphaf_varstabil":var_stabil_00, 
                         "alphaf_proportion":proportion_00, "alphaf_resource":resource_biomass_00, "alphaf_basalbiomass":basal_biomass_00,
                         "alphaf_12biomass": tl12_biomass_00, "alphaf_23biomass": tl23_biomass_00,
                         "alphaf_34biomass": tl34_biomass_00, "alphaf_4biomass": tl4_biomass_00, "alphaf_sub25": bodymass_sub25_00,
                         "alphaf_2550": bodymass_2550_00, "alphaf_5075": bodymass_5075_00, "alphaf_up75": bodymass_up75_00 
                         })"""
    
"""datapm65 = pd.DataFrame({"betaf_x":fm.normalize_x(beta_01), "betaf_y":mean_length_01, "betaf_varlen":var_length_01, 
                         "betaf_mtl": mean_trophic_01, "betaf_maxtl":max_trophic_01, "betaf_vartl": var_trophic_01,
                         "betaf_stabil":max_stabil_01, "betaf_varstabil":var_stabil_01, "betaf_proportion":proportion_01, 
                         "betaf_resource": resource_biomass_01, "betaf_basalbiomass":basal_biomass_01, "betaf_12biomass": tl12_biomass_01, 
                         "betaf_23biomass": tl23_biomass_01, "betaf_34biomass": tl34_biomass_01, 
                         "betaf_4biomass": tl4_biomass_01, "betaf_sub25":bodymass_sub25_01, "betaf_2550":bodymass_2550_01,
                         "betaf_5075": bodymass_5075_01, "betaf_up75": bodymass_up75_01
                         })"""


"""datapm65 = pd.DataFrame({"alphasigma_x":fm.normalize_x(beta_10), "alphasigma_y":mean_length_10, "alphasig_varlen":var_length_10, 
                         "alphasig_mtl": mean_trophic_10, "alphasig_maxtl":max_trophic_10, "alphasig_vartl":var_trophic_10, 
                         "alphasig_stabil":max_stabil_10, "alphasig_varstabil":var_stabil_10, "alphasig_proportion":proportion_10,                          
                         "alphasig_resource":resource_biomass_10, "alphasig_basalbiomass":basal_biomass_10, 
                         "alphasig_12biomass": tl12_biomass_10, "alphasig_23biomass": tl23_biomass_10, "alphasig_34biomass": tl34_biomass_10, 
                         "alphasig_4biomass": tl4_biomass_10, "alphasig_sub25": bodymass_sub25_10, "alphasig_2550": bodymass_2550_10, 
                         "alphasig_5075": bodymass_5075_10, "alphasig_up75": bodymass_up75_10   
                         })"""


datapm65 = pd.DataFrame({"betasig_x":fm.normalize_x(beta_11), "betasig_y":mean_length_11, "betasig_varlen":var_length_11,
                         "betasig_mtl": mean_trophic_11, "betasig_maxtl":max_trophic_11, "betasig_vartl":var_trophic_11,
                         "betasig_stabil":max_stabil_11, "betasig_varstabil":var_stabil_11, "betasig_proportion":proportion_11, 
                         "betasig_resource": resource_biomass_11, "betasig_basalbiomass":basal_biomass_11, "betasig_12biomass": tl12_biomass_11, 
                         "betasig_23biomass": tl23_biomass_11, "betasig_34biomass": tl34_biomass_11, "betasig_4biomass": tl4_biomass_11, 
                         "betasig_sub25":bodymass_sub25_11, "betasig_2550":bodymass_2550_11, "betasig_5075": bodymass_5075_11, 
                         "betasig_up75": bodymass_up75_11 
                         })

"""datapm65 = pd.DataFrame({"alphatheta_x":fm.normalize_x(beta_20), "alphatheta_y":mean_length_20, "alphatheta_varlen":var_length_20,
                         "alphatheta_mtl": mean_trophic_20, "alphatheta_maxtl":max_trophic_20, "alphatheta_vartl":var_trophic_20, 
                         "alphatheta_stabil":max_stabil_20, "alphatheta_varstabil":var_stabil_20, "alphatheta_proportion":proportion_20, 
                         "alphatheta_resource": resource_biomass_20, "alphatheta_basalbiomass":basal_biomass_20, 
                         "alphatheta_12biomass": tl12_biomass_20, "alphatheta_23biomass": tl23_biomass_20, 
                         "alphatheta_34biomass": tl34_biomass_20, "alphatheta_4biomass": tl4_biomass_20, "alphatheta_sub25": bodymass_sub25_20, 
                         "alphatheta_2550": bodymass_2550_20, "alphatheta_5075": bodymass_5075_20, "alphatheta_up75": bodymass_up75_20 
                         })"""
    
"""datapm65 = pd.DataFrame({"betatheta_x":fm.normalize_x(beta_21), "betatheta_y":mean_length_21, "betatheta_varlen":var_length_21, 
                         "betatheta_mtl": mean_trophic_21, "betatheta_maxtl":max_trophic_21, "betatheta_vartl":var_trophic_21,
                         "betatheta_stabil":max_stabil_21, "betatheta_varstabil":var_stabil_21, "betatheta_proportion":proportion_21, 
                         "betatheta_resource": resource_biomass_21,"betatheta_basalbiomass":basal_biomass_21, 
                         "betatheta_12biomass": tl12_biomass_21, "betatheta_23biomass": tl23_biomass_21, "betatheta_34biomass": tl34_biomass_21, 
                         "betatheta_4biomass": tl4_biomass_21, "betatheta_sub25":bodymass_sub25_21, "betatheta_2550":bodymass_2550_21,
                         "betatheta_5075": bodymass_5075_21, "betatheta_up75": bodymass_up75_21  })"""


"""datapm65 = pd.DataFrame({"alphax_x":fm.normalize_x(beta_30), "alphax_y":mean_length_30, "alphax_varlen":var_length_30,
                         "alphax_mtl": mean_trophic_30, "alphax_maxtl":max_trophic_30, "alphax_vartl":var_trophic_30, 
                         "alphax_stabil":max_stabil_30, "alphax_varstabil":var_stabil_30, "alphax_proportion":proportion_30, 
                         "alphax_basalbiomass":basal_biomass_30, "alphax_12biomass": tl12_biomass_30, 
                         "alphax_23biomass": tl23_biomass_30, "alphax_34biomass": tl34_biomass_30, "alphax_4biomass": tl4_biomass_30, 
                         "alphax_sub25": bodymass_sub25_30, "alphax_2550": bodymass_2550_30, "alphax_5075": bodymass_5075_30, 
                         "alphax_up75": bodymass_up75_30 
                         })"""


"""datapm65 = pd.DataFrame({"betax_x":fm.normalize_x(beta_31), "betax_y":mean_length_31, "betax_varlen":var_length_31, 
                         "betax_mtl": mean_trophic_31, "betax_maxtl":max_trophic_31, "betax_vartl":var_trophic_31, 
                         "betax_stabil":max_stabil_31, "betax_varstabil":var_stabil_31, "betax_proportion":proportion_31, 
                         "betax_basalbiomass":basal_biomass_31, "betaax_12biomass": tl12_biomass_31, 
                         "betax_23biomass": tl23_biomass_31, "betax_34biomass": tl34_biomass_31, "betax_4biomass": tl4_biomass_31, 
                         "betax_sub25": bodymass_sub25_31, "betax_2550": bodymass_2550_31, "betax_5075": bodymass_5075_31, 
                         "alphax_up75": bodymass_up75_31 
                         }) """

"""datapm65 = pd.DataFrame({"alphaa_x":fm.normalize_x(forage_1), "alphaa_y":mean_length_1, "alphaa_varlen":var_length_1,
                         "alphaa_mtl": mean_trophic_1, "alphaa_maxtl":max_trophic_1, "alphaa_vartl":var_trophic_1, 
                         "alphaa_stabil":max_stabil_1, "alphaa_varstabil":var_stabil_1, "alphaa_proportion":proportion_1,
                         "alphaa_resource" :resource_biomass_1, "alphaa_basalbiomass":basal_biomass_1, "alphaa_12biomass": tl12_biomass_1, 
                         "alphaa_23biomass": tl23_biomass_1, "alphaa_34biomass": tl34_biomass_1, "alphaa_4biomass": tl4_biomass_1, 
                         "alphaa_sub25": bodymass_sub25_1, "alphaa_2550": bodymass_2550_1, "alphaa_5075": bodymass_5075_1, 
                         "alphaa_up75": bodymass_up75_1                            
                         })"""

"""datapm65 = pd.DataFrame({"betaai_x":fm.normalize_x(forage_4), "betaai_y":mean_length_4, "betaai_varlen":var_length_4,
                         "betaai_mtl": mean_trophic_4, "betaai_maxtl":max_trophic_4, "betaai_vartl":var_trophic_4, 
                         "betaai_stabil":max_stabil_4, "betaai_varstabil":var_stabil_4, "betaai_proportion":proportion_4, 
                         "betaai_resource":resource_biomass_4, "betaai_basalbiomass":basal_biomass_4, "betaai_12biomass": tl12_biomass_4, 
                         "betaai_23biomass": tl23_biomass_4, "betaai_34biomass": tl34_biomass_4, 
                         "betaai_4biomass": tl4_biomass_4, "betaai_sub25":bodymass_sub25_4, "betaai_2550":bodymass_2550_4,
                         "betaai_5075": bodymass_5075_4, "betaai_up75": bodymass_up75_4                         
                         })"""

"""datapm65 = pd.DataFrame({"betaak_x":fm.normalize_x(forage_5), "betaak_y":mean_length_5, "betaak_varlen":var_length_5,
                         "betaak_mtl": mean_trophic_5, "betaak_maxtl":max_trophic_5, "betaak_vartl":var_trophic_5, 
                         "betaak_stabil":max_stabil_5, "betaak_varstabil":var_stabil_5, "betaak_proportion":proportion_5, 
                         "betaak_basalbiomass":basal_biomass_5, "betaak_12biomass": tl12_biomass_5, 
                         "betaak_23biomass": tl23_biomass_5, "betaak_34biomass": tl34_biomass_5, 
                         "betaak_4biomass": tl4_biomass_5, "betaak_sub25":bodymass_sub25_5, "betaak_2550":bodymass_2550_5,
                         "betaak_5075": bodymass_5075_5, "betaak_up75": bodymass_up75_5                                                      
                         })"""


#datapm65 = pd.DataFrame({"Cintra_x":beta_40, "Cintra_y":mean_length_40, "Cintra_varlen":var_length_40,"Cintra_mtl": mean_trophic_40, 
#"Cintra_maxtl":max_trophic_40, "Cintra_vartl":var_trophic_40, "Cintra_stabil":max_stabil_40, "Cintra_varstabil":var_stabil_40})


excel_path = "beta_sigmanew.xlsx"
datapm65.to_excel(excel_path, index=False)

"""species_id = fm.generate_species(n_spec =100, random=True, B0=1e-6)
species_id = fm.change_temperature(species_id, beta=beta)
def compute_community_matrix(n_spec = 100, beta = beta, species_id = species_id):
       
    mu, A = fm.compute_LV_param(species_id, beta=beta)            
    N_0 = np.full(n_spec, 1e2)
    N_0[0] = species_id["K"]
    N_0_log = np.log(N_0)
    t_eval = np.linspace(0, 100, 101)
    rel_size = np.log(species_id["m_i"])
    #rel_size = (rel_size - np.amin(rel_size))/(np.amax(rel_size)- np.amin(rel_size))
    
    sol = solve_ivp(fm.LV_model, [0, np.inf], 
                    t_eval= t_eval, 
                    y0=N_0, method="LSODA", args=(mu, A))  
    
    
    ind_s = np.where(sol.y[:,-1] > 1)[0]
    A_surv = A[ind_s[:, np.newaxis], ind_s]
    community_matrix = -np.diag(sol.y[ind_s,-1])@A_surv
    eiga, eigb = np.linalg.eig(-np.diag(sol.y[ind_s,-1])@A_surv)    
    
    
    return community_matrix, sol.y[:,-1], eigb, rel_size[ind_s]

community_matrix, survivors, eigb, rel_size = compute_community_matrix(n_spec=100)

extent = np.percentile(rel_size, [0,100,0,100])
cmap3 = ax4.imshow((community_matrix), vmin = -0.01, vmax= 0.01, extent = extent, origin ="lower")
#cmap3 = ax4.imshow((A), vmin = -0.01, vmax= 0.01, extent = extent, origin ="lower")
fig4.colorbar(cmap3, ax = ax4)
ax4.set_xlabel("Predator Bodymass [Log]")
ax4.set_ylabel("Prey bodymass [Log]")   
ax4.set_title("Interaction strength")
#ax4.legend()
ax4.grid(True)"""

##########################################     
