#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 28 12:03:10 2025

@author: bipul
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import solve_ivp
import Foodweb_model as fm
import networkx as nx

beta = fm.beta

#keywords = ['mean_length', 'mean_trophic'] # tzpe all the kezwords

def simulate_foodweb(beta_values, beta_index, n_spec=150, beta = beta):
    
    #dict_list = {key: [] for key in keywords}
    #initialize a list
    mean_length_list = []
    mean_trophic_list = []
    weighted_trophic_list = []
    max_trophic_list = []
    var_trophic_list = []
    max_stabil_list = []
    biomass_resource_list = []
    total_biomass_list = []
    sum_biomass_list = []
    biomass_tl1_list = []
    biomass_tl12_list = []
    biomass_tl23_list = []
    biomass_tl34_list = []
    biomass_tl4_list = []
    bioperbody_sub25_list = []
    bioperbody_2550_list = []
    bioperbody_5075_list = []
    bioperbody_up75_list = []
    linksperbody_sub25_list = []
    linksperbody_2550_list = []
    linksperbody_5075_list = []
    linksperbody_up75_list = []
    sum_link_list = []    
    
    for i in range(40):  # number of realizations
        species_id = fm.generate_species(n_spec, random=True, B0=1e-6, K =1e7, beta = beta)
        #initialize list
        length_list = []
        trophic_list = []
        weighted_list = []
        max_list = []
        var_list = []
        stabil_list = []
        biomass_resource = []
        biomass_list = []
        sum_list = []
        biomass_1 = []
        biomass_12 = []
        biomass_23 = []
        biomass_34 = []
        biomass_4 = []
        bioperbody_sub25 = []
        bioperbody_2550 = []
        bioperbody_5075 = []
        bioperbody_up75 = []
        linksperbody_sub25 = []
        linksperbody_2550 = []
        linksperbody_5075 = []
        linksperbody_up75 = []
        sum_links = []
        #dict_runs = {key: [] for key in keywords}
        
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
            
            #compute number of survivors and total population
            survivors = sol.y[:, -1] > 1            
            biomass = sum(num for num in sol.y[1:,-1] if num > 1)            
            #diversity = np.sum(survivors)   
            
            
            #compute stability
            A = A #/ np.linalg.norm(A, ord='fro')      #normalize A
            ind_s = np.where(sol.y[:,-1] > 1)[0]
            A_surv = A[ind_s[:, np.newaxis], ind_s]
            J = -np.diag(sol.y[ind_s,-1])@A_surv       
            J_norm = J 
            eigenvalues = np.linalg.eigvals(J_norm).real
            #stability = np.nanmax(eigenvalues)
            
            ##computing minimum distance to basal species
            B = fm.compute_links(species_id).T
            G = nx.DiGraph(B)
            
            # Step 2: Identify basal species (nodes with no outgoing edges in original graph)
            basal_species = [i for i in range(A.shape[0]) if np.sum(A[i, :]) == 0]

            # Step 3: For each node, find shortest path length to any basal species
            min_paths_to_basal = {}
                       
            for species in range(A.shape[0]):
                lengths = []
                for basal in basal_species:
                    try:
                        length = nx.shortest_path_length(G, source=species, target=basal)
                        lengths.append(length)
                    except nx.NetworkXNoPath:
                        continue  # No path to this basal
                min_paths_to_basal[species] = min(lengths) if lengths else np.inf  # inf = no path to any basal
                
            minimum_path = np.array(list(min_paths_to_basal.values()))    
                  
            #compute biomass for differennt trophic levels                  
            tlbasal_mask = (fm.compute_trophic_level(species_id, survivors) ==  0)           
            tl1_biomass_mask = ((fm.compute_trophic_level(species_id, survivors) > 0.6) &
                                 (fm.compute_trophic_level(species_id, survivors) <= 1))
            tl12_biomass_mask = ((fm.compute_trophic_level(species_id, survivors) > 1) &
                                 (fm.compute_trophic_level(species_id, survivors) <= 2))            
            tl23_biomass_mask = ((fm.compute_trophic_level(species_id, survivors) > 2) &
                                 (fm.compute_trophic_level(species_id, survivors) <= 3))
            tl34_biomass_mask = ((fm.compute_trophic_level(species_id, survivors) > 3) &
                                (fm.compute_trophic_level(species_id, survivors) <= 4))            
            tl4_biomass_mask = (fm.compute_trophic_level(species_id, survivors) > 4)
            
            tlbasal_biomass = np.sum(sol.y[survivors,-1][tlbasal_mask]) 
            tl1_biomass = np.sum(sol.y[:,-1][survivors][tl1_biomass_mask])    
            tl12_biomass = np.sum(sol.y[:,-1][survivors][tl12_biomass_mask])
            tl23_biomass = np.sum(sol.y[:,-1][survivors][tl23_biomass_mask])  
            tl34_biomass = np.sum(sol.y[:,-1][survivors][tl34_biomass_mask])         
            tl4_biomass = np.sum(sol.y[survivors,-1][tl4_biomass_mask])
            
            #compute biomass per bodymass quantiles
            bio_per_body_sub25 = np.sum(sol.y[1:,-1][species_id["m_i"][1:] <= np.percentile(species_id["m_i"][1:],25)]>1)
            bio_per_body_2550 = np.sum(sol.y[(species_id["m_i"] > np.percentile(species_id["m_i"],25)) & 
                                             (species_id["m_i"] <= np.percentile(species_id["m_i"],50)) ,-1]>1)
            bio_per_body_5075 = np.sum(sol.y[(species_id["m_i"] > np.percentile(species_id["m_i"],50)) & 
                                             (species_id["m_i"] <= np.percentile(species_id["m_i"],75)) ,-1]>1)
            bio_per_body_up75 = np.sum(sol.y[species_id["m_i"] > np.percentile(species_id["m_i"], 75) ,-1]>1)
            
            #compute weighted trophic level
            weighted_tl = np.sum(fm.compute_trophic_level(species_id,survivors) * sol.y[:,-1][survivors]) / np.sum( sol.y[:,-1][survivors])
            
            #compute minimum path to basal resource
            links_per_body_sub25 = np.nanmean(minimum_path[1:][(species_id["m_i"][1:] <= np.percentile(species_id["m_i"][1:],25)) & survivors[1:]])
            
            links_per_body_2550 = np.nanmean(minimum_path[1:][(species_id["m_i"][1:] > np.percentile(species_id["m_i"][1:],25))
                                                                            & (species_id["m_i"][1:] <= np.percentile(species_id["m_i"][1:],50)) & survivors[1:]])            
            
            links_per_body_5075 = np.nanmean(minimum_path[1:][(species_id["m_i"][1:] > np.percentile(species_id["m_i"][1:],50))
                                                                            & (species_id["m_i"][1:] <= np.percentile(species_id["m_i"][1:],75)) & survivors[1:]])
            
            links_per_body_up75 = np.nanmean(minimum_path[1:][(species_id["m_i"][1:] > np.percentile(species_id["m_i"][1:],75)) & survivors[1:]])
            
            #total number of links
            total_links = np.sum(fm.compute_links(species_id).T[survivors][:,survivors])
            
            if sum(survivors) == 1 or sum(survivors) == 150:
                #for key in keywords:
                # dict_runs[key].append(np.nan)
                length_list.append(np.nan)
                trophic_list.append(np.nan)  
                weighted_list.append(np.nan)
                max_list.append(np.nan)
                var_list.append(np.nan)
                stabil_list.append(np.nan)
                biomass_resource.append(np.nan)
                biomass_list.append(np.nan)
                sum_list.append(np.nan)
                biomass_1.append(np.nan)
                biomass_12.append(np.nan)
                biomass_23.append(np.nan)
                biomass_34.append(np.nan)
                biomass_4.append(np.nan)
                bioperbody_sub25.append(np.nan)
                bioperbody_2550.append(np.nan)
                bioperbody_5075.append(np.nan)
                bioperbody_up75.append(np.nan)
                linksperbody_sub25.append(np.nan)
                linksperbody_2550.append(np.nan)
                linksperbody_5075.append(np.nan)
                linksperbody_up75.append(np.nan)
                sum_links.append(np.nan)
            else:
                length_list.append(sum(survivors))            
                trophic_list.append(np.nanmedian(fm.compute_trophic_level(species_id, survivors)))      
                weighted_list.append(weighted_tl)
                max_list.append(np.nanmax(fm.compute_trophic_level(species_id, survivors)))
                var_list.append(np.nanvar(fm.compute_trophic_level(species_id, survivors)))
                stabil_list.append(np.nanmax(eigenvalues))
                biomass_resource.append(tlbasal_biomass)
                biomass_list.append(biomass)
                sum_list.append(biomass + tlbasal_biomass)
                
                
                biomass_1.append(tl1_biomass/biomass)
                biomass_12.append(tl12_biomass/biomass)
                biomass_23.append(tl23_biomass/biomass)
                biomass_34.append(tl34_biomass/biomass)
                biomass_4.append(tl4_biomass/biomass)
                
                bioperbody_sub25.append(bio_per_body_sub25)
                bioperbody_2550.append(bio_per_body_2550)
                bioperbody_5075.append(bio_per_body_5075)
                bioperbody_up75.append(bio_per_body_up75) 
                               
                linksperbody_sub25.append(links_per_body_sub25)
                linksperbody_2550.append(links_per_body_2550)
                linksperbody_5075.append(links_per_body_5075)
                linksperbody_up75.append(links_per_body_up75) 
                sum_links.append(total_links)
        
       # for key in keywords:
        #    dict_mean[key].append(dict_runs[key])
        mean_length_list.append(length_list)
        mean_trophic_list.append(trophic_list)
        weighted_trophic_list.append(weighted_list)
        max_trophic_list.append(max_list)
        var_trophic_list.append(var_list)  
        max_stabil_list.append(stabil_list) 
        biomass_resource_list.append(biomass_resource)
        total_biomass_list.append(biomass_list)
        sum_biomass_list.append(sum_list)
        biomass_tl1_list.append(biomass_1)
        biomass_tl12_list.append(biomass_12)
        biomass_tl23_list.append(biomass_23)
        biomass_tl34_list.append(biomass_34)
        biomass_tl4_list.append(biomass_4)
        bioperbody_sub25_list.append(bioperbody_sub25)
        bioperbody_2550_list.append(bioperbody_2550)
        bioperbody_5075_list.append(bioperbody_5075)
        bioperbody_up75_list.append(bioperbody_up75)
        linksperbody_sub25_list.append(linksperbody_sub25)
        linksperbody_2550_list.append(linksperbody_2550)
        linksperbody_5075_list.append(linksperbody_5075)
        linksperbody_up75_list.append(linksperbody_up75)
        sum_link_list.append(sum_links)
      
    proportion_formed = 1- (np.isnan(mean_length_list).sum(axis=0)/len(mean_length_list))
    
    #return community metrics
    return (np.nanmedian(mean_length_list, axis=0), 
            np.nanvar(mean_length_list, axis=0), 
            np.nanmean(mean_trophic_list, axis=0), 
            np.nanmean(weighted_trophic_list, axis = 0),
            np.nanmean(max_trophic_list, axis = 0), 
            np.nanmean(var_trophic_list, axis = 0), 
            np.nanmedian(max_stabil_list, axis = 0), 
            np.nanvar(max_stabil_list, axis=0),
            proportion_formed,
            np.nanmedian(biomass_resource_list, axis = 0),
            np.nanmedian(total_biomass_list,axis = 0),
            np.nanmedian(sum_biomass_list, axis = 0),
            np.nanmedian(biomass_tl1_list, axis = 0),
            np.nanmedian(biomass_tl12_list,axis = 0),
            np.nanmedian(biomass_tl23_list,axis = 0),
            np.nanmedian(biomass_tl34_list,axis = 0),
            np.nanmedian(biomass_tl4_list,axis = 0),
            np.nanmedian(bioperbody_sub25_list, axis =0),
            np.nanmedian(bioperbody_2550_list, axis = 0),
            np.nanmedian(bioperbody_5075_list, axis = 0),
            np.nanmedian(bioperbody_up75_list, axis = 0),
            np.nanmedian(linksperbody_sub25_list, axis =0),
            np.nanmedian(linksperbody_2550_list, axis = 0),
            np.nanmedian(linksperbody_5075_list, axis = 0),
            np.nanmedian(linksperbody_up75_list, axis = 0),
            np.nanmedian(sum_link_list, axis = 0))


# Initialize parameters
#Change parameter values from 1/2 to 2 times the reference value
beta_00 = np.sort(np.geomspace(beta[0, 0] / 2, 2 * beta[0, 0], 10))
beta_01 = np.sort(np.geomspace(beta[0, 1] / 2, 2 * beta[0, 1], 10))
beta_10 = np.sort(np.geomspace(beta[1, 0] / 2, 2 * beta[1, 0], 10))
beta_11 = np.sort(np.geomspace(beta[1, 1] / 2 ,2 * beta[1, 1], 10))
beta_20 = np.sort(np.geomspace(beta[2, 0] / 2, 2 * beta[2, 0], 10))
beta_21 = np.sort(np.geomspace(beta[2, 1] / 2, 2 * beta[2, 1], 10))
beta_30 = np.sort(np.geomspace(beta[3, 0] / 2, 2 * beta[3, 0], 10))
beta_31 = np.sort(np.geomspace(beta[3, 1] / 2, 2 * beta[3, 1], 10))
beta_40 = np.sort(np.geomspace(beta[4, 0] / 2, 2 * beta[4, 0], 10))
beta_50 = np.sort(np.geomspace(beta[5, 0] / 2, 2 * beta[5, 0], 10))
beta_60 = np.sort(np.geomspace(beta[6, 0] / 2, 2 * beta[6, 0], 10))
beta_61 = np.sort(np.geomspace(beta[6, 1] / 2, 2 * beta[6, 1], 10))
beta_70 = np.sort(np.geomspace(beta[7, 0] / 2, 2 * beta[7, 0], 10))
beta_80 = np.sort(np.geomspace(beta[8, 0] / 2, 2 * beta[8, 0], 10))
beta_81 = np.sort(np.geomspace(beta[8, 1] / 2, 2 * beta[8, 1], 10))


#beta_40 = np.sort(np.linspace(0.5,4, 20))

#Change parameter values up to aquatic-terrestrial reference value.
"""beta_00 = np.sort(np.linspace(beta[0, 0] - 2.59, beta[0, 0] + 2.59, 10))
beta_01 = np.sort(np.linspace(beta[0, 1] - 0.13667, beta[0, 1] + 0.13667, 10))
beta_10 = np.sort(np.linspace(beta[1, 0] - 0.9865, beta[1, 0] + 0.9865, 10))
beta_11 = np.sort(np.linspace(beta[1, 1] - 0.0001, beta[1, 1] + 0.0001, 10))
beta_20 = np.sort(np.linspace(beta[2, 0] - 1.0883, beta[2, 0] + 1.0883, 10))
beta_21 = np.sort(np.linspace(beta[2, 1] - 0.03114, beta[2, 1] + 0.03114, 10))
beta_30 = np.sort(np.linspace(beta[3, 0] - 0, beta[3, 0] + 0, 10))
beta_31 = np.sort(np.linspace(beta[3, 1] - 0, beta[3, 1] + 0, 10))
beta_50 = np.sort(np.linspace(beta[5, 0] - 7.47, beta[5, 0] + 7.47, 10))
beta_60 = np.sort(np.linspace(beta[6, 0] - 0.49, beta[6, 0] + 0.49, 10))
beta_61 = np.sort(np.linspace(beta[6, 1] - 0.0495, beta[6,1] + 0.0495, 10))
"""
#Change parameter values up to +- 0.5 of reference value.
"""beta_00 = np.sort(np.linspace(beta[0, 0] - 0.5, beta[0, 0] + 0.5, 10))
beta_01 = np.sort(np.linspace(beta[0, 1] - 0.5, beta[0, 1] + 0.5, 10))
beta_10 = np.sort(np.linspace(beta[1, 0] - 0.5, beta[1, 0] + 0.5, 10))
beta_11 = np.sort(np.linspace(beta[1, 1] - 0.5, beta[1, 1] + 0.5, 10))
beta_20 = np.sort(np.linspace(beta[2, 0] - 0.5, beta[2, 0] + 0.5, 10))
beta_21 = np.sort(np.linspace(beta[2, 1] - 0.5, beta[2, 1] + 0.5, 10))
beta_30 = np.sort(np.linspace(beta[3, 0] - 0.5, beta[3, 0] + 0.5, 10))
beta_31 = np.sort(np.linspace(beta[3, 1] - 0.5, beta[3, 1] + 0.5, 10))
beta_40 = np.sort(np.linspace(0, 0.5, 20))
beta_50 = np.sort(np.linspace(beta[5, 0] - 0.5, beta[5, 0] + 0.5, 10))
beta_60 = np.sort(np.linspace(beta[6, 0] - 0.5, beta[6, 0] + 0.5, 10))
beta_61 = np.sort(np.linspace(beta[6, 1] - 0.5, beta[6, 1] + 0.5, 10))"""


# Run simulations
(diversity, var_diversity, mean_trophic_level, weighted_trophic_level, max_trophic_level, 
var_trophic_level, stability, var_stability, proportion_formed, basal_biomass, 
nonbasal_biomass, total_biomass, tl1_biomass, tl12_biomass, tl23_biomass, tl34_biomass, 
tl4_biomass, bodymass_sub25, bodymass_2550, bodymass_5075, bodymass_up75, 
link_per_mass_sub25, link_per_mass_2550, link_per_mass_5075, link_per_mass_up75, 
sum_links) = simulate_foodweb(beta_30, beta_index=(3, 0))

#resultes = simulate_foodweb(beta_40, beta_index=(0, 0))
 


#reset beta values for next simulation run
beta[0,0] = -3.1503  
beta[0, 1] = 0.234519  
beta[1, 0] = 1.6645 
beta[1, 1] = 0.09433
beta[2, 0] = -1.1012   
beta[2, 1] = -0.066577
beta[3, 0] = -0.037
beta[3, 1] = -0.04
beta[4,0] = 1
beta[5, 0] = -11.83
beta[6, 0] = 0.54
beta[6, 1] = 0.05 
   
# Plot results
fig, ax1 = plt.subplots()
ax1.plot(fm.normalize_x(beta_60), diversity, label="$\\alpha_{f}$")
ax1.set_xlabel("Change from mean")
ax1.set_ylabel("links")
ax1.legend()
ax1.grid(True)
#ax1.set_title("change $\\pm$ 65%")
#fig.savefig("Cintra_stabil")
#plt.tight_layout()
#plt.show()        

#Save in an excel file
"""datapm65 = pd.DataFrame({
                         "change":fm.normalize_x(beta_00), "diversity":diversity, 
                         "var_diversity":var_diversity, "mean_tl": mean_trophic_level,"weighted_tl": weighted_trophic_level,
                         "max_tl":max_trophic_level, "var_tl":var_trophic_level,"stability":stability, "var_stability":var_stability, 
                         "proportion":proportion_formed, "basal_mass":basal_biomass, "nonbasal_biomass":nonbasal_biomass,
                         "total_biomass":total_biomass,  "tl1_biomass":tl1_biomass, "tl12_biomass": tl12_biomass, 
                         "tl23_biomass": tl23_biomass, "tl34_biomass": tl34_biomass, "tl4_biomass": tl4_biomass, 
                         "bodymass_sub25": bodymass_sub25, "bodymass_2550": bodymass_2550, "bodymass_5075": bodymass_5075, 
                         "bodymass_up75": bodymass_up75, "link_mass_sub25": link_per_mass_sub25, "link_mass_2550": link_per_mass_2550, 
                         "link_mass_5075": link_per_mass_5075, "link_mass_up75": link_per_mass_up75, "sum_links": sum_links                 
                         })
datapm65 = pd.DataFrame(results)
    
excel_path = "betaai_LV.xlsx"
datapm65.to_excel(excel_path, index=False)"""

##########################################     
