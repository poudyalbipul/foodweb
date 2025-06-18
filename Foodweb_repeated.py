#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 14 14:29:00 2025

@author: bipul
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import solve_ivp
import Foodweb_model as fm
import networkx as nx
from scipy.stats import entropy


beta = fm.beta

def simulate_communities(n_times, n_spec, beta = beta ):
    population_list = []
    nonbasal_population_list = []
    basal_population_list = []
    biomass_list = []
    #basal_biomass_list = []
    #nonbasal_biomass_list = []
    mean_bodymass_list = []
    diversity_list = []
    stability_list = []
    mtl_list = []   
    links_list = []
    cluster_list = []
    path_list = []
    modularity_score_list = []
    modularity_count_list = []    
    degree_list = []
    #trophic_coherence_list = []
    frequency_weak_list = []
    gini_coeff_list = []
    three_cycle_list = []
    cycle_2_list = []
    cycle_4_list = []
    for i in range(n_times):
        
        species_id = fm.generate_species(n_spec, e =0.85, random=True, B0=1e-7, K = 1e7, beta= beta )
        species_id = fm.change_temperature(species_id, beta=beta)
        mu, A = fm.compute_LV_param(species_id, beta=beta)            
        N_0 = np.full(n_spec, 1e2)
        N_0[0] = species_id["K"]
        #N_0_log = np.log(N_0)
        t_eval = np.linspace(0, 100, 101)
        sol = solve_ivp(fm.LV_model, [0, np.inf], 
                        t_eval= t_eval, 
                        y0=N_0, method="LSODA", args=(mu, A))  
        
        # compute total population and species richness
        survivors = sol.y[:, -1] > 1                   
        
        m_i = np.array(species_id["m_i"])
        survivors_nonbasal = sol.y[1:,-1] > 1
        #biomass = sum(num for num in sol.y[1:,-1] if num > 1)     
       # bodysize = m_i[1:][survivors_nonbasal] * sol.y[1:,-1][survivors_nonbasal]
 
        
        if np.any(survivors_nonbasal):
            # At least one non-basal species survived
            surviving_pops = sol.y[1:, -1][survivors_nonbasal]
            surviving_masses = m_i[1:][survivors_nonbasal]
            diversity = np.sum(survivors)
            
            nonbasal_population = np.nansum(surviving_pops) 
            basal_population = sol.y[0,-1]
            
            nonbasal_biomass = np.nansum(surviving_masses * surviving_pops)
            basal_biomass = np.nansum(sol.y[0,-1] * species_id["m_i"][0])
            mean_biomass = np.nansum(nonbasal_biomass) / nonbasal_population
            
            # Safe assertion - only compare among survivors
            #max_surviving_mass = np.max(surviving_masses)
            
            #biomass_gain = np.sum(sol.y[:,-1][survivors] * m_i[survivors]) 
           # assert mean_bodysize <= max_surviving_mass
        else:
            # No non-basal species survived - food web collapsed
            nonbasal_biomass = np.nan
            basal_biomass = np.nan
            mean_biomass = np.nan     
            nonbasal_population = np.nan   
            basal_population = np.nan 
            diversity = np.nan 
            #max_surviving_mass = np.nan    
        
        #mean_bodysize = np.nansum(bodysize) / biomass
        #assert mean_bodysize <= np.max(m_i[1:][survivors_nonbasal]), "Mean bodysize unexpectedly exceeds max(m_i)!"
    
        
        #compute stability
        ind_s = np.where(sol.y[:,-1] > 1)[0]
        A_surv = A[np.ix_(ind_s, ind_s)]
        J = -np.diag(sol.y[:,-1][ind_s])@A_surv
        
        J_norm = J
        eigenvalues = np.linalg.eigvals(J_norm).real
        stability = np.nanmax(eigenvalues)
        
        # Get off-diagonal interaction strengths
        mask = ~np.eye(J.shape[0], dtype=bool)
        nonzero_strengths = np.abs(J[mask])[np.abs(J[mask]) > 0]
        # Check if any interactions exist
        if len(nonzero_strengths) > 0:
            weak_threshold = 0.01 * np.max(nonzero_strengths)
            frequency_weak = np.sum(nonzero_strengths <= weak_threshold) / len(nonzero_strengths)
        else:
            frequency_weak = np.nan  # or 0, depending on what you prefer
        
        # Add this right after the frequency_weak calculation
        if len(nonzero_strengths) > 0:
            # Gini coefficient calculation
            sorted_strengths = np.sort(nonzero_strengths)
            n = len(sorted_strengths)
            cumsum = np.cumsum(sorted_strengths)
            gini_coefficient = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
        else:
            gini_coefficient = np.nan
        
        
        #compute mean trophic level
        
       
    
                 
        
        
        #compute resource population
        #tlresource_mask = (fm.compute_trophic_level(species_id, survivors) ==  0)       
        #tlresource_biomass = np.sum(sol.y[:,-1][survivors][tlresource_mask])         
        #Non_basal_biomass = biomass - tlresource_biomass
        
        #compute number of links and link density
        adjacency_matrix = fm.compute_links(species_id).T[survivors][:,survivors]
        s = fm.compute_trophic_level(species_id, survivors)
        mean_trophic_level = np.nanmean(s)
        


        #q_ij = []
        #L = 0
        #for i in range(len(adjacency_matrix)):
        #    for j in range(len(adjacency_matrix)):
        #        if adjacency_matrix[i, j] == 1:
        #            # i eats j, so s[i] should be higher than s[j]
        #            diff = s[i] - s[j]  # predator - prey
        #            q_ij.append((diff - 1) ** 2)
        #            L += 1

        #trophic_coherence = np.sqrt(np.sum(q_ij) / L) if L > 0 else np.nan
        
        
        
        degrees = np.nansum(adjacency_matrix, axis=1)
        mean_deg = np.nanmean(degrees)
        heterogeneity = np.nanstd(degrees) / mean_deg if mean_deg > 0 else 0
        total_links = np.sum(adjacency_matrix) / (diversity) 
        
        # Clustering coefficient
        G = nx.DiGraph(adjacency_matrix)
        
        all_cycles = list(nx.simple_cycles(G))
        #total_cycles = len(all_cycles)
    
        positive_cycles = {2: 0, 3: 0, 4: 0}
        negative_cycles = {2: 0, 3: 0, 4: 0}

        for cycle in all_cycles:
            if len(cycle) in [2, 3, 4]:
                sign_product = 1
                for i in range(len(cycle)):
                    src = cycle[i]
                    tgt = cycle[(i + 1) % len(cycle)]
                    interaction_sign = np.sign(J[src, tgt])
                    sign_product *= interaction_sign

                if sign_product > 0:
                    positive_cycles[len(cycle)] += 1
                elif sign_product < 0:
                    negative_cycles[len(cycle)] += 1   
        
        #clustering = nx.average_clustering(G)
        # Path length metrics
        path_lengths = []
        for node1 in G.nodes():
            for node2 in G.nodes():
                if node1 != node2:
                    try:
                        path_lengths.append(nx.shortest_path_length(G, node1, node2))
                    except nx.NetworkXNoPath:
                        pass
        
        avg_path_length = np.nanmean(path_lengths) if path_lengths else np.nan
        
        # Degree distribution
        #in_degrees = [d for n, d in G.in_degree()]
        #out_degrees = [d for n, d in G.out_degree()]
    
        # Entropy of degree distribution (higher entropy = more complex)
        #in_degree_entropy = entropy(in_degrees) if sum(in_degrees) > 0 else 0
        #out_degree_entropy = entropy(out_degrees) if sum(out_degrees) > 0 else 0
        
        G_undirected = G.to_undirected()
        clustering = nx.average_clustering(G_undirected)
        # Detect communities using greedy modularity maximization
        if G_undirected.number_of_edges() > 0:
            communities = list(nx.community.greedy_modularity_communities(G_undirected, weight="weight"))
            modularity_score = nx.community.modularity(G_undirected, communities, weight="weight")
            modularity_count = len(communities)
        else:
            modularity_score = np.nan
            modularity_count = 0

        population_list.append(nonbasal_population + basal_population)
        nonbasal_population_list.append(nonbasal_population)
        basal_population_list.append(basal_population)
        biomass_list.append(nonbasal_biomass + basal_biomass)
        #basal_biomass_list.append(basal_biomass)
        #nonbasal_biomass_list.append(nonbasal_biomass)
        mean_bodymass_list.append(mean_biomass)
        diversity_list.append(diversity)
        stability_list.append(stability)
        mtl_list.append(mean_trophic_level)
        links_list.append(total_links)
        cluster_list.append(clustering)
        path_list.append(avg_path_length)
        modularity_score_list.append(modularity_score)
        modularity_count_list.append(modularity_count)
        degree_list.append(heterogeneity)
        #trophic_coherence_list.append(trophic_coherence)
        frequency_weak_list.append(frequency_weak)
        gini_coeff_list.append(gini_coefficient)
        three_cycle_list.append(negative_cycles[3]-positive_cycles[3])
        cycle_4_list.append(negative_cycles[4]- positive_cycles[4])
        cycle_2_list.append(negative_cycles[2] - positive_cycles[2])
        
    return (diversity_list, population_list, nonbasal_population_list, basal_population_list,
            biomass_list, mean_bodymass_list, stability_list, mtl_list, links_list, cluster_list, path_list,
            modularity_count_list, modularity_score_list, degree_list, frequency_weak_list, gini_coeff_list, three_cycle_list,
            cycle_2_list, cycle_4_list
            )

(Diversity, Total_population, Nonbasal_population, Basal_population, Total_biomass, 
 Mean_bodymass, Stability, Mean_trophic_level, Links, Cluster_coeff,
Mean_path_length, Modularity_count, Modularity_score, Degree_distribution, Weak_frequency, Gini_coefficient, Three_cycles, 
Two_cycles, Four_cycles) = simulate_communities(1000, 150)



#simulation_data = pd.DataFrame({"Diversity": Diversity, "Total_population":Total_population,
#                                "Nonbasal_population": Nonbasal_population,
#                                "Basal_population": Basal_population, "Total_Biomass": Total_biomass,
#                                "Mean_bodymass": Mean_bodymass, "Stability":Stability,                                                                
#                               "Trophic_level": Mean_trophic_level, 
#                               "Total_links": Links, "Cluster_coefficient": Cluster_coeff, 
#                               "Path_length": Mean_path_length, "Modularity_count": Modularity_count, 
#                               "Modularity_score": Modularity_score, "Degree_distr": Degree_distribution
#                               })

#excel_path = "community_parameters9.xlsx"
#simulation_data.to_excel(excel_path, index=False)