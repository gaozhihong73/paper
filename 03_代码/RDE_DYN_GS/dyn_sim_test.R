
#A physics-informed dynamic variable selection framework for accurate phenotypic prediction in small-sample longitudinal cohorts
#https://www.cancerrxgene.org/downloads/bulk_download
source("dyn_sim.R")
source("dyn_curve.R")
source("dyn_screen_info.R")
source("dyn_gs.R")
source("dyn_rde_gs.R")
source("dyn_plot.R")
source("dyn_util.R")

# DynaGS data export example (run this block separately before model training):
# sim_data <- simulate_dynamic_LOP_Logistic_Types(
#   N = 400, P = 1000, nQTL = 10, h2_target = 0.5,
#   prop_qtl_type = c(0.6, 0.2, 0.2), LD_r = 0.1, seed = 523
# )
# export_dynags_simulation(
#   sim_data, output_dir = "D:/Data/研究生/03_学位论文相关/毕业论文/03_代码/DynaGS/dataset/simulation",
#   scenario = "baseline", rep = 1, split_seed = 100523
# )


#validation simulation
set.seed(523) 
sim_mixed <- simulate_dynamic_LOP_Logistic_Types(N = 400, P=1000,nQTL = 10, prop_qtl_type = c(0.6, 0.2, 0.2),h2_target = 0.8,LD_r=0.1)
result_sis <- DC.sis(dat=list(pheno_y = sim_mixed$Y,snps = sim_mixed$G))
evaluate_SIS_performance(sis_result=result_sis, sim_data=sim_mixed, mode = "Auto")


result_sis_AD <- DC.sis.AD(dat=list(pheno_y = sim_mixed$Y,snps = sim_mixed$G))
evaluate_SIS_performance(sis_result=result_sis_AD, sim_data=sim_mixed, mode = "Auto")


results <- solve_iterative_dynamic_system_AD(Y_matrix = sim_mixed$Y, Genotypes = sim_mixed$G, time_points = sim_mixed$Time,
  mean_func = Logistic_curve, start_params = c(30,45,0.5),basis_type = "LOP",lop_order = 3,max_iter = 10, tol = 1e-4,support_size = 0:30)
evaluate_dynamic_selection_results(results=results, true_qtl_index=sim_mixed$QTL_index)

cv_results_abess <- cv_dynamic_GS(Y_matrix = sim_mixed$Y,Genotypes = sim_mixed$G,time_points = sim_mixed$Time,mean_func = Logistic_curve, 
                            start_params = results$theta,lop_order = 3,effect_types = c("Add", "Dom"),n_folds = 5,max_iter = 5,
                            support_size = 0:(results$best_size+25))

if(is.null(rownames(cv_results_abess$cv_predictions))) rownames(cv_results_abess$cv_predictions) <- rownames(sim_mixed$Y)
plot_dynamic_GS(Y_obs=sim_mixed$Y,Y_pred=cv_results_abess$cv_predictions, time_points=sim_mixed$Time,mean_func=Logistic_curve,
                start_params=results$theta, sample_ids = rownames(sim_mixed$Y), filename = "Dynamic_GS_Report.pdf", width = 8, height = 6)

cv_rde_result <- run_rde_validation(Y_matrix = sim_mixed$Y, Genotypes = sim_mixed$G,time_points = sim_mixed$Time,mean_func = Logistic_curve,
                                    start_params = c(30, 45, 0.5),n_embeddings = 100,subset_size = 200,n_cores = 5)

