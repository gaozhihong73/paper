library(minpack.lm)
library(abess)
library(parallel)
library(Matrix)

# ==============================================================================
# Module 1: Unified Feature Store Preparation
# Handles flattening of dynamic omics and separating Fixed vs. Candidate features
# ==============================================================================

flatten_dynamic_features <- function(dynamic_list, time_labels) {
  if (is.null(dynamic_list) || length(dynamic_list) == 0) return(NULL)
  flattened_mat <- NULL
  for (name in names(dynamic_list)) {
    mat <- dynamic_list[[name]]
    if(is.matrix(mat)) {
      colnames(mat) <- paste0(name, "_T", time_labels)
      if (is.null(flattened_mat)) flattened_mat <- mat else flattened_mat <- cbind(flattened_mat, mat)
    }
  }
  return(flattened_mat)
}

prepare_feature_store <- function(
    Genotypes = NULL,         # Candidate: SNPs
    Static_Omics = NULL,      # Candidate: Static Omics
    Dynamic_Omics = NULL,     # Candidate: Dynamic Omics (List)
    Population_Struct = NULL, # Fixed: Population Structure PCs
    Env_Covariates = NULL,    # Fixed: Environmental Factors
    time_points
) {
  
  # 1. Process Fixed Effects (Must be in EVERY sub-model)
  Fixed_Mat <- NULL
  if(!is.null(Population_Struct)) Fixed_Mat <- cbind(Fixed_Mat, Population_Struct)
  if(!is.null(Env_Covariates))    Fixed_Mat <- cbind(Fixed_Mat, Env_Covariates)
  
  # 2. Process Candidate Effects (Subject to Random Sampling)
  Dyn_Flat <- flatten_dynamic_features(Dynamic_Omics, time_points)
  
  Candidate_Mat <- NULL
  if(!is.null(Genotypes))    Candidate_Mat <- cbind(Candidate_Mat, Genotypes)
  if(!is.null(Static_Omics)) Candidate_Mat <- cbind(Candidate_Mat, Static_Omics)
  if(!is.null(Dyn_Flat))     Candidate_Mat <- cbind(Candidate_Mat, Dyn_Flat)
  
  n_fixed <- if(is.null(Fixed_Mat)) 0 else ncol(Fixed_Mat)
  n_cand  <- if(is.null(Candidate_Mat)) 0 else ncol(Candidate_Mat)
  
  if (n_fixed + n_cand == 0) stop("Error: No input features provided!")
  
  return(list(
    Fixed_Mat = Fixed_Mat,
    Candidate_Mat = Candidate_Mat,
    n_fixed = n_fixed,
    n_cand = n_cand,
    feature_names_fixed = if(n_fixed > 0) colnames(Fixed_Mat) else NULL,
    feature_names_cand = if(n_cand > 0) colnames(Candidate_Mat) else NULL
  ))
}


# ==============================================================================
# Module 2: RDE Sub-Model Worker
# Trains a single "Weak Predictor" on a subset of features
# ==============================================================================

train_rde_sub_model_worker <- function(
    sub_model_id, 
    subset_indices,       # Indices of candidate features for this model
    Feature_Store,        # The full data object
    Y_matrix, time_points, 
    mean_func, start_params,
    lop_order, max_iter, support_size
) {
  
  # 1. Construct the Sub-Dataset for this embedding
  #    Always include Fixed_Mat + Randomly selected columns from Candidate_Mat
  Fixed_Part <- Feature_Store$Fixed_Mat
  Random_Part <- Feature_Store$Candidate_Mat[, subset_indices, drop = FALSE]
  
  X_sub <- cbind(Fixed_Part, Random_Part)
  
  # 2. Define Penalty Factor for Abess
  #    0 = Mandatory (Fixed Features), 1 = Selectable (Random Features)
  n_fixed <- Feature_Store$n_fixed
  n_random <- length(subset_indices)
  penalty_vec <- c(rep(0, n_fixed), rep(1, n_random))
  
  # 3. Build Dynamic Design Matrix (Basis Expansion)
  #    We need a mini-version of build_expanded_matrix here
  #    (Inline logic for speed or call helper)
  
  # --- Inline Basis Expansion for Sub-Model ---
  Basis_mat <- get_LOP_basis(time_points, lop_order)
  n_basis <- ncol(Basis_mat)
  N <- nrow(Y_matrix); T_len <- length(time_points)
  
  total_cols <- ncol(X_sub) * n_basis
  X_tilde <- matrix(0, nrow = N * T_len, ncol = total_cols)
  groups <- integer(total_cols)
  sub_penalty <- numeric(total_cols)
  
  Basis_long <- do.call(rbind, replicate(N, Basis_mat, simplify=FALSE))
  
  col_ptr <- 1
  for(j in 1:ncol(X_sub)) {
    xj_long <- rep(X_sub[, j], each = T_len)
    block_val <- xj_long * Basis_long
    
    idx_range <- col_ptr : (col_ptr + n_basis - 1)
    X_tilde[, idx_range] <- block_val
    groups[idx_range] <- j
    sub_penalty[idx_range] <- penalty_vec[j]
    
    col_ptr <- col_ptr + n_basis
  }
  # --------------------------------------------
  
  # 4. Train the Model (Iterative Solver)
  #    We use a simplified version of solve_iterative_system
  model_result <- tryCatch({
    solve_iterative_system_internal(
      Y_matrix = Y_matrix,
      X_tilde = X_tilde, 
      groups = groups, 
      penalty = sub_penalty,
      time_points = time_points,
      mean_func = mean_func,
      start_params = start_params,
      max_iter = max_iter,
      support_size = support_size
    )
  }, error = function(e) return(NULL))
  
  # 5. Calculate Training Error (RSS) for RDE weighting
  rss <- NA
  if(!is.null(model_result)) {
    # Quick prediction on training set
    # Note: solve_iterative_system_internal should return fitted values or we calc here
    # Assuming model_result contains 'beta' and 'intercept'
    # G_hat = X_tilde %*% beta + intercept
    # Y_hat = Mean_Curve + G_hat
    
    # Re-calculate Y_hat for error
    t_vec_all <- rep(time_points, N)
    mu_curve <- mean_func(model_result$theta, t_vec_all)
    
    g_hat <- as.numeric(X_tilde %*% model_result$beta + model_result$intercept)
    y_hat <- mu_curve + g_hat
    
    rss <- sum((as.vector(t(Y_matrix)) - y_hat)^2)
  }
  
  return(list(
    model = model_result, 
    candidate_idx = subset_indices, # Keep track of which features were used
    training_error = rss,
    fixed_included = TRUE
  ))
}

# Helper: Simplified Solver for internal use
solve_iterative_system_internal <- function(Y_matrix, X_tilde, groups, penalty, time_points, mean_func, start_params, max_iter, support_size) {
  
  N <- nrow(Y_matrix); Y_vec <- as.vector(t(Y_matrix)); t_vec_all <- rep(time_points, N)
  
  # Init ODE
  resid_fun_init <- function(p) Y_vec - mean_func(p, t_vec_all)
  init_fit <- nls.lm(par=start_params, fn=resid_fun_init)
  current_theta <- coef(init_fit)
  
  # Loop
  for(i in 1:max_iter) {
    # 1. Abess
    mu <- mean_func(current_theta, time_points); Y_res <- Y_vec - rep(mu, N)
    fit <- abess(x=X_tilde, y=Y_res, group.index=groups, penalty.factor=penalty, 
                 support.size=support_size, tune.type="gic", family="gaussian")
    
    k <- fit$best.size; cf <- coef(fit, support.size=k)
    int <- cf[1,1]; beta <- cf[-1,1]
    g_hat <- as.numeric(X_tilde %*% beta + int)
    
    # 2. ODE
    Y_clean <- Y_vec - g_hat
    fit_ode <- tryCatch(nls.lm(par=current_theta, fn=function(p) Y_clean - mean_func(p, t_vec_all)), error=function(e) NULL)
    if(is.null(fit_ode)) break
    current_theta <- coef(fit_ode)
  }
  
  return(list(theta=current_theta, beta=beta, intercept=int))
}


# ==============================================================================
# Module 3: Universal RDE Ensemble Trainer
# ==============================================================================

train_universal_RDE <- function(
    Y_matrix, 
    time_points, 
    mean_func, start_params,
    
    # --- Universal Data Inputs ---
    Genotypes = NULL,          
    Static_Omics = NULL,      
    Dynamic_Omics = NULL,     
    Population_Struct = NULL,  
    Env_Covariates = NULL,     
    
    # --- RDE Hyperparameters ---
    n_embeddings = 50,         
    subset_size = 200,         
    n_cores = 1,
    
    # --- Model Hyperparameters ---
    lop_order = 3,
    max_iter = 5,              # Weak learners need fewer iters
    support_size = NULL        # Let abess decide via GIC
) {
  
  # 1. Prepare Feature Store
  cat(">>> [RDE Setup] Preparing Universal Feature Store...\n")
  Feat_Store <- prepare_feature_store(
    Genotypes, Static_Omics, Dynamic_Omics, Population_Struct, Env_Covariates, time_points
  )
  
  n_cand <- Feat_Store$n_cand
  if(n_cand == 0) stop("No candidate features to sample from!")
  
  cat(sprintf("    Fixed Features: %d | Candidate Features: %d\n", Feat_Store$n_fixed, n_cand))
  
  # 2. Generate Random Embeddings (Indices for Candidates only)
  #    "Randomly pick s tuples from candidate set"
  actual_subset_size <- min(subset_size, n_cand)
  embedding_indices <- replicate(n_embeddings, 
                                 sample(1:n_cand, actual_subset_size, replace = FALSE), 
                                 simplify = FALSE)
  
  # 3. Define Worker Wrapper
  worker_wrapper <- function(k) {
    train_rde_sub_model_worker(
      sub_model_id = k,
      subset_indices = embedding_indices[[k]],
      Feature_Store = Feat_Store,
      Y_matrix = Y_matrix,
      time_points = time_points,
      mean_func = mean_func,
      start_params = start_params,
      lop_order = lop_order,
      max_iter = max_iter,
      support_size = support_size
    )
  }
  
  # 4. Parallel Execution
  cat(sprintf(">>> [RDE Training] Launching %d embeddings on %d cores...\n", n_embeddings, n_cores))
  
  if(n_cores > 1) {
    cl <- makeCluster(n_cores)
    # Export necessary functions and libraries
    clusterEvalQ(cl, {library(minpack.lm); library(abess)})
    clusterExport(cl, varlist = c("train_rde_sub_model_worker", "solve_iterative_system_internal", 
                                  "get_LOP_basis", "mean_func", "time_points"), envir = environment())
    
    results <- parLapply(cl, 1:n_embeddings, worker_wrapper)
    stopCluster(cl)
  } else {
    results <- lapply(1:n_embeddings, worker_wrapper)
  }
  
  # 5. Filter Invalid Models
  valid_results <- results[!sapply(results, function(x) is.null(x$model))]
  cat(sprintf(">>> RDE Complete. Valid Models: %d / %d\n", length(valid_results), n_embeddings))
  
  return(list(
    sub_models = valid_results,
    Feature_Store_Meta = list( # Save metadata only, not full data, to save RAM
      n_fixed = Feat_Store$n_fixed,
      n_cand = Feat_Store$n_cand,
      names_fixed = Feat_Store$feature_names_fixed,
      names_cand = Feat_Store$feature_names_cand
    ),
    time_points = time_points,
    mean_func = mean_func,
    lop_order = lop_order
  ))
}


# ==============================================================================
# Module 4: Universal RDE Predictor
# ==============================================================================

predict_universal_RDE <- function(
    rde_model, 
    
    # --- New Data Inputs ---
    Genotypes = NULL, Static_Omics = NULL, Dynamic_Omics = NULL, 
    Population_Struct = NULL, Env_Covariates = NULL,
    
    sigma_scale = 0.2
) {
  
  # 1. Re-create Feature Store for New Data
  #    MUST follow the exact same structure as training
  Feat_Store_New <- prepare_feature_store(
    Genotypes, Static_Omics, Dynamic_Omics, Population_Struct, Env_Covariates, 
    rde_model$time_points
  )
  
  # Validate dimensions
  if(Feat_Store_New$n_fixed != rde_model$Feature_Store_Meta$n_fixed) warning("Fixed feature count mismatch!")
  
  # 2. Calculate Weights (Same as before)
  sub_models <- rde_model$sub_models
  train_errors <- sapply(sub_models, function(x) x$training_error)
  min_err <- min(train_errors, na.rm=TRUE)
  effective_sigma <- quantile(train_errors - min_err, 0.2, na.rm=TRUE) * sigma_scale
  if(effective_sigma == 0) effective_sigma <- 1e-5
  weights <- exp(-(train_errors - min_err)/effective_sigma)
  weights <- weights / sum(weights)
  
  # 3. Generate Predictions
  N_new <- nrow(Feat_Store_New$Fixed_Mat)
  T_len <- length(rde_model$time_points)
  Y_final <- matrix(0, nrow = N_new, ncol = T_len)
  
  cat(">>> Aggregating predictions...\n")
  
  for(k in 1:length(sub_models)) {
    m <- sub_models[[k]]
    if(weights[k] < 1e-4) next # Skip negligible models
    
    # Reconstruct X_sub for this model
    Fixed_Part <- Feat_Store_New$Fixed_Mat
    Random_Part <- Feat_Store_New$Candidate_Mat[, m$candidate_idx, drop=FALSE]
    X_sub <- cbind(Fixed_Part, Random_Part)
    
    # Expand Basis (Inline)
    # ... (Same basis expansion logic as in worker) ...
    # For brevity, assuming we have a helper 'expand_basis_for_pred(X_sub, ...)'
    # Here is the simplified logic:
    
    Basis_mat <- get_LOP_basis(rde_model$time_points, rde_model$lop_order)
    n_basis <- ncol(Basis_mat)
    X_tilde_pred <- matrix(0, nrow=N_new*T_len, ncol=ncol(X_sub)*n_basis)
    
    col_ptr <- 1
    Basis_long <- do.call(rbind, replicate(N_new, Basis_mat, simplify=FALSE))
    for(j in 1:ncol(X_sub)) {
      block <- rep(X_sub[,j], each=T_len) * Basis_long
      X_tilde_pred[, col_ptr:(col_ptr+n_basis-1)] <- block
      col_ptr <- col_ptr + n_basis
    }
    
    # Predict: Y = Mean_Curve + X_beta + Int
    t_vec <- rep(rde_model$time_points, N_new)
    mu <- rde_model$mean_func(m$model$theta, t_vec)
    g_hat <- as.numeric(X_tilde_pred %*% m$model$beta + m$model$intercept)
    
    y_pred_vec <- mu + g_hat
    
    # Add to weighted sum (reshape to matrix first)
    y_pred_mat <- matrix(y_pred_vec, nrow=N_new, byrow=TRUE)
    Y_final <- Y_final + y_pred_mat * weights[k]
  }
  
  colnames(Y_final) <- paste0("T_", rde_model$time_points)
  return(list(Y_pred = Y_final, weights = weights))
}

# 1. Train
uni_model <- train_universal_RDE(
  Y_matrix = Y_train,
  time_points = Time,
  mean_func = Gompertz_curve, # For tumor/microbiome
  start_params = c(100, 5, 0.1),
  
  # --- Flexible Inputs ---
  Genotypes = SNP_Matrix,          # Candidate (Randomly sampled)
  Population_Struct = PC_Matrix,   # Fixed (Mandatory correction)
  Env_Covariates = Env_Data,       # Fixed (Baseline environment)
  Dynamic_Omics = list(Mic=M_dyn), # Candidate (Dynamic features)
  
  n_embeddings = 50
)

# 2. Predict
preds <- predict_universal_RDE(
  uni_model,
  Genotypes = SNP_New,
  Population_Struct = PC_New,
  Env_Covariates = Env_New,
  Dynamic_Omics = list(Mic=M_New)
)