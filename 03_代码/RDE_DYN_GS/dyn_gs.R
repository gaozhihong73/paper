
solve_iterative_dynamic_system_AD <- function(
    Y_matrix, Genotypes, time_points, 
    mean_func = NULL, 
    start_params = NULL,
    basis_type = "LOP", lop_order = 3, 
    effect_types = c("Add", "Dom"),
    max_iter = 10, tol = 1e-4,
    support_size = 1:20 # 可选: 强制指定选多少个变量，NULL则自动CV选择
) {
  
  # --- 0. 参数检查 ---
  if (is.null(mean_func)) stop("[Error] 'mean_func' is missing!")
  if (is.null(start_params)) stop("[Error] 'start_params' is missing!")
  if (length(effect_types) == 0) stop("[Error] 'effect_types' cannot be empty!")
  
  N <- nrow(Y_matrix)
  T_len <- length(time_points)
  P <- ncol(Genotypes)
  
  Y_vec <- as.vector(t(Y_matrix)) 
  t_vec_all <- rep(time_points, N)
  
  # --- 1. 构建设计矩阵 (保持之前的逻辑，这步是通用的) ---
  cat(">>> [Pre-calc] Constructing Design Matrix for abess...\n")
  
  if(basis_type == "LOP") {
    Basis_mat <- get_LOP_basis(time_points, lop_order)
    n_basis <- ncol(Basis_mat)
  } else { stop("Only LOP supported.") }
  
  # 编码基因型
  G_encoded_list <- list()
  if ("Add" %in% effect_types) {
    G_encoded_list[["Add"]] <- scale(Genotypes - 1, center=TRUE, scale=FALSE)
  }
  if ("Dom" %in% effect_types) {
    G_dom <- matrix(0, N, P); G_dom[Genotypes == 1] <- 1
    G_encoded_list[["Dom"]] <- scale(G_dom, center=TRUE, scale=FALSE)
  }
  
  # 初始化大矩阵
  n_effects <- length(G_encoded_list)
  total_cols <- P * n_effects * n_basis
  X_tilde <- matrix(0, nrow = N * T_len, ncol = total_cols)
  groups <- integer(total_cols) # abess 需要 group.index
  
  Basis_long <- do.call(rbind, replicate(N, Basis_mat, simplify=FALSE))
  G_long_list <- lapply(G_encoded_list, function(G) G[rep(1:N, each=T_len), ])
  
  col_idx <- 1
  curr_grp_id <- 1
  
  # 填充矩阵
  for (j in 1:P) {
    for (eff_name in names(G_long_list)) {
      g_col <- G_long_list[[eff_name]][, j]
      block_val <- g_col * Basis_long 
      
      cols_range <- col_idx : (col_idx + n_basis - 1)
      X_tilde[, cols_range] <- block_val
      groups[cols_range] <- curr_grp_id
      
      col_idx <- col_idx + n_basis
      curr_grp_id <- curr_grp_id + 1
    }
  }
  cat(sprintf("    Matrix Ready: %d rows x %d cols. Groups: %d\n", nrow(X_tilde), ncol(X_tilde), max(groups)))
  
  # --- 2. 初始化 ---
  cat(">>> [Init] Fitting Initial Mean Curve...\n")
  resid_fun_init <- function(p) Y_vec - mean_func(p, t_vec_all)
  init_fit <- tryCatch(nls.lm(par=start_params, fn=resid_fun_init), error=function(e) stop("Init failed"))
  
  current_theta <- coef(init_fit)
  
  # 初始化 Beta (全零) 用于收敛检查
  current_beta <- rep(0, total_cols)
  
  # --- 3. 迭代循环 ---
  for(iter in 1:max_iter) {
    
    # === Step A: Abess (L0 Selection) ===
    # 1. 计算当前残差
    mu_curve <- mean_func(current_theta, time_points)
    Y_mean_removed <- Y_vec - rep(mu_curve, N)
    
    # 2. 运行 Abess
    # abess 速度极快，不需要 SIS。
    # tune.type = "cv" (交叉验证) 或 "gic" (信息准则，更快)
    # group.index 指定分组结构
    
    # 如果用户没指定 support_size，让 abess 自动选
    if(is.null(support_size)) {
      abess_fit <- abess(x = X_tilde, y = Y_mean_removed, 
                         group.index = groups, 
                         tune.type = "gic", # GIC 通常比 CV 快且准
                         family = "gaussian")
    } else {
      # 强制选 k 个变量
      abess_fit <- abess(x = X_tilde, y = Y_mean_removed, 
                         group.index = groups, 
                         support.size = support_size,
                         family = "gaussian")
    }
    
    # 3. 提取系数和预测
    # abess 自动返回最佳模型的系数 (包含 0)
    best_k <- abess_fit$best.size
    
    # 2. 提取最佳系数
    # 注意: coef() 返回的是稀疏矩阵，我们需要指定 support.size 并将其转为普通向量
    # drop=FALSE 防止单列矩阵降维带来的潜在问题，然后强转 numeric
    best_coef_matrix <- coef(abess_fit, support.size = best_k) 
    
    # 去掉第一行(Intercept)，转为向量
    new_beta <- as.numeric(best_coef_matrix[-1, 1]) 
    
    # 3. 提取最佳预测值
    # 同样必须指定 support.size，否则它可能返回整个路径的预测矩阵
    G_hat_matrix <- predict(abess_fit, newx = X_tilde, support.size = best_k)
    G_hat_vec <- as.numeric(G_hat_matrix)
    
    # 统计选中组数
    selected_grp_ids <- unique(groups[new_beta != 0])
    n_selected <- length(selected_grp_ids)
    
    # === Step B: 更新均值 (NLS) ===
    Y_clean_vec <- Y_vec - G_hat_vec
    
    resid_fun_iter <- function(p) Y_clean_vec - mean_func(p, t_vec_all)
    new_fit <- tryCatch(nls.lm(par=current_theta, fn=resid_fun_iter), error=function(e) NULL)
    
    if(is.null(new_fit)) break
    new_theta <- coef(new_fit)
    
    # === Step C: 双重收敛检查 (Dual Convergence Check) ===
    
    # 1. 均值参数差异 (Theta)
    diff_theta <- sum((new_theta - current_theta)^2)
    
    # 2. 遗传系数差异 (Beta) - 修正了您之前的担忧
    # 计算 L2 范数差异
    diff_beta <- sum((new_beta - current_beta)^2)
    
    # 打印状态
    param_str <- paste(names(new_theta), round(new_theta, 3), sep="=", collapse=" ")
    cat(sprintf("Iter %2d | dTheta: %.2e | dBeta: %.2e | Selected: %d | %s\n", 
                iter, diff_theta, diff_beta, n_selected, param_str))
    
    # 只有当两者都收敛才停止
    # 注意: Beta 维度很高，tol 可能需要适当放宽，或者看相对变化
    if(diff_theta < tol && diff_beta < tol) {
      cat(">>> Converged! (Mean and Genetic effects stabilized)\n")
      current_theta <- new_theta
      current_beta <- new_beta
      break
    }
    
    current_theta <- new_theta
    current_beta <- new_beta
  }
  
  # --- 4. 结果整理 ---
  cat(">>> Formatting Results...\n")
  
  # 映射回 SNP 信息
  beta_matrix <- matrix(new_beta, nrow=max(groups), byrow=TRUE) # Groups x Basis
  grp_power <- rowSums(beta_matrix^2)
  sel_idx <- which(grp_power > 0)
  
  # 重建组信息 (为了映射回 SNP Index 和 Type)
  # 注意: 这里的顺序必须和构建矩阵时的循环顺序完全一致
  info_map <- list()
  curr_id <- 1
  for(j in 1:P) {
    for(eff_name in names(G_encoded_list)) {
      info_map[[curr_id]] <- list(SNP=j, Type=eff_name)
      curr_id <- curr_id + 1
    }
  }
  
  selected_info <- data.frame()
  if(length(sel_idx) > 0) {
    selected_info <- do.call(rbind, lapply(sel_idx, function(id) {
      data.frame(
        Group_ID = id,
        SNP_Index = info_map[[id]]$SNP,
        Effect_Type = info_map[[id]]$Type,
        Power = grp_power[id]
      )
    }))
  }
  
  return(list(
    theta = current_theta, 
    selected_effects = selected_info,
    beta_full = new_beta,
    best_size = best_k,
    model = abess_fit,
    basis = Basis_mat
  ))
}




build_dynamic_design_matrix <- function(Genotypes, time_points, lop_order=3, effect_types=c("Add", "Dom")) {
  
  N <- nrow(Genotypes)
  P <- ncol(Genotypes)
  T_len <- length(time_points)
  
  # 1. LOP Basis
  Basis_mat <- get_LOP_basis(time_points, lop_order)
  n_basis <- ncol(Basis_mat)
  # 预扩展 Basis (N*T x n_basis)
  Basis_long <- do.call(rbind, replicate(N, Basis_mat, simplify=FALSE))
  
  # 2. 基因型编码
  G_encoded_list <- list()
  if ("Add" %in% effect_types) {
    # 加性 (-1, 0, 1) -> 中心化
    G_encoded_list[["Add"]] <- scale(Genotypes - 1, center=TRUE, scale=FALSE)
  }
  if ("Dom" %in% effect_types) {
    # 显性 (0, 1, 0) -> 中心化
    G_dom <- matrix(0, N, P); G_dom[Genotypes == 1] <- 1
    G_encoded_list[["Dom"]] <- scale(G_dom, center=TRUE, scale=FALSE)
  }
  
  # 3. 构建矩阵
  n_effects <- length(G_encoded_list)
  total_cols <- P * n_effects * n_basis
  X_tilde <- matrix(0, nrow = N * T_len, ncol = total_cols)
  groups <- integer(total_cols)
  
  col_idx <- 1
  curr_grp_id <- 1 # <--- 定义在这里
  
  # 映射表 (用于后续解析)
  info_map <- list()
  
  # 预扩展基因型
  G_long_list <- lapply(G_encoded_list, function(G) G[rep(1:N, each=T_len), ])
  
  cat(sprintf("   -> Building Matrix: %d Individuals, %d TimePoints, %d Features...\n", N, T_len, total_cols))
  
  for (j in 1:P) {
    for (eff_name in names(G_long_list)) {
      g_col <- G_long_list[[eff_name]][, j]
      
      
      block_val <- g_col * Basis_long 
      
      cols_range <- col_idx : (col_idx + n_basis - 1)
      X_tilde[, cols_range] <- block_val
      groups[cols_range] <- curr_grp_id
      
      info_map[[curr_grp_id]] <- list(SNP=j, Type=eff_name)
      
      col_idx <- col_idx + n_basis
      curr_grp_id <- curr_grp_id + 1
    }
  }
  
  return(list(X = X_tilde, groups = groups, info_map = info_map, basis = Basis_mat))
}

#  curve + Abess 联合求解
train_dynamic_GS <- function(
    Y_train, G_train, time_points, 
    mean_func = NULL, start_params = NULL,
    lop_order = 3, effect_types = c("Add", "Dom"),
    max_iter = 5, tol = 1e-4, support_size = 0:20
) {
  
  N <- nrow(Y_train)
  Y_vec <- as.vector(t(Y_train))
  t_vec_all <- rep(time_points, N)
  
  # --- Step 1: 构建设计矩阵 ---
  dm <- build_dynamic_design_matrix(G_train, time_points, lop_order, effect_types)
  X_tilde <- dm$X
  groups <- dm$groups
  
  # --- Step 2: 初始化 ODE ---
  cat(">>> [Init] Initial Mean Curve Fit...\n")
  resid_fun_init <- function(p) Y_vec - mean_func(p, t_vec_all)
  init_fit <- tryCatch(nls.lm(par=start_params, fn=resid_fun_init), 
                       error=function(e) stop("Init NLS failed."))
  current_theta <- coef(init_fit)
  current_beta <- rep(0, ncol(X_tilde))
  current_intercept <- 0
  
  best_abess_model <- NULL
  
  # --- Step 3: 迭代求解 ---
  for(iter in 1:max_iter) {
    
    # A. 更新遗传效应 (Abess)
    mu_curve <- mean_func(current_theta, time_points)
    Y_mean_removed <- Y_vec - rep(mu_curve, N)
    
    abess_fit <- abess(x = X_tilde, y = Y_mean_removed, group.index = groups, 
                       support.size = support_size, tune.type = "gic", family = "gaussian")
    
    best_k <- abess_fit$best.size
    best_coef_mat <- coef(abess_fit, support.size = best_k)
    new_intercept <- as.numeric(best_coef_mat[1, 1])
    new_beta <- as.numeric(best_coef_mat[-1, 1]) 
    G_hat_vec <- as.numeric(X_tilde %*% new_beta + new_intercept)
    
    n_selected <- length(unique(groups[new_beta != 0]))
    
    # B. 更新均值参数 (ODE)
    Y_clean_vec <- Y_vec - G_hat_vec
    resid_fun_iter <- function(p) Y_clean_vec - mean_func(p, t_vec_all)
    new_fit <- tryCatch(nls.lm(par=current_theta, fn=resid_fun_iter), error=function(e) NULL)
    
    if(is.null(new_fit)) break
    new_theta <- coef(new_fit)
    
    # C. 收敛检查
    diff_theta <- sum((new_theta - current_theta)^2)
    diff_beta <- sum((new_beta - current_beta)^2)
    
    cat(sprintf("Iter %d | Size: %d | dTheta: %.1e | dBeta: %.1e | K=%.2f\n", 
                iter, best_k, diff_theta, diff_beta, new_theta[1]))
    
    if(diff_theta < tol && diff_beta < tol) {
      cat(">>> Converged!\n")
      current_theta <- new_theta
      current_beta <- new_beta
      current_intercept <- new_intercept
      best_abess_model <- abess_fit
      break
    }
    current_theta <- new_theta
    current_beta <- new_beta
    current_intercept <- new_intercept
    best_abess_model <- abess_fit
  }
  
  # --- Step 4: 结果打包 ---
  # 解析选中的位点
  beta_mat <- matrix(current_beta, nrow=max(groups), byrow=TRUE)
  grp_power <- rowSums(beta_mat^2)
  sel_idx <- which(grp_power > 0)
  
  selected_info <- data.frame()
  if(length(sel_idx) > 0) {
    selected_info <- do.call(rbind, lapply(sel_idx, function(id) {
      data.frame(Group_ID = id, SNP_Index = dm$info_map[[id]]$SNP, 
                 Effect_Type = dm$info_map[[id]]$Type, Power = grp_power[id])
    }))
  }
  
  # 返回训练好的模型对象
  return(list(
    theta = current_theta,       # 最终曲线参数
    beta = current_beta,         # 最终遗传系数 (稀疏)
    intercept = current_intercept,
    selected_snps = selected_info, # 选中 SNP 列表
    lop_order = lop_order,       # 记录超参数供预测用
    effect_types = effect_types, # 记录超参数
    mean_func = mean_func,       # 记录均值函数
    time_points = time_points,    # 记录时间点
    model = best_abess_model
  ))
}

#预测新个体的动态轨迹
predict_dynamic_GS <- function(model_obj, G_new, t_new = NULL) {
  
  # 如果未指定时间点，默认使用训练时的时间点
  if(is.null(t_new)) t_new <- model_obj$time_points
  
  N_new <- nrow(G_new)
  T_new <- length(t_new)
  
  cat(sprintf(">>> Predicting for %d individuals over %d time points...\n", N_new, T_new))
  
  # 1. 预测均值曲线 (Population Mean Trajectory)
  # 使用训练好的 ODE 参数 theta
  # 注意: mean_func 需要能接受向量化时间输入
  t_vec_all <- rep(t_new, N_new)
  mu_pred <- model_obj$mean_func(model_obj$theta, t_vec_all)
  
  # 2. 预测遗传偏差 (Genetic Deviation)
  # 必须构建与训练集完全一致的设计矩阵结构
  dm_new <- build_dynamic_design_matrix(
    Genotypes = G_new, 
    time_points = t_new, 
    lop_order = model_obj$lop_order, 
    effect_types = model_obj$effect_types
  )
  
  # 核心预测公式: g_hat = X_new * beta_trained
  # model_obj$beta 是训练好的全维度系数向量
  intercept_val <- if(!is.null(model_obj$intercept)) model_obj$intercept else 0
  g_pred <- as.numeric(dm_new$X %*% model_obj$beta) + intercept_val
  
  # 3. 合成最终表型
  y_pred_vec <- mu_pred + g_pred
  
  # 整理成矩阵返回 (N x T)
  Y_pred_mat <- matrix(y_pred_vec, nrow=N_new, byrow=TRUE)
  colnames(Y_pred_mat) <- paste0("T_", t_new)
  rownames(Y_pred_mat) <- rownames(G_new)
  
  return(list(
    Y_pred = Y_pred_mat,      # 最终预测值
    Mean_Component = matrix(mu_pred, nrow=N_new, byrow=TRUE), # 均值部分
    Genetic_Component = matrix(g_pred, nrow=N_new, byrow=TRUE) # 遗传部分
  ))
}





cv_dynamic_GS <- function(Y_matrix, Genotypes, time_points, mean_func = NULL, start_params = NULL,lop_order = 3, effect_types = c("Add", "Dom"),
                          support_size = 0:20, n_folds = 5,max_iter=5,seed = 2025) {
  
  cat("==========================================================\n")
  cat(sprintf("   Running %d-Fold Cross-Validation for Dynamic GS   \n", n_folds))
  cat("==========================================================\n")
  
  set.seed(seed)
  N <- nrow(Y_matrix)
  T_len <- length(time_points)
  
  # 1. 创建 Folds
  folds <- createFolds(1:N, k = n_folds, list = TRUE, returnTrain = FALSE)
  
  # 存储所有个体的预测值 (初始化为 NA)
  Y_pred_cv <- matrix(NA, nrow=N, ncol=T_len)
  colnames(Y_pred_cv) <- paste0("T_", time_points)
  
  # 存储每一折的评估指标
  fold_metrics <- list()
  
  # 2. 循环执行交叉验证
  for(k in 1:n_folds) {
    cat(sprintf("\n>>> Processing Fold %d / %d ...\n", k, n_folds))
    
    test_idx <- folds[[k]]
    train_idx <- setdiff(1:N, test_idx)
    
    # 分割数据
    Y_train <- Y_matrix[train_idx, ]
    G_train <- Genotypes[train_idx, ]
    G_test  <- Genotypes[test_idx, , drop=FALSE] # drop=FALSE 防止单样本变成向量
    
    # 训练
    # 注意: 这里调用之前定义的 train_dynamic_GS
    model_k <- train_dynamic_GS(
      Y_train = Y_train,
      G_train = G_train,
      time_points = time_points,
      mean_func = mean_func,
      start_params = start_params,
      lop_order = lop_order,
      effect_types = effect_types,
      support_size = support_size,
      max_iter = max_iter, # CV 时可以减少迭代次数以提速
      tol = 1e-3
    )
    
    # 预测
    pred_res <- predict_dynamic_GS(model_k, G_test)
    
    # 填充结果
    Y_pred_cv[test_idx, ] <- pred_res$Y_pred
    
    # 计算该折的精度 (平均相关系数)
    cor_t <- sapply(1:T_len, function(t) cor(Y_matrix[test_idx, t], pred_res$Y_pred[, t]))
    fold_metrics[[k]] <- cor_t
    
    cat(sprintf("    Fold %d Avg Correlation: %.4f\n", k, mean(cor_t, na.rm=TRUE)))
  }
  
  # 3. 汇总评估
  cat("\n----------------------------------------------------------\n")
  cat("   Cross-Validation Summary   \n")
  cat("----------------------------------------------------------\n")
  
  # 计算每个时间点的总体相关系数
  overall_cor_by_time <- sapply(1:T_len, function(t) cor(Y_matrix[, t], Y_pred_cv[, t], use="complete.obs"))
  
  cat("Correlation by Time Point:\n")
  print(round(overall_cor_by_time, 3))
  cat(sprintf("\nOverall Mean Correlation: %.4f\n", mean(overall_cor_by_time, na.rm=TRUE)))
  
  # 4. 用全量数据训练最终模型 (Final Model)
  cat("\n>>> Retraining Final Model on All Data...\n")
  final_model <- train_dynamic_GS(
    Y_train = Y_matrix,
    G_train = Genotypes,
    time_points = time_points,
    mean_func = mean_func,
    start_params = start_params,
    lop_order = lop_order,
    effect_types = effect_types,
    support_size = support_size,
    max_iter = 10, # 最终模型用更严格的标准
    tol = 1e-4
  )
  
  cat("Done.\n")
  
  return(list(
    final_model = final_model,      # 可用于后续预测
    cv_predictions = Y_pred_cv,     # CV 过程中的预测值 (N x T)
    accuracy_by_time = overall_cor_by_time,
    fold_metrics = fold_metrics
  ))
}


predict_phenotypes <- function(trained_model, markers) {
  # trained_model: run_cv_dynamic_GS 返回的 $final_model
  # markers: 一个向量 (单样本) 或 矩阵 (多样本)
  
  # 1. 格式标准化
  if (is.vector(markers)) {
    # 如果是单样本向量，转为 1 x P 矩阵
    markers <- matrix(markers, nrow = 1)
  }
  
  # 2. 检查维度
  # 训练时用的 SNP 数量隐含在设计矩阵构建逻辑中
  # 这里简单检查列数是否匹配
  # 注意: 严格来说应该检查 SNP 名称，但为了简单这里只检查数量
  # 我们需要从 model_obj 中推断 P。
  # build_dynamic_design_matrix 会处理 P，这里假设用户输入的 markers 维度是对的
  
  cat(sprintf("Predicting phenotypes for %d individuals...\n", nrow(markers)))
  
  # 3. 调用核心预测函数
  # 如果用户想预测未来的时间点，可以在这里扩展参数 t_new
  # 默认使用训练时的时间点
  res <- predict_dynamic_GS(trained_model, markers)
  
  # 4. 返回友好的结果
  # 包含 总表型 (Y_pred)
  # 也包含 遗传育种值 (Genetic_Component) -> 这对育种选择最重要
  return(list(
    Predicted_Phenotype = res$Y_pred,
    Breeding_Value_Trajectory = res$Genetic_Component,
    Baseline_Growth = res$Mean_Component
  ))
}

