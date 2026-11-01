




library(minpack.lm)
library(abess)
library(Matrix) # 推荐使用稀疏矩阵处理大规模数据

# ==============================================================================
# 模块 A1: 动态特征展平器
# 作用: 将动态组学/环境数据 (N x T) 展平为宽矩阵 (N x T)，每一列代表 "Time_t 的特征"
# ==============================================================================
flatten_dynamic_features <- function(dynamic_list, time_labels) {
  if (is.null(dynamic_list) || length(dynamic_list) == 0) return(NULL)
  
  flattened_mat <- NULL
  
  for (name in names(dynamic_list)) {
    # 假设输入是一个列表，每个元素是 [N x T] 矩阵 或 [N x T x P] 数组
    # 这里处理最简单的情况: List of [N x T] matrices (比如一种菌的动态)
    # 或者 List of Lists
    
    mat <- dynamic_list[[name]]
    if(is.matrix(mat)) {
      colnames(mat) <- paste0(name, "_T", time_labels)
      if (is.null(flattened_mat)) flattened_mat <- mat else flattened_mat <- cbind(flattened_mat, mat)
    }
  }
  return(flattened_mat)
}

# ==============================================================================
# 模块 A2: 通用特征组合器 (Feature Combiner)
# 作用: 将所有来源的数据合并，并标记哪些是"固定效应"(如群体结构)，哪些是"候选效应"
# ==============================================================================
prepare_feature_store <- function(
    Genotypes = NULL,         # [候选] 静态 SNP (N x P)
    Static_Omics = NULL,      # [候选] 静态组学 (N x Q)
    Dynamic_Omics = NULL,     # [候选] 动态组学列表 (List of N x T)
    Population_Struct = NULL, # [固定] 群体结构 PCs (N x K) - 强制保留
    Env_Covariates = NULL,    # [固定] 环境因子 (N x E) - 强制保留
    time_points
) {
  
  # 1. 处理固定效应 (Must keep)
  Fixed_Mat <- NULL
  if(!is.null(Population_Struct)) Fixed_Mat <- cbind(Fixed_Mat, Population_Struct)
  if(!is.null(Env_Covariates))    Fixed_Mat <- cbind(Fixed_Mat, Env_Covariates)
  
  # 2. 处理候选效应 (Selection candidates)
  #    A. 展平动态数据
  Dyn_Flat <- flatten_dynamic_features(Dynamic_Omics, time_points)
  
  #    B. 合并所有候选
  Candidate_Mat <- NULL
  if(!is.null(Genotypes))    Candidate_Mat <- cbind(Candidate_Mat, Genotypes)
  if(!is.null(Static_Omics)) Candidate_Mat <- cbind(Candidate_Mat, Static_Omics)
  if(!is.null(Dyn_Flat))     Candidate_Mat <- cbind(Candidate_Mat, Dyn_Flat)
  
  # 3. 记录元数据 (为了后续知道谁是谁)
  #    feature_type: 0 = 固定(不惩罚), 1 = 候选(惩罚)
  
  n_fixed <- if(is.null(Fixed_Mat)) 0 else ncol(Fixed_Mat)
  n_cand  <- if(is.null(Candidate_Mat)) 0 else ncol(Candidate_Mat)
  
  if (n_fixed + n_cand == 0) stop("Error: No input features provided!")
  
  # 合并成大矩阵 X_input
  X_input <- cbind(Fixed_Mat, Candidate_Mat)
  
  # 生成惩罚向量 (abess penalty.factor: 0表示不惩罚/必选)
  penalty_vec <- c(rep(0, n_fixed), rep(1, n_cand))
  
  # 生成组索引 (简单起见，每个原始变量自成一组)
  group_vec <- 1:(n_fixed + n_cand)
  
  return(list(
    X = X_input,
    penalty = penalty_vec,
    groups = group_vec,
    n_fixed = n_fixed,
    n_cand = n_cand,
    feature_names = colnames(X_input)
  ))
}



# ==============================================================================
# 模块 B: 构建动态交互设计矩阵
# 核心逻辑: 将每个静态特征与时间基函数相乘，赋予其动态调节能力
# ==============================================================================
build_expanded_matrix <- function(Feature_Store, time_points, basis_type="LOP", lop_order=3) {
  
  X_input <- Feature_Store$X
  N <- nrow(X_input)
  P_total <- ncol(X_input)
  T_len <- length(time_points)
  
  # 1. 生成基函数矩阵 (T x K)
  if(basis_type == "LOP") {
    Basis_mat <- get_LOP_basis(time_points, lop_order) # 需确保此函数已定义
  } else {
    stop("Basis type not supported")
  }
  n_basis <- ncol(Basis_mat)
  
  # 2. 预分配大矩阵 (NT x PK)
  total_cols <- P_total * n_basis
  # 建议使用 Matrix 包的稀疏矩阵，如果数据量巨大
  X_tilde <- matrix(0, nrow = N * T_len, ncol = total_cols)
  
  # 扩展后的组索引和惩罚向量 (每个原始变量扩展出的 K 列属于同一组)
  expanded_groups <- integer(total_cols)
  expanded_penalty <- numeric(total_cols)
  
  # 辅助: 预扩展 Basis (N*T x K)
  Basis_long <- do.call(rbind, replicate(N, Basis_mat, simplify=FALSE))
  
  # 3. 填充矩阵 (核心循环)
  #    X_input 的每一列 j，都要与 Basis 的每一列 k 相乘
  cat(sprintf(">>> Building Matrix: %d samples x %d timepoints. Features: %d -> %d cols\n", 
              N, T_len, P_total, total_cols))
  
  col_ptr <- 1
  for(j in 1:P_total) {
    # 取出第 j 个特征列，扩展为长向量
    xj <- X_input[, j]
    xj_long <- rep(xj, each = T_len) # 重复 T 次
    
    # 逐列相乘 (Hadamard product style expansion)
    # block_val 维度: (NT x K)
    block_val <- xj_long * Basis_long 
    
    # 填入大矩阵
    idx_range <- col_ptr : (col_ptr + n_basis - 1)
    X_tilde[, idx_range] <- block_val
    
    # 记录组信息 (第 j 个原始变量的所有 basis 展开属于同一组)
    expanded_groups[idx_range] <- j 
    
    # 记录惩罚权重 (继承原始变量的属性)
    expanded_penalty[idx_range] <- Feature_Store$penalty[j]
    
    col_ptr <- col_ptr + n_basis
  }
  
  return(list(
    X_tilde = X_tilde,
    groups = expanded_groups,
    penalty = expanded_penalty,
    basis = Basis_mat,
    orig_names = Feature_Store$feature_names
  ))
}


# ==============================================================================
# 模块 C: 迭代求解器 (Core Solver)
# ==============================================================================
solve_iterative_system <- function(
    Y_matrix, Expanded_Design, time_points,
    mean_func, start_params,
    max_iter = 10, tol = 1e-4, support_size = NULL
) {
  
  N <- nrow(Y_matrix)
  Y_vec <- as.vector(t(Y_matrix)) # 拉平观测值
  t_vec_all <- rep(time_points, N)
  
  X_tilde <- Expanded_Design$X_tilde
  groups <- Expanded_Design$groups
  penalty <- Expanded_Design$penalty
  
  # --- 1. 初始化 ---
  # 先只拟合 ODE 均值曲线
  resid_fun_init <- function(p) Y_vec - mean_func(p, t_vec_all)
  init_fit <- tryCatch(nls.lm(par=start_params, fn=resid_fun_init), 
                       error=function(e) stop("Init ODE fit failed."))
  
  current_theta <- coef(init_fit)
  current_beta <- rep(0, ncol(X_tilde))
  current_intercept <- 0
  
  best_model <- NULL
  
  # --- 2. 迭代循环 ---
  for(iter in 1:max_iter) {
    
    # === A. 固定 ODE，更新 Beta (Abess) ===
    mu_curve <- mean_func(current_theta, time_points)
    Y_mean_removed <- Y_vec - rep(mu_curve, N) # 去除均值趋势
    
    # 调用 Abess
    # 关键点: 传入 penalty.factor，确保 Fixed Covariates (0) 不被剔除
    abess_fit <- abess(
      x = X_tilde, 
      y = Y_mean_removed,
      group.index = groups,
      penalty.factor = penalty, # <--- 关键修改: 强制保留 PC 等固定效应
      support.size = support_size, # 如果为 NULL，让 abess 自己选
      tune.type = "gic",
      family = "gaussian"
    )
    
    best_k <- abess_fit$best.size
    best_coef <- coef(abess_fit, support.size = best_k)
    
    new_intercept <- best_coef[1, 1]
    new_beta <- best_coef[-1, 1]
    
    # 计算遗传/环境部分的预测值
    G_hat_vec <- as.numeric(X_tilde %*% new_beta + new_intercept)
    
    # === B. 固定 Beta，更新 ODE (NLS) ===
    Y_clean_vec <- Y_vec - G_hat_vec
    
    resid_fun_iter <- function(p) Y_clean_vec - mean_func(p, t_vec_all)
    new_fit <- tryCatch(nls.lm(par=current_theta, fn=resid_fun_iter), 
                        error=function(e) NULL)
    
    if(is.null(new_fit)) { warning("NLS step failed."); break }
    new_theta <- coef(new_fit)
    
    # === C. 收敛判定 ===
    diff_theta <- sum((new_theta - current_theta)^2)
    diff_beta <- sum((new_beta - current_beta)^2)
    
    n_selected <- length(unique(groups[new_beta != 0]))
    
    cat(sprintf("Iter %d | Selected: %d | dTheta: %.2e | dBeta: %.2e\n", 
                iter, n_selected, diff_theta, diff_beta))
    
    if(diff_theta < tol && diff_beta < tol) {
      cat(">>> Converged successfully.\n")
      current_theta <- new_theta
      current_beta <- new_beta
      current_intercept <- new_intercept
      best_model <- abess_fit
      break
    }
    
    current_theta <- new_theta
    current_beta <- new_beta
    current_intercept <- new_intercept
    best_model <- abess_fit
  }
  
  return(list(
    theta = current_theta,
    beta = current_beta,
    intercept = current_intercept,
    model = best_model,
    design_info = Expanded_Design # 保存设计矩阵信息供预测用
  ))
}


# ==============================================================================
# 模块 D: 通用动态系统训练主入口
# ==============================================================================
train_universal_dynamic_framework <- function(
    Y_matrix,                  # 表型矩阵
    time_points,               # 时间点
    mean_func, start_params,   # ODE 机制
    
    # --- 输入数据组 ---
    Genotypes = NULL,          # 候选: SNP
    Population_Struct = NULL,  # 固定: 群体结构 PCs
    Env_Covariates = NULL,     # 固定: 环境因子
    Dynamic_Omics = NULL,      # 候选: 动态组学 (List)
    
    # --- 超参数 ---
    lop_order = 3,
    max_iter = 10,
    support_size = NULL        # 变量选择数量
) {
  
  # 1. 准备特征仓库 (处理展平、合并、标记固定效应)
  cat(">>> [Step 1] Preparing Feature Store...\n")
  Feat_Store <- prepare_feature_store(
    Genotypes = Genotypes,
    Population_Struct = Population_Struct,
    Env_Covariates = Env_Covariates,
    Dynamic_Omics = Dynamic_Omics,
    time_points = time_points
  )
  
  # 2. 构建扩展设计矩阵 (Basis Expansion)
  cat(">>> [Step 2] Building Dynamic Design Matrix...\n")
  Expanded_Design <- build_expanded_matrix(
    Feature_Store = Feat_Store, 
    time_points = time_points, 
    basis_type = "LOP", 
    lop_order = lop_order
  )
  
  # 3. 迭代求解
  cat(">>> [Step 3] Running Iterative Solver...\n")
  Result <- solve_iterative_system(
    Y_matrix = Y_matrix,
    Expanded_Design = Expanded_Design,
    time_points = time_points,
    mean_func = mean_func,
    start_params = start_params,
    max_iter = max_iter,
    support_size = support_size
  )
  
  cat(">>> Training Complete.\n")
  
  # 4. 附加原始信息以便解析
  Result$feature_names <- Feat_Store$feature_names
  Result$time_points <- time_points
  Result$mean_func <- mean_func
  
  return(Result)
}

# 假设您有以下数据:
# Y_corn:       [400 x 10] 玉米表型
# G_snp:        [400 x 5000] SNP 矩阵
# PCs:          [400 x 3] 群体结构 PC1-PC3
# Soil_Env:     [400 x 5] 5种土壤指标 (静态环境)
# Microbe_Dyn:  List(BugA = [400 x 10], BugB = [400 x 10]) 动态微生物

# 定义 ODE
Logistic_NoT0 <- function(p, t) p[1] / (1 + p[2]*exp(-p[3]*t))

# 一键训练
Unified_Model <- train_universal_dynamic_framework(
  Y_matrix = Y_corn,
  time_points = 1:10,
  mean_func = Logistic_NoT0,
  start_params = c(200, 10, 0.2),
  
  # --- 灵活的数据输入 ---
  Genotypes = G_snp,               # 会被 Abess 筛选
  Population_Struct = PCs,         # 会被强制保留 (校正)
  Env_Covariates = Soil_Env,       # 会被强制保留 (环境基线)
  Dynamic_Omics = Microbe_Dyn,     # 会被展平后筛选 (动态修饰)
  
  lop_order = 3
)

# 结果解析
# Unified_Model$theta -> 最终生长参数
# Unified_Model$beta  -> 包含 SNP、微生物、环境的所有效应系数