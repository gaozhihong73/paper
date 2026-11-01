


solve_iterative_dynamic_system <- function(
    Y_matrix, Genotypes, time_points, 
    mean_func = NULL, 
    start_params = NULL,
    basis_type = "LOP",                
    lop_order = 3,                    
    max_iter = 10, tol = 1e-4,
    n_cores = detectCores() - 3  # <--- 默认使用 (总核数-1) 进行并行
) {
  
  if (is.null(mean_func)) stop("[Error] 'mean_func' is missing!")
  if (is.null(start_params)) stop("[Error] 'start_params' is missing!")
  
  N <- nrow(Y_matrix)
  T_len <- length(time_points)
  P <- ncol(Genotypes)
  
  # 拉直 Y
  Y_vec <- as.vector(t(Y_matrix)) 
  t_vec_all <- rep(time_points, N)
  
  # --- 1. 准备工作 ---
  
  # A. 开启并行集群
  if (n_cores > 1) {
    cat(sprintf(">>> Starting parallel cluster with %d cores...\n", n_cores))
    cl <- makeCluster(n_cores)
  } else {
    cl <- NULL
  }
  
  # 确保函数结束时关闭集群
  on.exit({
    if (!is.null(cl)) stopCluster(cl)
  })
  
  # B. 预计算 Basis 和 扩展的基因型矩阵 (移出循环以提速)
  cat(">>> Pre-calculating Design Matrix...\n")
  if(basis_type == "LOP") {
    Basis_mat <- get_LOP_basis(time_points, lop_order)
    n_basis <- ncol(Basis_mat)
  } else {
    stop("Only LOP supported.")
  }
  
  # 预先构建 X_tilde (设计矩阵)
  # 这是一个静态矩阵，不需要在循环里重复构建，节省巨大时间
  G_centered <- scale(Genotypes, center = TRUE, scale = FALSE)
  
  # 极速构建 Kronecker 积的方法 (避免显式循环)
  # X_tilde = G (x) Basis
  # 维度: (N*T) x (P*K)
  # 这里使用 lapply + do.call (比 for 循环快) 或者直接利用矩阵特性
  # 为了内存安全，我们还是用分块构建，但放在循环外只做一次
  
  # 快速构建 X_tilde
  X_tilde <- matrix(0, nrow=N*T_len, ncol=P*n_basis)
  Basis_long <- do.call(rbind, replicate(N, Basis_mat, simplify=FALSE)) # (NT x K)
  G_long <- G_centered[rep(1:N, each = T_len), ] # (NT x P)
  
  for(j in 1:P) {
    cols <- ((j-1)*n_basis + 1) : (j*n_basis)
    X_tilde[, cols] <- G_long[, j] * Basis_long
  }
  
  groups <- rep(1:P, each = n_basis)
  cat(">>> Design Matrix Built. Size:", dim(X_tilde), "\n")
  
  # --- 2. 初始化均值参数 ---
  cat(">>> Initializing Mean Curve...\n")
  resid_fun_init <- function(params) {
    preds <- mean_func(params, t_vec_all)
    return(Y_vec - preds)
  }
  init_fit <- nls.lm(par = start_params, fn = resid_fun_init, control = nls.lm.control(maxiter = 100))
  current_theta <- coef(init_fit)
  cat("Initial Theta:", round(current_theta, 3), "\n")
  
  # 初始化 LOP 系数 (用于计算收敛差值)
  current_beta <- rep(0, ncol(X_tilde)) 
  
  # --- 3. 迭代循环 ---
  for(iter in 1:max_iter) {
    
    # === Step A: Group Lasso (并行加速) ===
    
    # 计算当前均值曲线
    mu_curve <- mean_func(current_theta, time_points)
    Y_mean_removed <- Y_vec - rep(mu_curve, N)
    
    # 运行 CV Group Lasso
    # 注意: 把 cluster 传进去
    cv_fit <- cv.grpreg(X_tilde, Y_mean_removed, group=groups, penalty="grLasso", cluster=cl)
    
    # 获取最优 Beta
    new_beta <- coef(cv_fit, lambda=cv_fit$lambda.min)[-1] # 去掉截距
    
    # 预测遗传效应
    G_hat_vec <- predict(cv_fit, X_tilde, type="link", lambda=cv_fit$lambda.min)
    
    # 统计选中数量
    n_selected <- length(unique(groups[new_beta != 0]))
    
    # === Step B: 更新均值参数 (NLS) ===
    
    Y_clean_vec <- Y_vec - G_hat_vec
    
    resid_fun_iter <- function(params) {
      preds <- mean_func(params, t_vec_all)
      return(Y_clean_vec - preds)
    }
    
    new_fit <- tryCatch({
      nls.lm(par = current_theta, fn = resid_fun_iter)
    }, error = function(e) return(NULL))
    
    if(is.null(new_fit)) {
      warning("NLS failed to converge.")
      break
    }
    
    new_theta <- coef(new_fit)
    
    # === Step C: 双重收敛检查 ===
    
    # 1. 均值参数的变化
    diff_theta <- sum((new_theta - current_theta)^2)
    # 2. LOP 系数的变化 (归一化一下避免维度影响)
    diff_beta <- sum((new_beta - current_beta)^2) / (length(new_beta) + 1e-6)
    
    # 格式化打印
    cat(sprintf("Iter %2d | Theta Chg: %.6f | Beta Chg: %.6f | SNPs: %d | K=%.1f\n", 
                iter, diff_theta, diff_beta, n_selected, new_theta[1]))
    
    # 只有当 两者都稳定 时才停止
    if(diff_theta < tol && diff_beta < tol) {
      cat(">>> Converged! (Both Mean and Genetic Effects stabilized)\n")
      current_theta <- new_theta
      current_beta <- new_beta
      break
    }
    
    current_theta <- new_theta
    current_beta <- new_beta
  }
  
  return(list(theta = current_theta, model = cv_fit, basis = Basis_mat, beta = current_beta))
}





