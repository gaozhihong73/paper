

# 计算模型权重 (基于 PNAS Eq. 2) ---
# 论文公式: w_i = exp(-error / sigma)
calculate_rde_weight <- function(rss, sigma = 1) {
  return(exp(-rss / sigma))
}


train_RDE_ensemble <- function(
    Y_matrix, Genotypes, time_points, 
    mean_func = NULL, start_params = NULL,
    lop_order = 3, effect_types = c("Add", "Dom"),
    
    # --- RDE 核心超参数 ---
    n_embeddings = 50,       # s: 随机嵌入的数量 (建议 30-100)
    subset_size = 200,       # L: 每个嵌入包含的变量(SNP)数量 (远小于 P)
    n_cores = 1              # 并行核数
) {
  
  cat("==========================================================\n")
  cat(sprintf("   [RDE Training] Ensemble Size: %d | Subset Size: %d   \n", n_embeddings, subset_size))
  cat("==========================================================\n")
  
  N <- nrow(Y_matrix)
  P <- ncol(Genotypes)
  
  # 1. 生成随机嵌入索引 (Random Tuples) [cite: 126, 572]
  #    "Randomly pick s tuples from {1...n} with replacement"
  embedding_indices <- replicate(n_embeddings, 
                                 sample(1:P, subset_size, replace = FALSE), 
                                 simplify = FALSE)
  
  # 2. 定义单个嵌入的训练任务 (Mapping Psi_l) 
  train_worker <- function(k) {
    # 提取子集
    idx <- embedding_indices[[k]]
    G_sub <- Genotypes[, idx, drop = FALSE]
    
    # 训练弱预测器 (Weak Predictor)
    # 使用 abess 在小子集上进行特征选择和参数估计
    # 注意: 这里 max_iter 可以设小一点，因为是弱预测器
    model <- tryCatch({
      train_dynamic_GS(
        Y_train = Y_matrix,
        G_train = G_sub,
        time_points = time_points,
        mean_func = mean_func,
        start_params = start_params,
        lop_order = lop_order,
        effect_types = effect_types,
        max_iter = 5,       
        tol = 1e-3,
        support_size = 0:min(20, subset_size) # 限制子模型复杂度
      )
    }, error = function(e) return(NULL))
    
    # 计算训练误差 (In-sample Error) 用于后续加权 [cite: 245, 578]
    rss <- NA
    if(!is.null(model)) {
      # 快速预测训练集
      pred <- predict_dynamic_GS(model, G_sub)
      # 计算残差平方和 (RSS)
      rss <- sum((as.vector(Y_matrix) - as.vector(pred$Y_pred))^2)
    }
    
    return(list(model = model, snp_idx = idx, training_error = rss))
  }
  
  # 3. 并行训练
  cat(">>> Training sub-models in parallel...\n")
  if(n_cores > 1) {
    cl <- makeCluster(n_cores)
    # 导出依赖函数
    clusterExport(cl, varlist = c("train_dynamic_GS", "build_dynamic_design_matrix", 
                                  "get_LOP_basis", "mean_func", "predict_dynamic_GS",
                                  "time_points", "start_params", "lop_order", "effect_types"), 
                  envir = environment())
    clusterEvalQ(cl, {library(minpack.lm); library(abess)})
    results <- parLapply(cl, 1:n_embeddings, train_worker)
    stopCluster(cl)
  } else {
    results <- lapply(1:n_embeddings, train_worker)
  }
  
  # 4. 过滤失败模型
  valid_results <- results[!sapply(results, function(x) is.null(x$model))]
  cat(sprintf(">>> RDE Training Complete. Valid Models: %d / %d\n", length(valid_results), n_embeddings))
  
  return(list(
    sub_models = valid_results,
    time_points = time_points,
    mean_func = mean_func,
    lop_order = lop_order,
    effect_types = effect_types
  ))
}



predict_RDE_ensemble <- function(rde_model, G_new, sigma_scale = 0.1) {
  
  sub_models <- rde_model$sub_models
  n_models <- length(sub_models)
  N_new <- nrow(G_new)
  T_len <- length(rde_model$time_points)
  
  # 1. 收集预测值和训练误差
  pred_array <- array(NA, dim = c(N_new, T_len, n_models))
  train_errors <- sapply(sub_models, function(x) x$training_error)
  
  # --- 改进权重计算 (Sharpening Weights) ---
  # 我们希望权重集中在表现最好的前 10%-20% 模型上，忽略那些只预测均值的“混子”模型
  # 策略：减去最小误差，缩小 sigma，放大差异
  min_err <- min(train_errors, na.rm = TRUE)
  rel_errors <- train_errors - min_err
  
  # sigma_scale 越小，权重越集中在最优模型上 (Top-k effect)
  # 使用误差分布的 10% 分位数作为缩放因子，确保只有头部模型有高权重
  effective_sigma <- quantile(rel_errors, 0.2, na.rm=TRUE) * sigma_scale
  if(effective_sigma == 0) effective_sigma <- 1e-5
  
  weights <- exp(-rel_errors / effective_sigma)
  weights <- weights / sum(weights) # 归一化
  
  # 打印一下权重分布情况，看看是否足够集中
  cat(sprintf(">>> Top-1 Model Weight: %.4f\n", max(weights)))
  cat(sprintf(">>> Effective Models (Weight > 0.01): %d / %d\n", sum(weights > 0.01), n_models))
  
  # 2. 生成预测矩阵
  cat(">>> Generating raw predictions...\n")
  for(k in 1:n_models) {
    m <- sub_models[[k]]
    G_sub <- G_new[, m$snp_idx, drop = FALSE]
    res <- predict_dynamic_GS(m$model, G_sub)
    pred_array[,,k] <- res$Y_pred
  }
  
  # 3. 聚合 (不再盲目剔除 Outlier)
  Y_final <- matrix(0, nrow = N_new, ncol = T_len)
  
  for(i in 1:N_new) {
    for(t in 1:T_len) {
      preds <- pred_array[i, t, ]
      
      # 直接使用强加权平均
      # 那些没学到东西的模型，因为 Training Error 较高，权重会被指数级压低
      Y_final[i, t] <- sum(preds * weights)
    }
  }
  
  colnames(Y_final) <- paste0("T_", rde_model$time_points)
  rownames(Y_final) <- rownames(G_new)
  
  return(list(
    Y_pred = Y_final,
    Prediction_Array = pred_array,
    Model_Weights = weights
  ))
}

run_rde_validation <- function(Y_matrix,Genotypes,time_points,mean_func,start_params,n_embeddings=100, subset_size=200,n_cores=1,sigma_scale=0.5) {
  
  # 1. 训练 RDE 模型 (集成 Abess + ODE)
  rde_model <- train_RDE_ensemble(
    Y_matrix = Y_matrix,
    Genotypes = Genotypes,
    time_points = time_points,
    mean_func = mean_func,
    start_params = start_params,
    n_embeddings = n_embeddings,   #  "Sufficient number of embeddings"
    subset_size =  subset_size,   #  "Low-dimensional embeddings"
    n_cores = n_cores
  )
  
  # 2. 预测 (利用分布信息提升精度)
  rde_results <- predict_RDE_ensemble(rde_model, Genotypes,sigma_scale = 0.5)
  
  #predict_RDE_ensemble <- function(rde_model, G_new, sigma_scale = 0.1)
  
  # 3. 评估精度
  acc_per_time <- sapply(1:ncol(Y_matrix), function(i) {
    cor(Y_matrix[, i], rde_results$Y_pred[, i], use = "complete.obs")
  })
  
  names(acc_per_time) <- paste0("Time_", sim_mixed$Time) 
  
  if(is.null(names(acc_per_time))) names(acc_per_time) <- paste0("T", 1:length(acc_per_time))
  
  cat(">>> Accuracy per Time Point (Correlation):\n")
  print(round(acc_per_time, 4))
  
  # 4. 计算平均精度 (作为摘要)
  cat(sprintf("\n>>> Mean Accuracy across all time points: %.4f\n", mean(acc_per_time, na.rm=TRUE)))
  
  return(list(rde_results=rde_results,acc_per_time=acc_per_time,model=rde_model))
}