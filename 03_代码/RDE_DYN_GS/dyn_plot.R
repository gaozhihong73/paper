
plot_dynamic_GS <- function(
    Y_obs,          # 真实值矩阵 (N x T)
    Y_pred,         # 预测值矩阵 (N x T)
    time_points,    # 时间点向量
    mean_func,      # [新增] 生长方程函数 (如 Logistic_curve)
    start_params,   # [新增] nls.lm 的初始参数 (如 c(K=30, r=0.5, ...))
    sample_ids = NULL, 
    filename = "Dynamic_GS_Report.pdf", 
    width = 8, height = 6
) {
  
  # --- 1. 参数检查与准备 ---
  if(is.null(sample_ids)) {
    # 默认取前 10 个
    n_show <- min(10, nrow(Y_obs))
    sample_ids <- rownames(Y_obs)[1:n_show]
    if(is.null(sample_ids)) sample_ids <- 1:n_show 
  }
  
  # 检查 ID 有效性
  valid_ids <- intersect(sample_ids, rownames(Y_obs))
  if(length(valid_ids) == 0 && !is.null(rownames(Y_obs))) {
    if(is.numeric(sample_ids)) {
      valid_ids <- rownames(Y_obs)[sample_ids]
    } else {
      stop("None of the provided sample_ids were found in Y_obs rownames.")
    }
  }
  
  cat(sprintf(">>> Generating PDF report: %s\n", filename))
  cat(sprintf(">>> Plotting %d individuals with NLS fitting...\n", length(valid_ids)))
  
  # --- 2. 打开 PDF 设备 ---
  pdf(file = filename, width = width, height = height)
  
  # --- 3. 循环绘图 ---
  for (id in valid_ids) {
    
    # A. 提取数据
    y_real <- as.numeric(Y_obs[id, ])
    y_hat  <- as.numeric(Y_pred[id, ])
    
    # B. [核心修改] 使用 nls.lm 对真实数据进行拟合
    #    目的是得到该个体的"理论生长轨迹" (Observed Trend)
    resid_fun_ind <- function(p) y_real - mean_func(p, time_points)
    
    fit_ind <- tryCatch({
      nls.lm(par = start_params, fn = resid_fun_ind)
    }, error = function(e) NULL)
    
    if(!is.null(fit_ind)) {
      # 拟合成功：生成理论曲线
      theta_ind <- coef(fit_ind)
      y_real_trend <- mean_func(theta_ind, time_points)
      fit_msg <- "Fit: Converged"
    } else {
      # 拟合失败：回退到原始数据 (极少情况)
      y_real_trend <- y_real
      fit_msg <- "Fit: Failed (Raw Data Used)"
    }
    
    # C. 构建绘图数据框
    df_plot <- data.frame(
      Time = time_points,
      Observed_Point = y_real,       # 原始散点 (带噪音)
      Observed_Trend = y_real_trend, # 拟合后的理论曲线 (干净)
      Predicted = y_hat              # 我们的 GS 模型预测值
    )
    
    # 计算精度指标
    acc <- cor(y_real, y_hat)
    mse <- mean((y_real - y_hat)^2)
    
    # D. 使用 ggplot2 绘图
    p <- ggplot(df_plot, aes(x = Time)) +
      # 1. 真实值散点 (黑点)
      geom_point(aes(y = Observed_Point, color = "Observed (Points)"), 
                 size = 2.5, alpha = 0.5) +
      
      # 2. 真实值的 NLS 拟合曲线 (灰色虚线)
      geom_line(aes(y = Observed_Trend, color = "Observed (Fit)", linetype = "Observed (Fit)"), 
                linewidth = 1) +
      
      # 3. 预测曲线 (红色实线)
      geom_line(aes(y = Predicted, color = "Predicted (GS)", linetype = "Predicted (GS)"), 
                linewidth = 1.2) +
      
      # 颜色映射
      scale_color_manual(name = "Legend", 
                         values = c("Observed (Points)" = "black", 
                                    "Observed (Fit)" = "gray40", 
                                    "Predicted (GS)" = "red")) +
      
      # 线型映射
      scale_linetype_manual(name = "Legend",
                            values = c("Observed (Fit)" = "dashed", 
                                       "Predicted (GS)" = "solid")) +
      
      # 标题与标签
      labs(title = paste0("Individual: ", id),
           subtitle = sprintf("Cor: %.4f | MSE: %.2f | %s", acc, mse, fit_msg),
           x = "Time Points", y = "Phenotype Value") +
      
      theme_bw() +
      theme(
        plot.title = element_text(size = 16, face = "bold"),
        legend.position = "bottom",
        legend.box = "horizontal",
        axis.text = element_text(size = 12),
        axis.title = element_text(size = 14)
      )
    
    # E. 打印
    print(p)
  }
  
  # --- 4. 关闭 ---
  dev.off()
  cat(">>> Done! PDF saved.\n")
}