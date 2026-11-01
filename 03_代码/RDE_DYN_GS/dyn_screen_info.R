





# DC-SIS (向量化加速版)
# --- 辅助函数：计算欧氏距离矩阵 (Euclidean Distance Matrix) ---
# 计算所有样本对之间的距离 ||Y_i - Y_j||
get_dist_matrix_Y <- function(Y) {
  # Y 是 N x T 矩阵
  # dist 函数计算行与行之间的距离，返回 dist 对象，转为矩阵
  as.matrix(dist(Y, method = "euclidean"))
}

get_dist_matrix_X <- function(snp) {
  as.matrix(dist(snp, method = "euclidean")) # 对于一维数据等同于 abs(xi - xj)
}

calc_dcov_sq <- function(D1, D2) {
  n <- nrow(D1)
  S1 <- sum(D1 * D2) / (n^2)
  S2 <- mean(D1) * mean(D2)
  S3 <- sum(rowSums(D1) * rowSums(D2)) / (n^3)
  dcov2 <- S1 + S2 - 2 * S3
  return(dcov2)
}

DC.sis <- function(dat,screen_factor=2.0) {
  
  cat("Running DC-SIS (Pure R Optimized)...\n")
  
  y <- dat$pheno_y     # 表型矩阵 N x T
  snps <- dat$snps  # 基因型矩阵 N x P
  
  nn <- nrow(y)
  P <- ncol(snps)
  # 1. 预先计算 Y 的距离矩阵 (只算一次，大幅加速)
  cat("Calculating Distance Matrix for Y...\n")
  dist_Y <- get_dist_matrix_Y(y)
  
  # 预先计算 dCov(Y, Y) 的各部分，用于分母标准化 (dVar(Y))
  dvar2_Y <- calc_dcov_sq(dist_Y, dist_Y)
  snpdc <- numeric(P)
  names(snpdc) <- colnames(snps)
  
  # 2. 循环遍历每个 SNP
  cat("Scanning SNPs...\n")
  # 添加进度条，因为 P 可能很大
  pb <- txtProgressBar(min = 0, max = P, style = 3)
  
  for(j in 1:P) {
    snp_vec <- snps[, j]
    
    # 处理缺失值
    if(any(is.na(snp_vec))) {
      idx <- !is.na(snp_vec)
      curr_y <- y[idx, ]
      curr_snp <- snp_vec[idx]
      curr_dist_Y <- get_dist_matrix_Y(curr_y)
      curr_dvar2_Y <- calc_dcov_sq(curr_dist_Y, curr_dist_Y)
    } else {
      curr_snp <- snp_vec
      curr_dist_Y <- dist_Y
      curr_dvar2_Y <- dvar2_Y
    }
    
    # 计算 X (SNP) 的距离矩阵
    dist_X <- get_dist_matrix_X(curr_snp)
    
    # 计算 dCov^2(Y, X) (分子)
    dcov2_YX <- calc_dcov_sq(curr_dist_Y, dist_X)
    
    # 计算 dVar^2(X) = dCov^2(X, X) (分母部分)
    dvar2_X <- calc_dcov_sq(dist_X, dist_X)
    
    # 计算距离相关系数 (Distance Correlation)
    # 防止分母为0
    denom <- sqrt(curr_dvar2_Y) * sqrt(dvar2_X)
    
    if (denom > 1e-10) {
      dcorryx <- sqrt(max(0, dcov2_YX)) / sqrt(denom) # max(0,...) 防止浮点误差导致微小负数
    } else {
      dcorryx <- 0
    }
    
    snpdc[j] <- dcorryx
    
    if(j %% 100 == 0) setTxtProgressBar(pb, j)
  }
  close(pb)
  
  #筛选 (Screening)
  # 按照 d = n / log(n) 进行阈值截断 (Fan & Lv, 2008 经典阈值，或者直接选 Top K)
  d_base <- floor(nn / log(nn))
  d_thresh <- floor(d_base * screen_factor) # <--- 乘以放宽系数
  
  # 保护措施：如果放宽后超过了 SNP 总数，就取 SNP 总数
  if(d_thresh > P) d_thresh <- P
  # 保护措施：通常筛选数量不建议超过样本量 (虽然 SIS 可以，但为了后续分析方便)
  # 如果你后续用 LASSO，d_thresh 可以接近 N
  # if(d_thresh > nn - 1) d_thresh <- nn - 1 
  
  cat("\n------------------------------------------------\n")
  cat(sprintf("Screening Threshold: Top %d SNPs (Base: %d, Factor: %.1f)\n", 
              d_thresh, d_base, screen_factor))
  cat("------------------------------------------------\n")
  
  sorted_indices <- order(snpdc, decreasing = TRUE)
  rr <- snpdc[sorted_indices][1:d_thresh]
  
  sissnp <- snps[, names(rr)]
  ii.id <- sorted_indices[1:d_thresh]
  # 排序
  sorted_indices <- order(snpdc, decreasing = TRUE)
  rr <- snpdc[sorted_indices][1:d_thresh]
  
  rrnames <- names(rr)
  sissnp <- snps[, names(rr)]
  ii.id <- sorted_indices[1:d_thresh]
  
  return(list(rr = rr, sisnp = sissnp, iid = ii.id, all_dcor = snpdc))
}


DC.sis.AD <- function(dat, screen_factor = 2.0) {
  cat("Running DC-SIS with Additive & Dominance Separation...\n")
  
  # --- 1. 数据提取 ---
  if (!is.null(dat$Y)) y <- dat$Y else y <- dat$pheno_y
  if (!is.null(dat$G)) snps <- dat$G else snps <- dat$snps
  
  nn <- nrow(y)
  P <- ncol(snps)
  
  # 确保 SNP 是 N x P
  if (nrow(snps) != nn && ncol(snps) == nn) snps <- t(snps)
   # 构造加性(A)和显性(D)矩阵
  cat("Decomposing Genotypes into Additive and Dominance components...\n")
  
  # 1. 加性编码 (Additive): 0, 1, 2 -> -1, 0, 1
  Mat_A <- snps - 1
  colnames(Mat_A) <- paste0(colnames(snps), "_Add")
  
  # 2. 显性编码 (Dominance): 0, 1, 2 -> 0, 1, 0
  Mat_D <- matrix(0, nrow = nn, ncol = P)
  # 只有基因型为 1 (杂合子) 的地方设为 1
  Mat_D[snps == 1] <- 1
  colnames(Mat_D) <- paste0(colnames(snps), "_Dom")
  
  # 3. 合并成 N x 2P 的大矩阵
  X_full <- cbind(Mat_A, Mat_D)
  
  cat("Total Features to Screen:", ncol(X_full), "(", P, "SNPs x 2 components)\n")
  
  # 预计算 Y 的距离矩阵
  y <- as.matrix(y)
  calc_dcov_sq <- function(D1, D2) {
    n <- nrow(D1)
    S1 <- sum(D1 * D2) / (n^2)
    S2 <- mean(D1) * mean(D2)
    S3 <- sum(rowSums(D1) * rowSums(D2)) / (n^3)
    return(S1 + S2 - 2 * S3)
  }
  
  dist_Y <- as.matrix(dist(y, method = "euclidean"))
  dvar2_Y <- calc_dcov_sq(dist_Y, dist_Y)
  
  # 扫描所有 2P 个特征
  n_features <- ncol(X_full)
  feat_dc <- numeric(n_features)
  names(feat_dc) <- colnames(X_full)
  
  cat("Scanning features...\n")
  pb <- txtProgressBar(min = 0, max = n_features, style = 3)
  
  for(j in 1:n_features) {
    curr_feat <- X_full[, j]
    dist_X <- as.matrix(dist(curr_feat, method = "euclidean")) # 0/1 或 -1/0/1 的距离
    
    dcov2_YX <- calc_dcov_sq(dist_Y, dist_X)
    dvar2_X <- calc_dcov_sq(dist_X, dist_X)
    
    denom <- sqrt(dvar2_Y) * sqrt(dvar2_X)
    if (denom > 1e-10) {
      dcorryx <- sqrt(max(0, dcov2_YX)) / sqrt(denom)
    } else {
      dcorryx <- 0
    }
    feat_dc[j] <- dcorryx
    if(j %% 100 == 0) setTxtProgressBar(pb, j)
  }
  close(pb)
  
  # 注意：因为特征数量翻倍了，我们在筛选时，d_thresh 是针对特征数的
  d_base <- floor(nn / log(nn))
  d_thresh <- floor(d_base * screen_factor)
  
  # 排序并取 Top K
  sorted_indices <- order(feat_dc, decreasing = TRUE)
  rr <- feat_dc[sorted_indices][1:d_thresh]
  
  sissnp <- X_full[, names(rr)] # 选出来的是 A 或 D 的分量
  
  cat("\n------------------------------------------------\n")
  cat(sprintf("Screening Threshold: Top %d Features\n", d_thresh))
  cat("------------------------------------------------\n")
  
  return(list(rr = rr, sisnp = sissnp, all_scores = feat_dc))
}




