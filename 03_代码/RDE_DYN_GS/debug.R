




pdf("tmp.pdf")
i= 212
plot(1:15,sim_mixed$Y[i,])
lines(1:15,cv_results_abess$cv_predictions[i,],col="red")
lines(1:15,cv_rde_result$rde_results$Y_pred[i,],col="blue")
dev.off()





library(minpack.lm)

# ==============================================================================
# 1. 定义机制注册表 (包含方程 + 智能初始值逻辑)
# ==============================================================================

# --- 辅助函数：无t0版方程定义 ---
Logistic_NoT0 <- function(p, x) p[1] / (1 + p[2] * exp(-p[3] * x))
Gompertz_NoT0 <- function(p, x) p[1] * exp(-p[2] * exp(-p[3] * x))
Hill_4Param   <- function(p, x) p[1] + (p[2] - p[1]) / (1 + (x / p[3])^p[4])

# --- 注册表 (Dictionary) ---
MECHANISM_REGISTRY <- list(
  
  # === 选项 1: Logistic (适合标准生长) ===
  "Logistic" = list(
    fn = Logistic_NoT0,
    # 初始值猜测逻辑: 根据数据的 max, min 和斜率估算
    init_fn = function(x, y) {
      K <- max(y)
      y0 <- max(1e-6, y[1]) # 防止除0
      b <- (K / y0) - 1
      if(b < 0) b <- 0.1
      r <- 0.1 # 默认速率，也可以通过 log(y) 估算
      return(c(K=K, b=b, r=r))
    }
  ),
  
  # === 选项 2: Gompertz (适合肿瘤/细菌) ===
  "Gompertz" = list(
    fn = Gompertz_NoT0,
    init_fn = function(x, y) {
      K <- max(y)
      y0 <- max(1e-6, y[1])
      b <- -log(y0 / K)
      if(b < 0) b <- 0.1
      return(c(K=K, b=b, r=0.1))
    }
  ),
  
  # === 选项 3: Hill Equation (适合药物剂量反应) ===
  "Hill" = list(
    fn = Hill_4Param,
    init_fn = function(x, y) {
      # 药物反应通常是下降的 (E_inf < E_0)
      # 或者是上升的 (E_inf > E_0)
      # 我们直接取极值
      E_inf <- min(y) # Bottom
      E_0   <- max(y) # Top
      
      # 估算 IC50: 找到 y 值最接近中间值的那个 x
      mid_y <- (E_inf + E_0) / 2
      idx <- which.min(abs(y - mid_y))
      IC50 <- x[idx]
      if(IC50 == 0) IC50 <- mean(x)
      
      return(c(E_inf=E_inf, E_0=E_0, IC50=IC50, H=1.0))
    }
  )
  
  # 您可以在这里继续添加 "Richards", "DoubleLogistic" 等...
)