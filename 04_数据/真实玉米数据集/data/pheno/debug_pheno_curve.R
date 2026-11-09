#1. CIVE (Color Index of Vegetation Extraction)
#反映的性状：植物的光合有效辐射吸收率（可通过绿色通道反映）和叶绿素含量。
#与玉米性状的关系：CIVE 可以帮助判断玉米植株的叶绿素含量，进而反映植物的光合作用能力。叶绿素是植物光合作用的关键，因此较高的 CIVE 值可能表明玉米植株健康，且光合作用较强，适合生长。

#2. COM (Complementary Vegetation Index)
#反映的性状：植被健康和绿度。
#与玉米性状的关系：COM 主要通过多波段信息补充植被信息，有助于在不同环境或光照条件下提高植被健康评估的准确性。对于玉米，COM 可用于监测植株的健康状态，尤其是在光照条件变化较大的区域。

#3. ExG (Excess Green)
#反映的性状：植株的绿色覆盖度与健康状况。
#与玉米性状的关系：ExG 提取了图像中的绿色信息，用于反映植物的叶绿素含量。ExG 的增加通常意味着叶片生长良好，绿度较高，因此玉米植株健康，生长状况良好。

#4. ExGR (Excess Green Ratio)
#反映的性状：植被绿色覆盖的比例与健康状况。
#与玉米性状的关系：ExGR 是对 ExG 的进一步改进，常用于区分绿植和非绿植区域。在玉米监测中，ExGR 可帮助识别玉米叶片的绿色含量，以及整体的生长状况。

#5. ExR (Excess Red)
#反映的性状：植株的红色反射率。
#与玉米性状的关系：ExR 反映了红色波段的反射强度，通常与叶片的色素（如叶绿素）密切相关。较高的 ExR 值可能意味着玉米植株的红色反射强度较高，可能是由于植株衰老、叶片黄化或抗逆环境条件下的光合作用效率低。

#6. GLI (Green Leaf Index)
#反映的性状：植株的绿度，尤其是叶片的绿色程度。
#与玉米性状的关系：GLI 反映了玉米植株的绿度，是判断玉米植株健康的关键指标。健康的玉米植株通常具有较高的 GLI 值，这与植株的生长、发育和光合作用直接相关。

#7. GLI2 (Green Leaf Index 2)
#反映的性状：更精确的植株绿度和生长状况。
#与玉米性状的关系：GLI2 是 GLI 的改进版本，能够更有效地提取植物的绿色区域。在玉米生长过程中，GLI2 可以帮助更准确地反映玉米的生长和健康状态。

#8. gn (Green Normalized Difference Index)
#反映的性状：植株的绿度。
#与玉米性状的关系：gn 指数特别敏感于绿色区域，常用于估算植物的绿度。较高的 gn 值可能表示玉米植株叶片健康，绿度较高，表明植物在光合作用中处于较佳状态。

#9. GreenCoverage (Green Coverage)
#反映的性状：植被覆盖度。
#与玉米性状的关系：GreenCoverage 用于估算玉米田间绿色覆盖的比例，这反映了玉米田的整体覆盖状况和生长程度。较高的 GreenCoverage 值通常与玉米植株的生长旺盛、叶片展开密切相关。

#10. GRVI (Green-Red Vegetation Index)
#反映的性状：植株的生长健康状况。
#与玉米性状的关系：GRVI 可以帮助区分植被和其他地物，尤其在不同的植物和土壤反射条件下。对于玉米，较高的 GRVI 值表明玉米植株健康，生长良好。

#11. IKAW (Infrared Kauth-Thomas Transformation)
#反映的性状：植株的健康状况，特别是光合作用与植被的水分状况。
#与玉米性状的关系：IKAW 基于红外波段和可见光波段的数据，适用于评估植株的健康状况。玉米植株的健康状态可以通过 IKAW 来监测，尤其是在不同生长阶段的变化。

#12. ints (Intensity)
#反映的性状：图像反射光的强度。
#与玉米性状的关系：Intensity 是图像的整体反射强度，它可以用于评估植被的生长密度，较强的反射强度可能与玉米植株的叶片覆盖度、植物群体的密集度相关。

#13. MGRVI (Modified Green-Red Vegetation Index)
#反映的性状：改进后的植被健康和生长情况。
#与玉米性状的关系：MGRVI 是 GRVI 的改进版本，旨在减少大气效应和其他非植被因素的干扰，能更精确地反映玉米的生长状态和健康状况。

#14. NDI (Normalized Difference Index)
#反映的性状：植被的健康状况和叶绿素含量。
#与玉米性状的关系：NDI 是一个常用的植被指数，用于监测玉米植株的健康、绿度和叶绿素含量。较高的 NDI 值通常表示植株健康，叶绿素含量丰富。

#15. RGB_R (Red channel of RGB image)
#反映的性状：玉米植株红色反射特征。
#与玉米性状的关系：该指数反映了红色波段的反射强度，通常与植物衰老和健康状况密切相关。较高的红色反射可能意味着叶片黄化或老化。

#16. RGBVI (RGB Vegetation Index)
#反映的性状：植株的绿色覆盖度与健康状况。
#与玉米性状的关系：RGBVI 是结合 RGB 图像的植被指数，它反映了玉米植株的绿色覆盖和健康状况。适用于植被监测和土壤-植被分离。

#17. rn (Red-Blue Normalized Difference Index)
#反映的性状：植株绿度与健康状况。
#与玉米性状的关系：rn 通过红色和蓝色波段的反射率差异来计算，能够有效地提取玉米植株的绿色信息。

#18. VARI (Visible Atmospherically Resistant Index)
#反映的性状：植株的绿色程度与抗大气影响能力。
#与玉米性状的关系：VARI 在有较强大气干扰的条件下，依然能有效反映植被的绿度。它是监测玉米健康，尤其是在不同大气条件下的有效工具。

#19. veg (Vegetation Index)
#反映的性状：植株的健康和绿色覆盖度。
#与玉米性状的关系：通常是 NDVI 或类似指数，反映了植被的健康状况、光合能力以及叶绿素含量。


#GRVI
#double_logistic_model <- nls(y ~ (K1 / (1 + exp(-r1 * (t - t01)))) - 
#                               (K2 / (1 + exp(-r2 * (t - t02)))),
#                             data = data_frame,
#                             start = list(K1 = 0.2, r1 = 0.1, t01 = 30, 
#                                          K2 = 0.15, r2 = 0.1, t02 = 70))
#summary(double_logistic_model)
#K1 (0.3128)：第一个逻辑斯蒂模型的最大值，表示前期增长的峰值高度。
#r1 (0.1290)：第一个逻辑斯蒂模型的增长速率，表示早期增长的速度。
#t01 (32.47)：第一个逻辑斯蒂模型的拐点时间，大约在32天时达到增长拐点。
#K2 (0.2209)：第二个逻辑斯蒂模型的最大值，表示后期的下降趋势的幅度。
#r2 (0.1252)：第二个逻辑斯蒂模型的增长速率，表示后期衰减的速度。
#t02 (73.90)：第二个逻辑斯蒂模型的拐点时间，大约在74天时开始明显下降。



# 定义一个函数用于填充矩阵中的缺失值
fill_missing_values <- function(matrix) {
  # 将输入的矩阵转换为数据框格式
  # stringsAsFactors = FALSE 确保字符串不被转换为因子
  matrix <- as.data.frame(matrix, stringsAsFactors = FALSE)
  
  # 遍历数据框的每一列
  for (col in 1:ncol(matrix)) {
    # 提取当前列的所有值
    column_values <- matrix[[col]]
    
    # 尝试将当前列的值转换为数值型
    column_values_numeric <- as.numeric(column_values)
    
    # 识别无效值（NA值或无穷大值）
    # is.na() 检测NA值，is.infinite() 检测无穷大值
    invalid_values <- is.na(column_values_numeric) | is.infinite(column_values_numeric)
    
    # 提取有效的数值（排除NA和无穷大值）
    valid_column_values <- column_values_numeric[!invalid_values]
    
    # 如果有有效值，计算平均值
    if (length(valid_column_values) > 0) {
      column_mean <- mean(valid_column_values, na.rm = TRUE)
    } else {
      # 如果没有有效值，平均值设为NA
      column_mean <- NA
    }
    
    # 用平均值填充无效值的位置
    column_values_numeric[invalid_values] <- column_mean
    
    # 将处理后的列赋回数据框
    matrix[[col]] <- column_values_numeric
  }
  
  # 返回处理后的数据框
  return(matrix)
}

# 加载之前保存的表型数据
load("./final/pheno.RData")

# 获取所有数据框的名称（所有表型指标的名称）
allpn <- names(data_frames)

# 提取第6个数据框作为示例
p1 <- data_frames[[6]]

# 从列名中提取数值信息
# gsub("X", "", ...) 移除列名前缀"X"，然后转换为数值
# 这些数值可能代表时间点、波长或其他测量参数
tt <- as.numeric(gsub("X", "",colnames(p1)))

# 创建PDF文件用于保存图形，设置高度5英寸，宽度6英寸
pdf("tmp_pheno.pdf", heigh=5, width=6)

# 循环处理前25个表型数据并绘图
for(i in 1:25){
  # 提取第i个数据框
  p1 <- data_frames[[i]]
  
  # 填充该数据框中的缺失值
  p1 <- fill_missing_values(p1)
  
  # 绘制折线图
  # x轴：tt（从列名提取的数值）
  # y轴：每列的平均值（colMeans计算列均值，na.rm=T忽略NA值）
  # type="l" 指定为折线图
  # main=allpn[i] 设置图标题为当前数据框的名称
  plot(tt, colMeans(p1, na.rm=T), type="l", main=allpn[i])
}

# 关闭PDF设备，保存文件
dev.off()

#=============== 数据选择和配置部分 ===============

# 定义要选择的数据框索引
# 这19个索引对应特定的表型指标
ii <- c(2,3,5,6,7,8,9,10,11,12,13,14,16,17,20,21,22,23,24)

# 为每个选择的指标定义加法调整值
# 这些值可能用于数据标准化或校正
addv <- c(15,5,0,1,0,0,0,0,0,0.1,0,100,0.2,0.2,100,0,0,0.2,0)

# 为每个选择的指标定义符号/方向
# "+" 可能表示正相关，"/" 可能表示比值或特殊处理
signc <- c(rep("+",11),"/","+","+","/","+","+","+","+")

# 定义每个选择指标的简化名称
# 这些是植物表型分析中常用的植被指数缩写：
# CIVE: 颜色指数植被提取
# COM: 组合指数
# ExG: 超绿指数
# ExGR: 超绿减去超红指数
# ExR: 超红指数
# GLI: 绿叶指数
# GLI2: 绿叶指数2
# gn: 绿色标准化
# GreenCoverage: 绿色覆盖度
# GRVI: 绿红植被指数
# IKAW: IKAW指数
# ints: 强度
# MGRVI: 修正绿红植被指数
# NDI: 标准化差异指数
# RGB_R: RGB红色分量
# RGBVI: RGB植被指数
# rn: 红色标准化
# VARI: 可见光大气阻抗植被指数
# veg: 植被指数
spn <- c("CIVE","COM","ExG","ExGR","ExR","GLI","GLI2","gn","GreenCoverage","GRVI","IKAW","ints",
         "MGRVI","NDI","RGB_R","RGBVI","rn","VARI","veg")

# 将所有选择参数组织成一个列表
# si: 选择的索引
# addv: 加法调整值
# signc: 符号/方向
# spn: 简化名称
pheno_select <- list(si=ii, addv=addv, signc=signc, spn=spn)

# 保存选择参数到RData文件
save(pheno_select, file="pheno_select.RData")