# 设置包含CSV文件的文件夹路径
folder_path <- "./RGB/"

# 获取指定文件夹中所有CSV文件的完整路径
# pattern = "\\.csv$" 使用正则表达式匹配以.csv结尾的文件
# full.names = TRUE 返回完整的文件路径而不仅仅是文件名
csv_files <- list.files(folder_path, pattern = "\\.csv$", full.names = TRUE)

# 创建一个空的列表来存储所有的数据框
data_frames <- list()

# 遍历每个CSV文件
for (file in csv_files) {
  # 获取不带扩展名的文件名，用作数据框的名称
  # basename(file) 提取文件名部分
  # gsub("\\.csv$", "", ...) 移除.csv扩展名
  file_name_no_ext <- gsub("\\.csv$", "", basename(file))
  
  # 读取CSV文件为数据框
  # stringsAsFactors = FALSE 确保字符串不被自动转换为因子
  df <- read.csv(file, stringsAsFactors = FALSE)
  
  # 检查第一列是否有重复的ID
  # anyDuplicated()函数返回第一个重复元素的位置，如果没有重复则返回0
  if (anyDuplicated(df[, 1]) > 0) {
    # 如果发现重复ID，打印警告信息
    print(paste("Duplicate IDs found in file:", basename(file)))
    
    # 使用make.unique()函数为重复的ID添加后缀使其唯一
    # 例如：如果有重复的"ABC"，会变成"ABC", "ABC.1", "ABC.2"等
    df[, 1] <- make.unique(as.character(df[, 1]))
  }
  
  # 将第一列设置为行名
  rownames(df) <- df[, 1]
  
  # 删除第一列（因为已经设为行名，不再需要作为数据列）
  df <- df[, -1]
  
  # 将处理后的数据框存储到列表中，使用文件名作为键
  data_frames[[file_name_no_ext]] <- df
}

# 提取所有数据框的行名
row_names_list <- lapply(data_frames, rownames)

# 检查所有数据框的行名是否一致
# sapply()对每个数据框的行名与第一个数据框的行名进行比较
# all()函数检查是否所有比较结果都为TRUE
consistent <- all(sapply(row_names_list, function(x) all(x == row_names_list[[1]])))

if (consistent) {
  # 如果所有CSV文件的行名都一致，打印确认信息
  print("All CSV files have consistent row names.")
} else {
  # 如果行名不一致，进行处理
  print("Row names are not consistent across all files. Sorting row names...")
  
  # 找到所有数据框共同的行名
  # Reduce(intersect, ...) 找到所有列表的交集
  common_rows <- Reduce(intersect, row_names_list)
  
  # 对每个数据框进行处理
  data_frames <- lapply(data_frames, function(df) {
    # 只保留共同的行
    # drop = FALSE 确保即使只有一行也保持数据框格式
    df <- df[common_rows, , drop = FALSE]
    
    # 按行名排序
    df[order(rownames(df)), ]
  })
  
  print("Row names have been sorted and matched.")
}

# 显示最终的数据框列表
data_frames

# 将处理后的数据框列表保存为RData文件
# 文件名为"pheno.RData"，可以在以后的R会话中加载
save(data_frames, file = "pheno.RData")