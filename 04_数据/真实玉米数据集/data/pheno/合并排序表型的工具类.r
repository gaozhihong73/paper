# 加载必要的包
library(dplyr)
library(stringr)

# 分析分组情况的函数
analyze_grouping <- function(df) {
  row_names <- rownames(df)
  
  # 提取M数字
  extract_m_number <- function(row_name) {
    if (str_detect(row_name, "^M\\d+-+")) {
      return(str_extract(row_name, "(?<=^M)\\d+(?=-)"))
    }
    return(NA)
  }
  
  m_numbers <- sapply(row_names, extract_m_number)
  
  # 统计每个分组的行数
  group_counts <- table(m_numbers[!is.na(m_numbers)])
  
  cat("=== 分组分析 ===\n")
  cat("总行数:", length(row_names), "\n")
  cat("可分组的行数:", sum(!is.na(m_numbers)), "\n")
  cat("分组数量:", length(group_counts), "\n")
  cat("每组行数分布:\n")
  print(table(group_counts))
  
  cat("\n前20个分组示例:\n")
  group_details <- data.frame(
    M_number = names(group_counts),
    count = as.numeric(group_counts),
    stringsAsFactors = FALSE
  )[1:min(20, length(group_counts)), ]
  
  for (i in 1:nrow(group_details)) {
    m_num <- group_details$M_number[i]
    pattern <- paste0("^M", m_num, "-")
    matching_rows <- row_names[str_detect(row_names, pattern)]
    cat("M", m_num, " (", group_details$count[i], "行): ", 
        paste(matching_rows, collapse = ", "), "\n", sep = "")
  }
  
  return(invisible(group_counts))
}


# 使用dplyr的现代化版本
merge_rows_dplyr <- function(df, output_path) {
  # 将数据转换为长格式便于分组处理
  df_long <- df %>%
    tibble::rownames_to_column("row_name") %>%
    # 提取M后面-前面的数字
    mutate(
      m_number = str_extract(row_name, "(?<=^M)\\d+(?=-)"),
      group_id = paste0("M", m_number)
    ) %>%
    # 只保留能够提取到M数字的行
    filter(!is.na(m_number))
  
  # 按分组计算均值
  result <- df_long %>%
    group_by(group_id) %>%
    summarise(across(where(is.numeric), mean, na.rm = TRUE), .groups = 'drop') %>%
    tibble::column_to_rownames("group_id")
  
  # 排序
  sorted_rows <- sort_rows_by_number(rownames(result))
  result <- result[sorted_rows, , drop = FALSE]

  # 保存
  write.csv(result, file=output_path, row.names = FALSE)
  print(paste0(output_path, "已保存"))
    
  return(result)
}

# 自定义行排序函数：按M后面的数字排序
sort_rows_by_number <- function(row_names) {
  extract_number <- function(row_name) {
    if (str_detect(row_name, "^M\\d+$")) {
      num <- as.numeric(str_extract(row_name, "\\d+"))
      return(ifelse(is.na(num), Inf, num))
    }
    return(Inf)
  }
  
  sort_keys <- sapply(row_names, extract_number)
  return(row_names[order(sort_keys)])
}