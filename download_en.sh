#!/bin/bash

# 克隆Git仓库
git clone git@github.com:computationalstylistics/100_english_novels.git

# 定义源文件夹和目标文件夹
source_folder="100_english_novels"
destination_folder="data/general/en/"

# 创建目标文件夹（如果不存在的话）
mkdir -p "$destination_folder"

# 查找并移动所有txt文件
find "$source_folder" -type f -name "*.txt" -exec mv {} "$destination_folder" \;

# 删除源文件夹
rm -rf "$source_folder"

echo "所有txt文件已成功移动到 $destination_folder，并删除了源文件夹 $source_folder"