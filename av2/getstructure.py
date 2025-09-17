#!/usr/bin/env python3
# filepath: /home/lzq/workspace/gan_seg/av2/getstructure.py

import subprocess
import sys
from pathlib import Path

def print_directory_structure(s3_path, prefix="", max_depth=5, current_depth=0):
    """
    递归打印S3目录结构
    """
    if current_depth >= max_depth:
        print(f"{prefix}... (已达到最大深度 {max_depth})")
        return
    
    try:
        # 使用s5cmd列出目录结构
        cmd = ["s5cmd", "--no-sign-request", "ls", s3_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            
            dirs = []
            files = []
            
            # 分离目录和文件
            for line in lines:
                if not line.strip():
                    continue
                    
                parts = line.split()
                if len(parts) < 2:
                    continue
                
                if parts[0] == "DIR":
                    dirs.append(parts[-1].rstrip('/'))
                else:
                    if len(parts) >= 4:
                        file_name = parts[-1]
                        file_size = parts[-2]
                        files.append((file_name, file_size))
                    else:
                        files.append((parts[-1], "unknown"))
            
            # 打印文件
            for i, (file_name, file_size) in enumerate(files):
                is_last_file = (i == len(files) - 1) and len(dirs) == 0
                connector = "└── " if is_last_file else "├── "
                print(f"{prefix}{connector}📄 {file_name} ({file_size} bytes)")
            
            # 打印并递归进入目录
            for i, dir_name in enumerate(dirs):
                is_last = i == len(dirs) - 1
                connector = "└── " if is_last else "├── "
                print(f"{prefix}{connector}📁 {dir_name}/")
                
                # 递归进入子目录
                sub_path = s3_path + dir_name + "/"
                new_prefix = prefix + ("    " if is_last else "│   ")
                print_directory_structure(sub_path, new_prefix, max_depth, current_depth + 1)
                        
        else:
            print(f"{prefix}└── ❌ 错误: {result.stderr.strip()}")
            
    except subprocess.TimeoutExpired:
        print(f"{prefix}└── ⏰ 超时")
    except Exception as e:
        print(f"{prefix}└── ❌ 错误: {e}")

def main():
    if len(sys.argv) > 1:
        s3_path = sys.argv[1]
    else:
        s3_path = "s3://argoverse/datasets/av2/sensor/train/00a6ffc1-6ce9-3bc3-a060-6006e9893a1a/"
    
    if not s3_path:
        print("错误: 请提供S3路径")
        return
    
    # 确保路径以/结尾
    if not s3_path.endswith('/'):
        s3_path += '/'
    
    # 设置最大递归深度
    max_depth = 5
    if len(sys.argv) > 2:
        try:
            max_depth = int(sys.argv[2])
        except ValueError:
            print("警告: 无效的深度参数，使用默认值 5")
    
    print(f"递归目录结构: {s3_path}")
    print(f"最大深度: {max_depth}")
    print("=" * 80)
    print_directory_structure(s3_path, max_depth=max_depth)

if __name__ == "__main__":
    main()