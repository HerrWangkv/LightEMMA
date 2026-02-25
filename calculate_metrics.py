import os
import json
import numpy as np
import glob

def summarize_final_comparison(root_output_dir):
    # 1. 识别包含数据的有效文件夹
    subfolders = [f.path for f in os.scandir(root_output_dir) if f.is_dir()]
    valid_folders = []
    for folder in sorted(subfolders):
        if glob.glob(os.path.join(folder, "scene_*.json")):
            valid_folders.append(folder)
    
    if not valid_folders:
        print(f"未在 {root_output_dir} 下发现有效实验数据。")
        return

    # 表格显示的指标 vs 底部统计的指标
    table_metrics = ["ADE_1s", "FDE"]
    all_metrics = ["ADE_1s", "ADE_2s", "ADE_3s", "ADE_avg", "FDE"]
    folder_names = [os.path.basename(f) for f in valid_folders]
    
    # 2. 统计并对齐场景：仅保留所有文件夹中都存在的共同场景 (Common Scenes)
    common_scenes = None
    for folder in valid_folders:
        files = glob.glob(os.path.join(folder, "scene_*.json"))
        scenes_in_folder = set([os.path.basename(f).replace(".json", "") for f in files])
        
        if common_scenes is None:
            common_scenes = scenes_in_folder
        else:
            common_scenes &= scenes_in_folder  # 取交集

    sorted_scenes = sorted(list(common_scenes))
    
    if not sorted_scenes:
        print("警告：各文件夹之间没有共同场景，无法进行公平对比。")
        return

    print(f"检测到 {len(sorted_scenes)} 个共同场景，开始进行公平对比评估...\n")

    # 3. 打印主对比表格 (仅显示 ADE_1s 和 FDE)
    col_width = 12
    scene_col_width = 20
    group_width = (col_width + 3) * len(valid_folders) - 1
    
    header_row1 = f"{'Scene / Metric':<{scene_col_width}} |"
    for m in table_metrics:
        header_row1 += f" {m:^{group_width}} |"
    
    header_row2 = f"{'':<{scene_col_width}} |"
    for m in table_metrics:
        for name in folder_names:
            short_name = (name[:col_width-2] + "..") if len(name) > col_width else name
            header_row2 += f" {short_name:<{col_width}} |"

    print("\n" + "=" * len(header_row1))
    print("CORE METRIC SIDE-BY-SIDE COMPARISON")
    print("=" * len(header_row1))
    print(header_row1)
    print(header_row2)
    print("-" * len(header_row1))

    # 初始化用于最终统计的容器
    folder_stats = {f: {m: [] for m in all_metrics} for f in valid_folders}

    for scene in sorted_scenes:
        row = f"{scene:<{scene_col_width}} |"
        for m in table_metrics:
            for folder in valid_folders:
                file_path = os.path.join(folder, f"{scene}.json")
                val_str = "N/A"
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        frames = data.get("frames", [])
                        # 获取该场景下该指标的平均值
                        m_vals = [fr["metrics"][m] for fr in frames if fr.get("metrics") and fr["metrics"].get(m) is not None]
                        if m_vals:
                            avg_val = np.mean(m_vals)
                            val_str = f"{avg_val:.3f}"
                            
                row += f" {val_str:<{col_width}} |"
        
        # 收集所有四个指标的数据
        for folder in valid_folders:
            file_path = os.path.join(folder, f"{scene}.json")
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    frames = data.get("frames", [])
                    for m_key in all_metrics:
                        m_vals = [fr["metrics"][m_key] for fr in frames if fr.get("metrics") and fr["metrics"].get(m_key) is not None]
                        if m_vals:
                            folder_stats[folder][m_key].append(np.mean(m_vals))

        print(row)
    print("=" * len(header_row1))

    # 4. 底部输出：每个文件夹的所有 4 个指标总平均值
    print("\nFULL EXPERIMENT SUMMARY (Global Averages)")
    print("-" * 100)
    print(f"{'Folder Name':<35} | {'ADE_1s':<10} | {'ADE_2s':<10} | {'ADE_3s':<10} | {'ADE_avg':<10}| {'FDE':<10}")
    print("-" * 100)
    for folder in valid_folders:
        name = os.path.basename(folder)
        summary_row = f"{name[:35]:<35} |"
        for m_key in all_metrics:
            vals = folder_stats[folder][m_key]
            avg = f"{np.mean(vals):.4f}" if vals else "N/A"
            summary_row += f" {avg:<10} |"
        print(summary_row)
    print("-" * 100)

if __name__ == "__main__":
    # 默认路径为当前目录下的 output 文件夹
    summarize_final_comparison("./output")