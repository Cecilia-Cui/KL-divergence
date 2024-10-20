import subprocess
import time
import sys
import os

def run_fm_with_data(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # 创建唯一的输出目录，避免文件冲突
    output_dir = f"results_{int(time.time())}"
    os.makedirs(output_dir, exist_ok=True)

    for i, line in enumerate(lines):
        # 解析每一行的参数
        params = line.strip().split()
        Ra = params[0]
        Dec = params[1]
        iota = params[2]  # Inclination
        Pol = params[3]  # Polarization
        Dist = params[4]  # Luminosity Distance
        M1 = params[6]
        M2 = params[7]

        # 设置初始相位和injection参数
        Phi_c = 0.0  # 初始相位
        injection = 0  # 注入标识

        # 生成唯一的输出文件名
        output_filename = os.path.join(output_dir, f"output_{i}.pkl")

        # 打印调试信息，确保参数完整
        print(f"Running iteration {i}: Ra={Ra}, Dec={Dec}, M1={M1}, M2={M2}, "
              f"iota={iota}, Pol={Pol}, Dist={Dist}, Phi_c={Phi_c}, "
              f"injection={injection}, Output={output_filename}")

        # 构建命令并运行 fm.py
        command = [
            sys.executable, 'fm.py', Ra, Dec, M1, M2, iota, Pol, Dist,
            str(Phi_c), str(injection), '-O', output_filename, 
            '-d', 'ET_1', 'ET_2', 'ET_3'
        ]

        # 捕获错误并打印详细信息以帮助调试
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running command: {e}")
            sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_fm.py <data_file>")
        sys.exit(1)

    data_file = sys.argv[1]
    run_fm_with_data(data_file)
