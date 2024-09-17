import subprocess

# 打开生成的参数文件
with open('Sub_ET_1_ET_2_ET_3_40Mpc.txt', 'r') as f:
    lines = f.readlines()

# 遍历每一行
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

    # initial phase 和 injection_number 固定为 0
    initial_phase = 0
    injection_number = 0

    # 构建命令来运行 fm.py
    command = [
        'python', 'fm.py', Ra, Dec, M1, M2, iota, Pol, Dist,
        str(initial_phase), str(injection_number), '-O', f'output_{i}', '-d', 'ET_1', 'ET_2', 'ET_3'
    ]

    # 运行命令
    subprocess.run(command)
