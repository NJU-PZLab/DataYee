import os
import shutil
import subprocess
import sys


env = sys.argv[1] if len(sys.argv) > 1 else "datayee"


def find_conda():
    """自动查找 conda 可执行文件路径"""
    candidates = [
        os.environ.get("CONDA_EXE", ""),
        shutil.which("conda") or "",
        shutil.which("conda.bat") or "",
        r"G:\Anaconda\Scripts\conda.exe",
        r"G:\Anaconda\condabin\conda.bat",
        r"C:\Users\admin\anaconda3\Scripts\conda.exe",
        r"C:\Users\admin\miniconda3\Scripts\conda.exe",
        r"C:\ProgramData\anaconda3\Scripts\conda.exe",
    ]
    for c in candidates:
        if c and os.path.isfile(c):
            return c
    return "conda"


CONDA = find_conda()
print(f"使用 conda: {CONDA}")


def run_cmd(cmd, critical=False):
    """运行命令，优先使用参数列表，避免 shell 解析版本号。"""
    if isinstance(cmd, list):
        printable = " ".join(f'"{x}"' if " " in str(x) else str(x) for x in cmd)
    else:
        printable = str(cmd)
    print(f">>> {printable}")
    ret = subprocess.run(cmd, shell=isinstance(cmd, str)).returncode
    if ret != 0:
        msg = f"  FAILED (code={ret})"
        if critical:
            print(msg)
            sys.exit(ret)
        print(msg)
    return ret


def conda(*args):
    return [CONDA] + list(args)


def pip_run(env_name, *args, critical=False):
    """在 conda 环境里执行一次 python -m pip 命令。"""
    cmd = conda("run", "-n", env_name, "python", "-m", "pip", *args)
    return run_cmd(cmd, critical=critical)


def env_exists(env_name):
    result = subprocess.run(
        conda("env", "list"),
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and any(
        line.split()[0] == env_name for line in result.stdout.splitlines() if line.strip() and not line.startswith("#")
    )


print("\n===== 1. 创建 conda 环境 =====")
if env_exists(env):
    print(f"环境 {env} 已存在，跳过创建")
else:
    run_cmd(conda("create", "-y", "-n", env, "python=3.12", "pip", "--solver", "classic"), critical=True)


print("\n===== 2. 安装本地 DataYee =====")
pip_args = [
    "install",
    "--trusted-host",
    "pypi.org",
    "--trusted-host",
    "files.pythonhosted.org",
    "--extra-index-url",
    "https://download.pytorch.org/whl/cpu",
    ".",
]

pip_run(env, *pip_args, critical=True)


print(f"\n{'=' * 50}")
print("安装完成!")
print(f"运行: conda activate {env}")
print("      python UI.py")
print(f"{'=' * 50}")
