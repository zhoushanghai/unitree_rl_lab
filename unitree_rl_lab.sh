#!/usr/bin/env bash

# ===================== 核心修改：兼容 uv/venv/Conda 环境 =====================
# 定义脚本所在路径（保留官方逻辑）
export UNITREE_RL_LAB_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# 优先检测 uv/venv 环境（VIRTUAL_ENV 是 uv/venv 激活后自动设置的变量）
if ! [[ -z "${VIRTUAL_ENV}" ]]; then
    python_exe=${VIRTUAL_ENV}/bin/python
# 兼容 Conda 环境（保留原有逻辑，不影响 Conda 用户）
elif ! [[ -z "${CONDA_PREFIX}" ]]; then
    python_exe=${CONDA_PREFIX}/bin/python
# 都没检测到则报错退出
else
    echo "[Error] No virtual environment activated (uv/venv/Conda). Please activate first."
    echo "For uv environment: source /path/to/env_isaaclab/bin/activate"
    exit 1
fi
# ===================== 环境检测逻辑修改结束 =====================

# ===================== 保留官方自动补全功能 =====================
_ut_rl_lab_python_argcomplete_wrapper() {
    local IFS=$'\013'
    local SUPPRESS_SPACE=0
    if compopt +o nospace 2> /dev/null; then
        SUPPRESS_SPACE=1
    fi

    COMPREPLY=( $(IFS="$IFS" \
                    COMP_LINE="$COMP_LINE" \
                    COMP_POINT="$COMP_POINT" \
                    COMP_TYPE="$COMP_TYPE" \
                    _ARGCOMPLETE=1 \
                    _ARGCOMPLETE_SUPPRESS_SPACE=$SUPPRESS_SPACE \
                    ${python_exe} ${UNITREE_RL_LAB_PATH}/scripts/rsl_rl/train.py 8>&1 9>&2 1>/dev/null 2>/dev/null) )
}
complete -o nospace -F _ut_rl_lab_python_argcomplete_wrapper "./unitree_rl_lab.sh"
# ===================== 自动补全功能保留结束 =====================

# ===================== 移除 Conda 专属配置函数（uv 无需） =====================
# 空函数，替代原有的 Conda 配置逻辑，避免报错
_ut_setup_conda_env() {
    : # 空操作，不执行任何内容
}

# ===================== 保留官方命令分发逻辑 =====================
case "$1" in
    -i|--install)
        # 安装 git-lfs（确保大文件支持）
        if ! command -v git-lfs &> /dev/null; then
            echo "[Info] git-lfs not found, installing..."
            sudo apt update && sudo apt install -y git-lfs
        fi
        git lfs install
        
        # 可编辑模式安装 unitree_rl_lab（核心逻辑）
        pip install -e ${UNITREE_RL_LAB_PATH}/source/unitree_rl_lab/
        
        # 配置 argcomplete（自动补全）
        activate-global-python-argcomplete
        
        echo "[Success] unitree_rl_lab installed in editable mode (uv environment compatible)!"
        ;;
    -l|--list)
        shift
        ${python_exe} ${UNITREE_RL_LAB_PATH}/scripts/list_envs.py "$@"
        ;;
    -p|--play)
        shift
        ${python_exe} ${UNITREE_RL_LAB_PATH}/scripts/rsl_rl/play.py "$@"
        ;;
    -t|--train)
        shift
        ${python_exe} ${UNITREE_RL_LAB_PATH}/scripts/rsl_rl/train.py --headless "$@"
        ;;
    *) # 未知参数：提示帮助信息
        echo "Usage: ./unitree_rl_lab.sh [option] [args]"
        echo "Options:"
        echo "  -i/--install   Install unitree_rl_lab in editable mode (uv compatible)"
        echo "  -l/--list      List all available tasks"
        echo "  -p/--play      Inference with trained agent (--task <task-name>)"
        echo "  -t/--train     Train agent (--task <task-name>)"
        echo "Example:"
        echo "  ./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Velocity"
        ;;
esac
