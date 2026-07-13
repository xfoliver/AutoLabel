"""
DLCV AI 训练平台自动化入口 3.0（Agent 层）

一句话用法：
    python auto_train3.0.py --task OCR --latest

功能：
1. 扫描 AI训练平台测试 目录，按任务类型匹配文件夹。
2. 自动挑选最新（或指定）的数据集文件夹。
3. 自动探测数据集路径（训练集 / 训练 / train / dataset）。
4. 自动生成 work 目录和 .dva 文件路径。
5. 根据任务类型加载对应模板配置，生成 config。
6. 调用 auto_train2.0.py 完成：生成/更新 .dva -> 启动平台 -> WebBridge 点击训练。

命令行示例：
    # 训练最新 OCR 模型
    python auto_train3.0.py --task OCR --latest
    # 默认已启用 --reuse --keep-alive

    # 训练指定文件夹
    python auto_train3.0.py --task OCR --folder 1-OCR

    # 只生成配置，不启动训练
    python auto_train3.0.py --task OCR --latest --dry-run

    # 复用已运行的平台实例
    python auto_train3.0.py --task OCR --latest --reuse
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 在 Windows Git Bash / CMD / PowerShell 中保持中文日志不乱码
if sys.platform == "win32" and sys.stdout.encoding not in ("utf-8", "UTF-8"):
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")
    except Exception:
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class AutoTrainAgentError(Exception):
    """Agent 层自定义异常。"""
    pass


# ---------------------------------------------------------------------------
# 任务类型映射
# ---------------------------------------------------------------------------
# 用户可能说的别名 -> 内部规范任务类型
TASK_ALIASES: Dict[str, str] = {
    # OCR
    "ocr": "OCR",
    "OCR": "OCR",
    # 分类
    "cls": "分类",
    "CLS": "分类",
    "分类": "分类",
    "classification": "分类",
    # 目标检测
    "det": "目标检测",
    "DET": "目标检测",
    "目标检测": "目标检测",
    "detection": "目标检测",
    # DETR（底层任务类型仍用 目标检测，但模板用 DETR 专用）
    "detr": "DETR",
    "DETR": "DETR",
    # 实例分割
    "iseg": "实例分割",
    "ISEG": "实例分割",
    "实例分割": "实例分割",
    "instance": "实例分割",
    # 语义分割
    "sseg": "语义分割",
    "SSEG": "语义分割",
    "语义分割": "语义分割",
    "semantic": "语义分割",
    # 旋转框检测
    "rotate": "旋转框检测",
    "ROTATE": "旋转框检测",
    "旋转框检测": "旋转框检测",
    "rotation": "旋转框检测",
}

# 规范任务类型 -> 模板配置文件
TASK_TEMPLATES: Dict[str, str] = {
    "OCR": "config/ocr_train_automation_config.json",
    "分类": "config/cls_train_automation_config.json",
    "目标检测": "config/det_train_automation_config.json",
    "DETR": "config/detr_train_automation_config.json",
    "实例分割": "config/iseg_train_automation_config.json",
    "语义分割": "config/sseg_train_automation_config.json",
    "旋转框检测": "config/rotate_train_automation_config.json",
}

# 用于匹配文件夹名的关键字（不区分大小写）
TASK_FOLDER_KEYWORDS: Dict[str, List[str]] = {
    "OCR": ["ocr"],
    "分类": ["cls", "分类", "classification"],
    "目标检测": ["det", "目标检测", "detection"],
    "DETR": ["detr"],
    "实例分割": ["iseg", "实例分割", "instance"],
    "语义分割": ["sseg", "语义分割", "semantic"],
    "旋转框检测": ["rotate", "旋转框检测", "rotation"],
}

# 数据集子目录探测优先级
DATASET_CANDIDATES = ["训练集", "1-训练集", "训练", "train", "dataset"]

# 默认 WebBridge 配置（Playwright，推荐）
DEFAULT_WEBBRIDGE = {
    "enabled": True,
    "adapter": "playwright",
    "frontend_url": "http://localhost:8843",
    "timeout": 120,
    "frontend_ready_timeout": 60,
    "frontend_ready_interval": 1,
    "screenshot_dir": "./webbridge_screenshots",
    "headless": False,
    "browser": "chromium",
    "slow_mo": 50,
    "keep_browser_open": True,
}


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
class AutoTrainAgent:
    """DLCV 训练自动化 Agent。"""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.project_root = Path(__file__).parent.resolve()
        self.scan_dir: Path = self._resolve_path(args.scan_dir or "AI训练平台测试")
        self.task_type: str = self._resolve_task_type(args.task)
        self.template: Dict[str, Any] = {}
        self.chosen_folder: Optional[Path] = None
        self.dataset_dir: Optional[Path] = None
        self.work_dir: Optional[Path] = None
        self.dva_path: Optional[Path] = None
        self.config_path: Optional[Path] = None
        self.project_name: str = ""

    # ------------------------------------------------------------------
    # 路径/类型工具
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_path(raw: str) -> Path:
        """解析相对路径，支持 %VAR% 环境变量。"""
        expanded = os.path.expandvars(raw)
        p = Path(expanded)
        if not p.is_absolute():
            p = (Path(__file__).parent / p).resolve()
        return p

    @staticmethod
    def _resolve_task_type(task_arg: str) -> str:
        """把用户输入的任务别名转成规范任务类型。"""
        if task_arg in TASK_ALIASES:
            return TASK_ALIASES[task_arg]
        # 尝试直接匹配规范名
        for canonical in TASK_TEMPLATES.keys():
            if task_arg.lower() == canonical.lower():
                return canonical
        raise AutoTrainAgentError(
            f"不支持的任务类型: {task_arg}。支持: {', '.join(sorted(set(TASK_ALIASES.keys())))}"
        )

    @staticmethod
    def _sanitize_name(name: str) -> str:
        """把文件夹名转成合法的项目名（去掉首尾空格，替换特殊字符）。"""
        name = name.strip()
        name = re.sub(r"[\\/:*?\"<>|]+", "-", name)
        name = re.sub(r"\s+", "_", name)
        return name or "Project"

    # ------------------------------------------------------------------
    # 文件夹扫描与选择
    # ------------------------------------------------------------------
    def _match_task_folder(self, folder: Path) -> bool:
        """判断文件夹名是否匹配当前任务类型。"""
        folder_lower = folder.name.lower()
        keywords = TASK_FOLDER_KEYWORDS.get(self.task_type, [self.task_type.lower()])
        for kw in keywords:
            if kw.lower() in folder_lower:
                return True
        return False

    def _score_folder_by_date(self, folder: Path) -> Tuple[int, int, int, int]:
        """
        尝试从文件夹名解析日期，用于辅助排序。
        返回 (year, month, day, hour)，解析不到则全 0。
        支持：OCR-0618、2026-OCR-Test、OCR-20260625 等。
        """
        name = folder.name
        # 优先找 8 位日期 20260625
        m = re.search(r"(20\d{2})(\d{2})(\d{2})", name)
        if m:
            return (int(m.group(1)), int(m.group(2)), int(m.group(3)), 0)
        # 其次找 MMDD，如 OCR-0618
        m = re.search(r"(\d{2})(\d{2})", name)
        if m:
            month, day = int(m.group(1)), int(m.group(2))
            if 1 <= month <= 12 and 1 <= day <= 31:
                return (datetime.now().year, month, day, 0)
        return (0, 0, 0, 0)

    def _select_latest_folder(self, candidates: List[Path]) -> Path:
        """在候选文件夹中选最新：优先 mtime，mtime 相同时按名字里的日期排。"""
        if not candidates:
            raise AutoTrainAgentError(f"未找到任务类型 {self.task_type} 的文件夹")

        def sort_key(p: Path) -> Tuple[float, Tuple[int, int, int, int]]:
            return (p.stat().st_mtime, self._score_folder_by_date(p))

        candidates_sorted = sorted(candidates, key=sort_key, reverse=True)
        latest = candidates_sorted[0]
        logger.info(
            "在 %s 个候选中选中最新文件夹: %s (mtime: %s)",
            len(candidates),
            latest.name,
            datetime.fromtimestamp(latest.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
        )
        return latest

    def find_folder(self) -> Path:
        """找到要训练的数据集文件夹。"""
        if self.args.folder:
            explicit = self.scan_dir / self.args.folder
            if not explicit.exists():
                raise AutoTrainAgentError(f"指定文件夹不存在: {explicit}")
            if not explicit.is_dir():
                raise AutoTrainAgentError(f"指定路径不是目录: {explicit}")
            logger.info("使用指定文件夹: %s", explicit.name)
            return explicit

        if not self.scan_dir.exists():
            raise AutoTrainAgentError(f"扫描目录不存在: {self.scan_dir}")

        candidates = [p for p in self.scan_dir.iterdir() if p.is_dir() and self._match_task_folder(p)]
        if not candidates:
            raise AutoTrainAgentError(
                f"在 {self.scan_dir} 下未找到任务 {self.task_type} 的文件夹"
            )

        if not self.args.latest and len(candidates) > 1:
            names = [p.name for p in candidates]
            raise AutoTrainAgentError(
                f"找到多个匹配文件夹，请用 --folder 指定或 --latest 选最新: {names}"
            )

        return self._select_latest_folder(candidates)

    # ------------------------------------------------------------------
    # 数据集路径探测
    # ------------------------------------------------------------------
    def find_dataset_dir(self, folder: Path) -> Path:
        """按优先级探测数据集子目录。"""
        for candidate in DATASET_CANDIDATES:
            cand_path = folder / candidate
            if cand_path.exists() and cand_path.is_dir():
                logger.info("数据集路径: %s", cand_path)
                return cand_path
        raise AutoTrainAgentError(
            f"在 {folder} 下未找到数据集目录，期望存在: {DATASET_CANDIDATES}"
        )

    # ------------------------------------------------------------------
    # 平台 exe 自动探测
    # ------------------------------------------------------------------
    @staticmethod
    def _detect_platform_exe(template_path: Optional[str] = None) -> Optional[str]:
        """
        自动探测平台可执行文件路径。
        优先级：
        1. 环境变量 DLCV_TRAIN_EXE
        2. 模板中配置的路径（若存在）
        3. 常见安装路径 win_x64 / win_64 两种命名
        """
        env_exe = os.environ.get("DLCV_TRAIN_EXE")
        if env_exe:
            expanded = os.path.expandvars(env_exe)
            if Path(expanded).exists():
                return expanded
            logger.warning("环境变量 DLCV_TRAIN_EXE 指向的文件不存在: %s", expanded)

        if template_path:
            expanded = os.path.expandvars(template_path)
            if Path(expanded).exists():
                return expanded

        base_dirs = [
            "C:/dlcv/Lib/site-packages/dlcv_train",
            "C:/Program Files/dlcv_train",
            "C:/Program Files (x86)/dlcv_train",
        ]
        names = [
            "DLCV AI Training Platform-win_x64.exe",
            "DLCV AI Training Platform-win_64.exe",
            "DLCV AI Training Platform.exe",
        ]
        for base in base_dirs:
            for name in names:
                candidate = Path(base) / name
                if candidate.exists():
                    return str(candidate)
        return None

    # ------------------------------------------------------------------
    # 配置模板加载
    # ------------------------------------------------------------------
    def load_template(self) -> None:
        """加载任务类型对应的模板配置。"""
        template_file = TASK_TEMPLATES.get(self.task_type)
        if not template_file:
            raise AutoTrainAgentError(f"未找到任务类型 {self.task_type} 的模板配置")

        template_path = self.project_root / template_file
        if not template_path.exists():
            raise AutoTrainAgentError(f"模板配置文件不存在: {template_path}")

        with open(template_path, "r", encoding="utf-8") as f:
            self.template = json.load(f)
        logger.info("已加载任务模板: %s", template_path)

    # ------------------------------------------------------------------
    # 生成项目名 / 路径
    # ------------------------------------------------------------------
    def build_project_paths(self, folder: Path) -> None:
        """生成 project_name、dataset_dir、work_dir、dva_file。"""
        folder_name = self._sanitize_name(folder.name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.project_name = f"{folder_name}-{timestamp}"

        self.dataset_dir = self.find_dataset_dir(folder)
        self.work_dir = folder / "work"
        self.work_dir.mkdir(parents=True, exist_ok=True)

        dva_name = f"{self.project_name}.dva"
        self.dva_path = folder / dva_name

        logger.info("项目名: %s", self.project_name)
        logger.info("数据集: %s", self.dataset_dir)
        logger.info("工作目录: %s", self.work_dir)
        logger.info("DVA 文件: %s", self.dva_path)

    # ------------------------------------------------------------------
    # 生成最终 config
    # ------------------------------------------------------------------
    def build_config(self) -> Dict[str, Any]:
        """基于模板生成最终配置。"""
        cfg = json.loads(json.dumps(self.template))  # 深拷贝

        # 覆盖 project 区块
        project = cfg.setdefault("project", {})
        project["name"] = self.project_name
        # DETR 模板里 task_type 是 目标检测，保持即可；其它模板本身已正确
        project["dataset_dir"] = str(self.dataset_dir.resolve())
        project["work_dir"] = str(self.work_dir.resolve())
        project["dva_file"] = str(self.dva_path.resolve())
        project.setdefault("description", f"自动化 {self.task_type} 训练项目")

        # 平台 exe 路径：命令行 > 自动探测 > 模板原值
        platform = cfg.setdefault("platform", {})
        if self.args.exe:
            platform["executable_path"] = os.path.expandvars(self.args.exe)
        else:
            detected = self._detect_platform_exe(platform.get("executable_path"))
            if detected:
                if detected != platform.get("executable_path"):
                    logger.info("自动修正平台 exe 路径: %s", detected)
                    platform["executable_path"] = detected
            else:
                logger.warning(
                    "未探测到平台可执行文件，将使用模板路径: %s",
                    platform.get("executable_path"),
                )

        # 对齐 auto_train2.0 工作配置的默认值：给后端更充裕的初始化时间
        platform["backend_ready_timeout"] = 300
        platform["backend_ready_interval"] = 2.0

        # 如果没有 webbridge 区块，注入默认 Playwright 配置
        if "webbridge" not in cfg:
            cfg["webbridge"] = dict(DEFAULT_WEBBRIDGE)

        # 允许命令行覆盖 adapter / headless
        wb = cfg["webbridge"]
        if self.args.adapter:
            wb["adapter"] = self.args.adapter
        if self.args.headless is not None:
            wb["headless"] = self.args.headless

        return cfg

    def save_config(self, cfg: Dict[str, Any]) -> Path:
        """保存生成的配置文件。"""
        if self.args.config_out:
            config_path = self._resolve_path(self.args.config_out)
        else:
            config_dir = self.project_root / "config"
            config_dir.mkdir(parents=True, exist_ok=True)
            # 每种任务类型固定一个文件，每次运行覆盖，避免 config/ 目录不断膨胀
            safe_task = re.sub(r"[^\w\-]+", "_", self.task_type)
            config_path = config_dir / f"auto_latest_{safe_task}.json"

        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        logger.info("已生成配置文件: %s", config_path)
        return config_path

    # ------------------------------------------------------------------
    # 调用 auto_train2.0.py
    # ------------------------------------------------------------------
    def run_auto_train(self, config_path: Path) -> int:
        """调用 auto_train2.0.py 执行训练。"""
        cmd = [
            sys.executable,
            str(self.project_root / "auto_train2.0.py"),
            "--config",
            str(config_path),
        ]
        if self.args.force_new:
            cmd.append("--force-new")
        if self.args.dry_run:
            cmd.append("--dry-run")
        if self.args.keep_alive:
            cmd.append("--keep-alive")
        if self.args.keep_browser_open is False:
            cmd.append("--no-keep-browser-open")
        if self.args.reuse:
            cmd.append("--reuse")

        logger.info("========== AUTO TRAIN 3.0 AGENT ==========")
        logger.info("启动子进程: %s", " ".join(cmd))

        # 直接继承当前标准输出，让用户看到 auto_train2.0 的日志
        result = subprocess.run(cmd, check=False)
        return result.returncode

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------
    def run(self) -> int:
        """执行完整 Agent 流程。"""
        logger.info("任务类型: %s", self.task_type)
        logger.info("扫描目录: %s", self.scan_dir)

        self.load_template()
        self.chosen_folder = self.find_folder()
        self.build_project_paths(self.chosen_folder)
        cfg = self.build_config()
        self.config_path = self.save_config(cfg)

        if self.args.generate_config_only:
            logger.info("--generate-config-only: 已生成配置，不启动训练")
            return 0

        return self.run_auto_train(self.config_path)


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="DLCV AI 训练平台自动化入口 3.0：一句话启动训练"
    )
    parser.add_argument(
        "--task",
        required=True,
        help="任务类型，例如 OCR、CLS、DET、DETR、ISEG、SSEG、ROTATE 或中文名",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="自动选择扫描目录下最新（修改时间）的匹配文件夹",
    )
    parser.add_argument(
        "--folder",
        help="直接指定数据集文件夹名（相对于 --scan-dir）",
    )
    parser.add_argument(
        "--scan-dir",
        default="AI训练平台测试",
        help="数据集根目录，默认: AI训练平台测试",
    )
    parser.add_argument(
        "--config-out",
        help="生成的配置文件保存路径",
    )
    parser.add_argument(
        "--exe",
        help="覆盖平台可执行文件路径（支持 %%VAR%% 环境变量）",
    )
    parser.add_argument(
        "--adapter",
        choices=["playwright", "skill", "http", "stub"],
        help="覆盖 WebBridge adapter 类型",
    )
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        help="是否无头运行浏览器（仅对 playwright 有效）",
    )
    parser.add_argument(
        "--generate-config-only",
        action="store_true",
        help="仅生成配置文件，不调用 auto_train2.0.py",
    )
    parser.add_argument(
        "--force-new",
        action="store_true",
        help="强制创建新项目（传给 auto_train2.0.py）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅生成 .dva，不启动平台（传给 auto_train2.0.py）",
    )
    parser.add_argument(
        "--keep-alive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="训练结束后保持平台运行（默认开启，传给 auto_train2.0.py）",
    )
    parser.add_argument(
        "--keep-browser-open",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="训练结束后保持浏览器窗口打开，按 Enter 关闭（默认开启，传给 auto_train2.0.py）",
    )
    parser.add_argument(
        "--reuse",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="复用已运行的平台/后端实例（默认开启，传给 auto_train2.0.py）",
    )
    args = parser.parse_args()

    try:
        agent = AutoTrainAgent(args)
        return agent.run()
    except AutoTrainAgentError as e:
        logger.error("Agent 流程失败: %s", e)
        return 1
    except KeyboardInterrupt:
        logger.warning("用户中断")
        return 130


if __name__ == "__main__":
    sys.exit(main())
