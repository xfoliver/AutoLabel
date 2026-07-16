"""
DLCV 推理接口客户端。

封装本地推理服务 http://127.0.0.1:9890 的调用，
支持模型加载、版本检查、单张与批量图片推理。

关键流程：
    1. GET /version 确认服务在线。
    2. POST /load_model {"model_path": "<.dvp>"} 加载模型。
    3. POST /api/inference {"model_path": "<.dvp>", "img": "<base64>"} 执行推理。

注意：
    - /api/inference 不会自动加载模型，必须先调用 /load_model。
    - 若未加载模型就调用 /api/inference，会返回：
      {"code": "00001", "message": "模型未加载"}
"""

import base64
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

VERSION_ENDPOINTS = (
    "/version",
    "/get_version",
    "/getVersion",
    "/api/version",
    "/api/get_version",
    "/api/getVersion",
    "/v1/version",
)


class InferenceError(Exception):
    """推理调用异常。"""

    pass


class InferenceClient:
    """DLCV 推理接口客户端。"""

    SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:9890",
        timeout: int = 60,
        max_retries: int = 2,
        retry_delay: float = 1.0,
        progress: bool = True,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.progress = progress
        self.session = requests.session()
        self._loaded_model_path: Optional[str] = None

    def get_version(self, timeout: float = 10.0) -> Dict[str, Any]:
        """调用真实版本接口确认推理服务在线且已 ready。"""
        total_start = time.time()
        last_error: Optional[Exception] = None
        last_status_code: Optional[int] = None

        for endpoint in VERSION_ENDPOINTS:
            url = f"{self.base_url}{endpoint}"
            try:
                start = time.time()
                response = self.session.get(url, timeout=timeout)
                last_status_code = response.status_code
                response.raise_for_status()
                elapsed = time.time() - start
                data = response.json()
                version = self._extract_version(data)

                logger.info(
                    "推理服务 get_version 成功: version=%s, 耗时=%.3fs, url=%s",
                    version,
                    elapsed,
                    url,
                )
                return {
                    "success": True,
                    "version": version,
                    "elapsed": elapsed,
                    "backend_url": self.base_url,
                    "reachable": True,
                    "raw": data,
                }
            except requests.exceptions.ConnectionError as e:
                last_error = e
                logger.error("推理服务未 ready: %s (url=%s)", e, url)
            except requests.exceptions.HTTPError as e:
                last_error = e
                last_status_code = e.response.status_code if e.response else None
                logger.debug("版本接口 %s 返回 HTTP %s", endpoint, last_status_code)
            except Exception as e:
                last_error = e
                logger.debug("版本接口 %s 不可用: %s", endpoint, e)

        total_elapsed = time.time() - total_start
        raise InferenceError(
            f"推理服务未 ready，请检查 backend / 模型测试 API "
            f"(url={self.base_url}, status={last_status_code}, error={last_error})"
        ) from last_error

    @staticmethod
    def _extract_version(data: Any) -> str:
        if not isinstance(data, dict):
            return str(data)
        version = data.get("version")
        if version is None:
            nested = data.get("data") or data.get("result") or {}
            version = nested.get("version") if isinstance(nested, dict) else nested
        return str(version) if version is not None else str(data)

    def load_model(
        self,
        model_path: Union[str, Path],
        force_reload: bool = False,
    ) -> Dict[str, Any]:
        """加载 .dvp 模型到推理服务。"""
        model_path = Path(model_path).resolve()
        if not model_path.exists():
            raise InferenceError(f"模型文件不存在: {model_path}")

        model_path_str = str(model_path)
        if not force_reload and self._loaded_model_path == model_path_str:
            logger.debug("模型已加载，跳过重复加载: %s", model_path.name)
            return {"code": "00000", "model_path": model_path_str}

        url = f"{self.base_url}/load_model"
        payload = {"model_path": model_path_str}

        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.post(url, json=payload, timeout=self.timeout)
                response.raise_for_status()
                result = response.json()

                if result.get("code") not in (None, "00000", "0"):
                    raise InferenceError(f"加载模型失败 [{model_path.name}]: {result}")

                self._loaded_model_path = model_path_str
                logger.info("模型加载成功: %s", model_path.name)
                return result
            except requests.exceptions.RequestException as e:
                logger.warning(
                    "模型加载请求失败 (%s/%s): %s - %s",
                    attempt + 1,
                    self.max_retries + 1,
                    model_path.name,
                    e,
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * (attempt + 1))

        raise InferenceError(f"加载模型失败 [{model_path.name}]")

    def infer(
        self,
        model_path: Union[str, Path],
        image_path: Union[str, Path],
    ) -> Dict[str, Any]:
        """对单张图片进行推理，返回服务端原始响应。"""
        model_path = Path(model_path)
        image_path = Path(image_path)

        if not model_path.exists():
            raise InferenceError(f"模型文件不存在: {model_path}")
        if not image_path.exists():
            raise InferenceError(f"图片不存在: {image_path}")

        self.load_model(model_path)

        img_base64 = self._encode_image(image_path)
        payload = {
            "model_path": str(model_path.resolve()),
            "img": img_base64,
        }

        url = f"{self.base_url}/api/inference"
        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.post(
                    url,
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                result = response.json()

                if result.get("code") == "00001" and "模型未加载" in str(
                    result.get("message", "")
                ):
                    logger.warning("模型在服务端被卸载，尝试重新加载: %s", model_path.name)
                    self.load_model(model_path, force_reload=True)
                    continue

                logger.debug("推理成功: %s -> %s", image_path.name, result)
                return result
            except requests.exceptions.Timeout as e:
                last_error = e
                logger.warning("推理超时 (%s/%s): %s", attempt + 1, self.max_retries + 1, image_path.name)
            except requests.exceptions.RequestException as e:
                last_error = e
                logger.warning("推理请求失败 (%s/%s): %s - %s", attempt + 1, self.max_retries + 1, image_path.name, e)

            if attempt < self.max_retries:
                time.sleep(self.retry_delay * (attempt + 1))

        raise InferenceError(f"推理失败 [{image_path.name}]: {last_error}")

    def infer_batch(
        self,
        model_path: Union[str, Path],
        image_dir: Union[str, Path],
        image_paths: Optional[List[Union[str, Path]]] = None,
        progress: Optional[bool] = None,
        on_progress: Optional[Callable[[int, int, str, str], None]] = None,
    ) -> List[Dict[str, Any]]:
        """对文件夹或指定图片列表批量推理。"""
        model_path = Path(model_path)

        # 批量推理前先统一加载一次模型，避免每张图片都重复加载
        self.load_model(model_path)

        if image_paths is None:
            image_dir = Path(image_dir)
            if not image_dir.is_dir():
                raise InferenceError(f"图片目录不存在: {image_dir}")
            image_paths = sorted(
                p for p in image_dir.rglob("*")
                if p.is_file() and p.suffix.lower() in self.SUPPORTED_IMAGE_SUFFIXES
            )
        else:
            image_paths = [Path(p) for p in image_paths]

        results: List[Dict[str, Any]] = []
        total = len(image_paths)

        show_progress = self.progress if progress is None else progress
        iterator = enumerate(image_paths, 1)

        use_tqdm = show_progress and on_progress is None and total > 0
        if use_tqdm:
            try:
                iterator = tqdm(
                    iterator,
                    total=total,
                    desc=f"推理 [{model_path.name}]",
                    unit="img",
                    ncols=80,
                    ascii=True,
                    disable=False,
                )
            except Exception:
                use_tqdm = False

        report_interval = max(1, total // 10) if total > 0 else 1
        for idx, img_path in iterator:
            should_notify = idx == 1 or idx == total or idx % report_interval == 0
            if on_progress is not None and should_notify:
                try:
                    on_progress(idx, total, img_path.name, model_path.name)
                except Exception:
                    pass
            elif not use_tqdm and should_notify:
                logger.info("推理进度: %s/%s - %s", idx, total, img_path.name)
            else:
                logger.debug("推理进度: %s/%s - %s", idx, total, img_path.name)
            try:
                result = self.infer(model_path, img_path)
                results.append({
                    "image": img_path.name,
                    "image_path": str(img_path),
                    "model_path": str(model_path),
                    "success": True,
                    "result": result,
                })
            except InferenceError as e:
                logger.error("推理失败: %s - %s", img_path.name, e)
                results.append({
                    "image": img_path.name,
                    "image_path": str(img_path),
                    "model_path": str(model_path),
                    "success": False,
                    "error": str(e),
                })

        return results

    @staticmethod
    def _encode_image(image_path: Path) -> str:
        """把图片文件编码为 base64 字符串。"""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
