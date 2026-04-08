import os
import sys
import time
import logging
import threading
import numpy as np
import scipy.io.wavfile as wav
from typing import Any, Dict, Tuple, Optional, List

# ===================== 全局配置（匹配你的参考代码） =====================
logger = logging.getLogger(__name__)
_MODEL_CACHE: Dict[Tuple[str, str, str], Any] = {}
_MODEL_CACHE_LOCK = threading.Lock()
_DEFAULT_OMNIVOICE_DIR = "models/OmniVoice"
_SUPPORTED_DEVICES = ["auto", "cuda", "cpu"]
_SUPPORTED_DTYPES = ["float16", "float32"]
_SUPPORTED_LANGUAGES = ["Auto", "zh", "en", "ja", "ko"]
_OUTPUT_DIR = "output/omnivoice"

# ===================== 懒加载依赖（匹配你的参考代码） =====================
def _import_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch 未安装！请在 ComfyUI 环境中安装 torch 和 torchaudio。"
        ) from exc
    return torch

def _import_omnivoice():
    try:
        from omnivoice import OmniVoice, OmniVoiceGenerationConfig
    except ImportError as exc:
        raise ImportError(
            f"OmniVoice 导入失败！错误信息：{exc}"
        ) from exc
    return OmniVoice, OmniVoiceGenerationConfig

def _get_folder_paths_module():
    try:
        import folder_paths
    except ImportError:
        return None
    return folder_paths

# ===================== 路径/设备/精度解析（匹配你的参考代码） =====================
def _resolve_model_path(model_path: str) -> str:
    model_path = model_path.strip().replace("\\", "/")
    if os.path.isabs(model_path):
        return model_path

    folder_paths = _get_folder_paths_module()
    if folder_paths and hasattr(folder_paths, "models_dir"):
        comfy_model_path = os.path.join(folder_paths.models_dir, model_path)
        if os.path.exists(comfy_model_path):
            return os.path.abspath(comfy_model_path)

    abs_path = os.path.abspath(model_path)
    if os.path.exists(abs_path):
        return abs_path

    default_path = os.path.abspath(_DEFAULT_OMNIVOICE_DIR)
    if os.path.exists(default_path):
        logger.warning(f"模型路径不存在，使用默认路径：{default_path}")
        return default_path

    raise FileNotFoundError(f"模型路径不存在：{model_path}")

def _resolve_device(device: str) -> str:
    torch = _import_torch()
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device not in _SUPPORTED_DEVICES:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device

def _resolve_dtype(dtype_str: str) -> Any:
    torch = _import_torch()
    if dtype_str == "float16":
        return torch.float16
    elif dtype_str == "float32":
        return torch.float32
    return torch.float16

# ===================== 标准化音频处理（匹配你的参考代码） =====================
def _to_audio_tuple(audio: Dict[str, Any]) -> Tuple[Any, int]:
    if audio is None:
        raise ValueError("参考音频输入为空！")
    if not isinstance(audio, dict):
        raise TypeError(f"音频必须为字典类型，当前类型：{type(audio)}")
    if "waveform" not in audio or "sample_rate" not in audio:
        raise KeyError("音频必须包含 waveform 和 sample_rate")

    waveform = audio["waveform"]
    sample_rate = int(audio["sample_rate"])

    torch = _import_torch()
    if not isinstance(waveform, torch.Tensor):
        waveform = torch.tensor(np.array(waveform), dtype=torch.float32)

    waveform = waveform.detach().cpu()

    if waveform.dim() == 3:
        waveform = waveform[0]
    elif waveform.dim() == 2:
        waveform = waveform[0:1]
    elif waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    else:
        raise ValueError(f"不支持的音频维度：{waveform.dim()}")

    return waveform.float(), sample_rate

def _save_audio(waveform: np.ndarray, sampling_rate: int, prefix: str) -> str:
    os.makedirs(_OUTPUT_DIR, exist_ok=True)
    file_name = f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}.wav"
    file_path = os.path.join(_OUTPUT_DIR, file_name)

    try:
        waveform = waveform / np.max(np.abs(waveform)) if np.max(np.abs(waveform)) > 0 else waveform
        wav.write(file_path, sampling_rate, (waveform * 32767).astype(np.int16))
        logger.info(f"音频已保存：{file_path}")
        return file_path
    except Exception as exc:
        raise IOError(f"保存音频失败：{exc}") from exc

# ===================== 1. 模型加载节点（中文界面 + 缓存） =====================
class OmniVoiceModelLoader:
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "模型路径": ("STRING", {
                    "default": _DEFAULT_OMNIVOICE_DIR,
                    "multiline": False,
                    "placeholder": "输入 OmniVoice 模型目录"
                }),
                "计算设备": (_SUPPORTED_DEVICES, {"default": "auto"}),
                "精度": (_SUPPORTED_DTYPES, {"default": "float16"}),
            }
        }

    RETURN_TYPES = ("OMNIVOICE_MODEL",)
    RETURN_NAMES = ("OmniVoice模型",)
    FUNCTION = "load_model"
    CATEGORY = "OmniVoice TTS"

    def load_model(self, 模型路径: str, 计算设备: str, 精度: str) -> Tuple[Any]:
        resolved_model_path = _resolve_model_path(模型路径)
        resolved_device = _resolve_device(计算设备)
        resolved_dtype = _resolve_dtype(精度)
        cache_key = (resolved_model_path, resolved_device, 精度)

        with _MODEL_CACHE_LOCK:
            if cache_key in _MODEL_CACHE:
                logger.info("从缓存加载模型")
                return (_MODEL_CACHE[cache_key],)

        OmniVoice, _ = _import_omnivoice()
        model = OmniVoice.from_pretrained(
            resolved_model_path,
            device_map=resolved_device,
            dtype=resolved_dtype,
        )

        with _MODEL_CACHE_LOCK:
            _MODEL_CACHE[cache_key] = model

        logger.info(f"模型加载完成 | 采样率：{model.sampling_rate}")
        return (model,)

# ===================== 2. 声音设计节点（匹配你的参考代码） =====================
class OmniVoiceVoiceDesign:
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "模型": ("OMNIVOICE_MODEL",),
                "合成文本": ("STRING", {
                    "default": "你好，这是 OmniVoice 声音设计",
                    "multiline": True
                }),
                "语言": (_SUPPORTED_LANGUAGES, {"default": "Auto"}),
            },
            "optional": {
                "生成步数": ("INT", {"default": 32, "min": 1, "max": 128}),
                "指导尺度": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 10.0}),
                "去噪": ("BOOLEAN", {"default": True}),
                "语速": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0}),
                "音频时长": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 60.0}),
                "声音设计指令": ("STRING", {"default": "female, low pitch", "multiline": True}),
                "运行后卸载模型": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("音频数据",)
    FUNCTION = "generate"
    CATEGORY = "OmniVoice TTS"

    def generate(
        self, 模型: Any, 合成文本: str, 语言: str,
        生成步数=32, 指导尺度=2.0, 去噪=True, 语速=1.0, 音频时长=0.0,
        声音设计指令="", 运行后卸载模型=False
    ):
        合成文本 = 合成文本.strip()
        if not 合成文本:
            raise ValueError("合成文本不能为空！")

        _, OmniVoiceGenerationConfig = _import_omnivoice()
        gen_config = OmniVoiceGenerationConfig(
            num_step=生成步数, guidance_scale=指导尺度, denoise=去噪
        )

        gen_kwargs = {
            "text": 合成文本,
            "language": 语言 if 语言 != "Auto" else None,
            "generation_config": gen_config
        }
        if 语速 != 1.0: gen_kwargs["speed"] = 语速
        if 音频时长 > 0: gen_kwargs["duration"] = 音频时长
        if 声音设计指令.strip(): gen_kwargs["instruct"] = 声音设计指令.strip()

        audio = 模型.generate(**gen_kwargs)
        torch = _import_torch()
        waveform = torch.squeeze(audio[0]).cpu().numpy()
        _save_audio(waveform, 模型.sampling_rate, "design")

        # 转换为ComfyUI期望的音频格式
        waveform_tensor = torch.from_numpy(waveform).float()
        # 确保维度为 [batch, channels, samples]
        if waveform_tensor.dim() == 1:
            waveform_tensor = waveform_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, samples]
        elif waveform_tensor.dim() == 2:
            waveform_tensor = waveform_tensor.unsqueeze(0)  # [1, channels, samples]

        audio_output = {
            "waveform": waveform_tensor,
            "sample_rate": 模型.sampling_rate
        }

        # 显存管理
        if 运行后卸载模型:
            with _MODEL_CACHE_LOCK:
                for k in list(_MODEL_CACHE.keys()):
                    if _MODEL_CACHE[k] == 模型: del _MODEL_CACHE[k]
            torch.cuda.empty_cache()

        return (audio_output,)

# ===================== 3. 声音克隆节点（修复错误） =====================
class OmniVoiceVoiceClone:
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "模型": ("OMNIVOICE_MODEL",),
                "合成文本": ("STRING", {"multiline": True}),
                "参考音频": ("AUDIO",),
                "参考文本": ("STRING", {"multiline": True}),
                "语言": (_SUPPORTED_LANGUAGES, {"default": "Auto"}),
            },
            "optional": {
                "生成步数": ("INT", {"default": 32}),
                "指导尺度": ("FLOAT", {"default": 2.0}),
                "语速": ("FLOAT", {"default": 1.0}),
                "运行后卸载模型": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("音频数据",)
    FUNCTION = "generate"
    CATEGORY = "OmniVoice TTS"

    def generate(
        self, 模型: Any, 合成文本: str, 参考音频: dict, 参考文本: str, 语言: str,
        生成步数=32, 指导尺度=2.0, 语速=1.0, 运行后卸载模型=False
    ):
        合成文本 = 合成文本.strip()
        if not 合成文本:
            raise ValueError("合成文本不能为空！")
        
        # 正确提取参考音频的波形和采样率
        ref_waveform = 参考音频["waveform"]
        ref_sample_rate = int(参考音频["sample_rate"])
        
        # 确保波形是正确的张量格式
        torch = _import_torch()
        if not isinstance(ref_waveform, torch.Tensor):
            ref_waveform = torch.tensor(ref_waveform.numpy())
        
        # 确保波形维度正确
        if ref_waveform.dim() == 3:
            ref_waveform = ref_waveform[0]
        elif ref_waveform.dim() == 2:
            ref_waveform = ref_waveform[0]  # 取第一个声道
        
        # 将参考音频移到模型设备上
        ref_waveform = ref_waveform.to(模型.device)

        _, OmniVoiceGenerationConfig = _import_omnivoice()
        gen_config = OmniVoiceGenerationConfig(num_step=生成步数, guidance_scale=指导尺度)

        # 修复声音克隆方法调用
        try:
            # 正确格式化参考音频参数
            ref_audio_tuple = (ref_waveform, ref_sample_rate)
            clone_prompt = 模型.create_voice_clone_prompt(
                ref_audio=ref_audio_tuple, 
                ref_text=参考文本 or ""
            )
        except Exception as e:
            raise RuntimeError(f"创建声音克隆提示失败: {str(e)}")

        # 生成音频
        try:
            audio = 模型.generate(
                text=合成文本,
                language=语言 if 语言 != "Auto" else None,
                voice_clone_prompt=clone_prompt,
                speed=语速,
                generation_config=gen_config
            )
        except Exception as e:
            raise RuntimeError(f"生成音频失败: {str(e)}")

        # 处理生成的音频
        waveform = torch.squeeze(audio[0]).cpu().numpy()
        _save_audio(waveform, 模型.sampling_rate, "clone")

        # 转换为ComfyUI期望的音频格式
        waveform_tensor = torch.from_numpy(waveform).float()
        # 确保维度为 [batch, channels, samples]
        if waveform_tensor.dim() == 1:
            waveform_tensor = waveform_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, samples]
        elif waveform_tensor.dim() == 2:
            waveform_tensor = waveform_tensor.unsqueeze(0)  # [1, channels, samples]

        audio_output = {
            "waveform": waveform_tensor,
            "sample_rate": 模型.sampling_rate
        }

        if 运行后卸载模型:
            with _MODEL_CACHE_LOCK:
                for k in list(_MODEL_CACHE.keys()):
                    if _MODEL_CACHE[k] == 模型: del _MODEL_CACHE[k]
            torch.cuda.empty_cache()

        return (audio_output,)

# ===================== 节点注册（完全匹配你的参考代码） =====================
NODE_CLASS_MAPPINGS = {
    "OmniVoiceModelLoader": OmniVoiceModelLoader,
    "OmniVoiceVoiceDesign": OmniVoiceVoiceDesign,
    "OmniVoiceVoiceClone": OmniVoiceVoiceClone
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OmniVoiceModelLoader": "OmniVoice 模型加载",
    "OmniVoiceVoiceDesign": "OmniVoice 声音设计",
    "OmniVoiceVoiceClone": "OmniVoice 声音克隆"
}

# 初始化模型文件夹
def _init():
    folder_paths = _get_folder_paths_module()
    if folder_paths:
        try:
            folder_paths.add_model_folder_path("omnivoice", _DEFAULT_OMNIVOICE_DIR)
        except:
            pass
_init()
