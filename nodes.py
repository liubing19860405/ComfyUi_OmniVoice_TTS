import torch
import numpy as np
import scipy.io.wavfile as wav
import os
import time
from omnivoice import OmniVoice, OmniVoiceGenerationConfig

def _to_audio_tuple(audio):
    """将ComfyUI的AUDIO输入转换为(audio_tensor, sample_rate)元组"""
    if audio is None:
        raise ValueError("参考音频输入为空。")
    if not isinstance(audio, dict):
        raise TypeError("预期ComfyUI AUDIO输入为字典格式。")
    if "waveform" not in audio or "sample_rate" not in audio:
        raise KeyError("AUDIO输入必须包含'waveform'和'sample_rate'。")

    waveform = audio["waveform"]
    sample_rate = int(audio["sample_rate"])

    if not isinstance(waveform, torch.Tensor):
        waveform = torch.tensor(waveform)

    waveform = waveform.detach().cpu()

    # 确保波形形状为 [batch, channels, samples] 或 [channels, samples]
    if waveform.dim() == 3:
        # 如果是 [batch, channels, samples]，取第一个批次
        waveform = waveform[0]
    elif waveform.dim() == 2:
        # 如果是 [channels, samples]，保持不变
        pass
    elif waveform.dim() == 1:
        # 如果是 [samples]，添加通道维度
        waveform = waveform.unsqueeze(0)
    else:
        raise ValueError(
            f"不支持的AUDIO波形形状 {tuple(waveform.shape)}。 "
            "预期为 [batch, channels, samples], [channels, samples], 或 [samples]。"
        )

    return waveform.float(), sample_rate


def _to_comfy_audio(waveform, sample_rate):
    """将生成的音频转换回ComfyUI的AUDIO格式"""
    waveform = waveform.detach().cpu().float()
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.dim() != 2:
        raise ValueError(
            f"生成的波形必须具有形状 [channels, samples]，得到 {tuple(waveform.shape)}。"
        )
    return {
        "waveform": waveform.unsqueeze(0),
        "sample_rate": int(sample_rate),
    }


def _clean_optional_text(value):
    """清理可选文本"""
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def _resolve_optional_duration(duration):
    """处理可选的持续时间"""
    return float(duration) if duration and float(duration) > 0 else None


def _resolve_optional_speed(speed):
    """处理可选的速度"""
    speed = float(speed)
    if speed <= 0:
        raise ValueError("速度必须大于0。")
    if abs(speed - 1.0) < 1e-6:
        return None
    return speed


class OmniVoiceModelLoader:
    def __init__(self):
        self.model = None
        self.sampling_rate = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {"default": "D:/ComfyUI/ComfyUI/models/OmniVoice", "multiline": False}),
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("OMNIVOICE_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "OmniVoice"

    def load_model(self, model_path, keep_model_loaded):
        print(f"正在加载模型从路径: {model_path}")
        
        # 设置环境变量
        os.environ["OMNIVOICE_MODEL"] = model_path
        
        # 加载模型
        model = OmniVoice.from_pretrained(
            model_path,
            device_map="cuda",
            dtype=torch.float16,
            load_asr=False  # 不加载ASR模型
        )
        
        sampling_rate = model.sampling_rate
        print("模型加载成功!")
        
        return ((model, sampling_rate, keep_model_loaded),)


class OmniVoiceClone:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_data": ("OMNIVOICE_MODEL",),
                "text": ("STRING", {"multiline": True, "default": "你好，这是一个测试。"}),
                "ref_audio": ("AUDIO",),
                "language": (["Auto", "en", "zh", "ja", "ko", "fr", "es", "de"], {"default": "Auto"}),
                "num_step": ("INT", {"default": 32, "min": 1, "max": 128}),
                "guidance_scale": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 20.0}),
                "denoise": ("BOOLEAN", {"default": True}),
                "preprocess_prompt": ("BOOLEAN", {"default": True}),
                "postprocess_output": ("BOOLEAN", {"default": True}),
                "duration": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 30.0}),
                "t_shift": ("FLOAT", {"default": 0.10, "min": 0.05, "max": 4.0}),
                "layer_penalty_factor": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 20.0}),
                "position_temperature": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 20.0}),
                "class_temperature": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 20.0}),
                "audio_chunk_threshold": ("FLOAT", {"default": 30.0, "min": 0.1, "max": 600.0}),
            },
            "optional": {
                "ref_text": ("STRING", {"multiline": True}),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.05, "max": 8.0}),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "status_message")
    FUNCTION = "generate_clone"
    CATEGORY = "OmniVoice"

    def generate_clone(self, model_data, text, ref_audio, language, num_step, guidance_scale, denoise, 
                      preprocess_prompt, postprocess_output, duration, t_shift, layer_penalty_factor, 
                      position_temperature, class_temperature, audio_chunk_threshold, 
                      ref_text=None, speed=1.0):
        
        model, sampling_rate, keep_model_loaded = model_data

        if not text or not text.strip():
            # 创建一个空的音频对象作为错误处理
            empty_waveform = torch.zeros((1, 100))  # 1 channel, 100 samples
            empty_audio = _to_comfy_audio(empty_waveform, sampling_rate)
            return (empty_audio, "请输入要合成的文本。")

        if ref_audio is None:
            # 创建一个空的音频对象作为错误处理
            empty_waveform = torch.zeros((1, 100))  # 1 channel, 100 samples
            empty_audio = _to_comfy_audio(empty_waveform, sampling_rate)
            return (empty_audio, "请提供参考音频用于声音克隆。")

        gen_config = OmniVoiceGenerationConfig(
            num_step=int(num_step),
            guidance_scale=float(guidance_scale),
            t_shift=float(t_shift),
            layer_penalty_factor=float(layer_penalty_factor),
            position_temperature=float(position_temperature),
            class_temperature=float(class_temperature),
            denoise=bool(denoise),
            preprocess_prompt=bool(preprocess_prompt),
            postprocess_output=bool(postprocess_output),
            audio_chunk_threshold=float(audio_chunk_threshold),
        )

        lang = language if (language and language != "Auto") else None

        kw = {
            "text": text.strip(),
            "language": lang,
            "generation_config": gen_config
        }

        duration_value = _resolve_optional_duration(duration)
        speed_value = None if duration_value is not None else _resolve_optional_speed(speed)

        if speed_value is not None:
            kw["speed"] = speed_value
        if duration_value is not None:
            kw["duration"] = duration_value

        # 转换音频格式为OmniVoice接受的格式
        try:
            ref_audio_tensor, ref_sample_rate = _to_audio_tuple(ref_audio)
        except Exception as e:
            # 创建一个空的音频对象作为错误处理
            empty_waveform = torch.zeros((1, 100))  # 1 channel, 100 samples
            empty_audio = _to_comfy_audio(empty_waveform, sampling_rate)
            return (empty_audio, f"音频格式错误: {str(e)}")
        
        # 确保音频是单声道
        if ref_audio_tensor.shape[0] > 1:
            ref_audio_tensor = ref_audio_tensor[0:1, :]  # 取第一个声道
        elif ref_audio_tensor.dim() == 1:
            ref_audio_tensor = ref_audio_tensor.unsqueeze(0)

        try:
            kw["voice_clone_prompt"] = model.create_voice_clone_prompt(
                ref_audio=(ref_audio_tensor, ref_sample_rate),
                ref_text=ref_text
            )
        except Exception as e:
            # 创建一个空的音频对象作为错误处理
            empty_waveform = torch.zeros((1, 100))  # 1 channel, 100 samples
            empty_audio = _to_comfy_audio(empty_waveform, sampling_rate)
            return (empty_audio, f"创建声音克隆提示失败: {str(e)}")

        # 执行音频生成
        try:
            audio = model.generate(**kw)
            if audio is None or len(audio) == 0:
                # 创建一个空的音频对象作为错误处理
                empty_waveform = torch.zeros((1, 100))  # 1 channel, 100 samples
                empty_audio = _to_comfy_audio(empty_waveform, sampling_rate)
                return (empty_audio, "生成失败：返回了空音频")
                
            waveform = audio[0].squeeze(0).cpu()
            
            # 检查生成的音频是否为空
            if waveform.numel() == 0:
                # 创建一个空的音频对象作为错误处理
                empty_waveform = torch.zeros((1, 100))  # 1 channel, 100 samples
                empty_audio = _to_comfy_audio(empty_waveform, sampling_rate)
                return (empty_audio, "生成失败：音频长度为0")
            
            # 保存音频文件
            output_dir = "output"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(output_dir, f"omnivoice_clone_{timestamp}.wav")
            waveform_np = waveform.numpy()
            waveform_int16 = (waveform_np * 32767).astype(np.int16)
            wav.write(file_path, sampling_rate, waveform_int16)
            
            # 返回音频数据和状态信息
            audio_data = _to_comfy_audio(waveform, sampling_rate)
            
            return (audio_data, f"音频已生成并保存至: {file_path}")
            
        except Exception as e:
            # 创建一个空的音频对象作为错误处理
            empty_waveform = torch.zeros((1, 100))  # 1 channel, 100 samples
            empty_audio = _to_comfy_audio(empty_waveform, sampling_rate)
            error_msg = f"生成失败: {type(e).__name__}: {str(e)}"
            return (empty_audio, error_msg)


class OmniVoiceDesign:
    @classmethod
    def INPUT_TYPES(cls):
        # 由于无法导入OmniVoice内部模块，我们使用通用的语音设计参数
        return {
            "required": {
                "model_data": ("OMNIVOICE_MODEL",),
                "text": ("STRING", {"multiline": True, "default": "你好，这是一个测试。"}),
                "language": (["Auto", "en", "zh", "ja", "ko", "fr", "es", "de"], {"default": "Auto"}),
                "num_step": ("INT", {"default": 32, "min": 1, "max": 128}),
                "guidance_scale": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 20.0}),
                "denoise": ("BOOLEAN", {"default": True}),
                "preprocess_prompt": ("BOOLEAN", {"default": True}),
                "postprocess_output": ("BOOLEAN", {"default": True}),
                "duration": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 30.0}),
                "t_shift": ("FLOAT", {"default": 0.10, "min": 0.05, "max": 4.0}),
                "layer_penalty_factor": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 20.0}),
                "position_temperature": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 20.0}),
                "class_temperature": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 20.0}),
                "audio_chunk_threshold": ("FLOAT", {"default": 30.0, "min": 0.1, "max": 600.0}),
                "gender": (["none", "male", "female"], {"default": "none"}),
                "age": (["none", "child", "teenager", "young adult", "middle-aged", "elderly"], {"default": "none"}),
                "pitch": (["none", "very low pitch", "low pitch", "moderate pitch", "high pitch", "very high pitch"], {"default": "none"}),
                "style": (["none", "whisper"], {"default": "none"}),
                "accent": (["none", "american", "british", "australian", "indian", "chinese", "japanese", "korean"], {"default": "none"}),
                "dialect": (["none", "henan", "shaanxi", "sichuan", "guizhou", "yunnan", "guilin", "jinan", "shijiazhuang", "gansu", "ningxia", "qingdao", "northeast"], {"default": "none"}),
            },
            "optional": {
                "custom_attributes": ("STRING", {"multiline": True}),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.05, "max": 8.0}),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "status_message")
    FUNCTION = "generate_design"
    CATEGORY = "OmniVoice"

    def generate_design(self, model_data, text, language, num_step, guidance_scale, denoise, 
                       preprocess_prompt, postprocess_output, duration, t_shift, layer_penalty_factor, 
                       position_temperature, class_temperature, audio_chunk_threshold,
                       gender, age, pitch, style, accent, dialect,
                       custom_attributes="", speed=1.0):
        
        model, sampling_rate, keep_model_loaded = model_data

        if not text or not text.strip():
            # 创建一个空的音频对象作为错误处理
            empty_waveform = torch.zeros((1, 100))  # 1 channel, 100 samples
            empty_audio = _to_comfy_audio(empty_waveform, sampling_rate)
            return (empty_audio, "请输入要合成的文本。")

        gen_config = OmniVoiceGenerationConfig(
            num_step=int(num_step),
            guidance_scale=float(guidance_scale),
            t_shift=float(t_shift),
            layer_penalty_factor=float(layer_penalty_factor),
            position_temperature=float(position_temperature),
            class_temperature=float(class_temperature),
            denoise=bool(denoise),
            preprocess_prompt=bool(preprocess_prompt),
            postprocess_output=bool(postprocess_output),
            audio_chunk_threshold=float(audio_chunk_threshold),
        )

        lang = language if (language and language != "Auto") else None

        # 构建声音描述
        voice_parts = []
        
        if gender != "none":
            voice_parts.append(gender)
        if age != "none":
            voice_parts.append(age)
        if pitch != "none":
            voice_parts.append(pitch)
        if style != "none":
            voice_parts.append(style)
        if accent != "none":
            voice_parts.append(f"{accent} accent")
        if dialect != "none":
            voice_parts.append(f"{dialect} dialect")
        
        # 添加自定义属性
        if custom_attributes.strip():
            custom_parts = [
                item.strip()
                for item in custom_attributes.replace("\n", ",").split(",")
                if item.strip()
            ]
            voice_parts.extend(custom_parts)

        # 构建instruct字符串，如果没有任何描述则设为None
        instruct = ", ".join(voice_parts) if voice_parts else None

        kw = {
            "text": text.strip(),
            "language": lang,
            "generation_config": gen_config,
        }
        
        if instruct:
            kw["instruct"] = instruct

        duration_value = _resolve_optional_duration(duration)
        speed_value = None if duration_value is not None else _resolve_optional_speed(speed)

        if speed_value is not None:
            kw["speed"] = speed_value
        if duration_value is not None:
            kw["duration"] = duration_value

        # 执行音频生成
        try:
            audio = model.generate(**kw)
            if audio is None or len(audio) == 0:
                # 创建一个空的音频对象作为错误处理
                empty_waveform = torch.zeros((1, 100))  # 1 channel, 100 samples
                empty_audio = _to_comfy_audio(empty_waveform, sampling_rate)
                return (empty_audio, "生成失败：返回了空音频")
                
            waveform = audio[0].squeeze(0).cpu()
            
            # 检查生成的音频是否为空
            if waveform.numel() == 0:
                # 创建一个空的音频对象作为错误处理
                empty_waveform = torch.zeros((1, 100))  # 1 channel, 100 samples
                empty_audio = _to_comfy_audio(empty_waveform, sampling_rate)
                return (empty_audio, "生成失败：音频长度为0")
            
            # 保存音频文件
            output_dir = "output"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(output_dir, f"omnivoice_design_{timestamp}.wav")
            waveform_np = waveform.numpy()
            waveform_int16 = (waveform_np * 32767).astype(np.int16)
            wav.write(file_path, sampling_rate, waveform_int16)
            
            # 返回音频数据和状态信息
            audio_data = _to_comfy_audio(waveform, sampling_rate)
            
            return (audio_data, f"音频已生成并保存至: {file_path}")
            
        except Exception as e:
            # 创建一个空的音频对象作为错误处理
            empty_waveform = torch.zeros((1, 100))  # 1 channel, 100 samples
            empty_audio = _to_comfy_audio(empty_waveform, sampling_rate)
            error_msg = f"生成失败: {type(e).__name__}: {str(e)}"
            return (empty_audio, error_msg)


# 注册节点
NODE_CLASS_MAPPINGS = {
    "OmniVoiceModelLoader": OmniVoiceModelLoader,
    "OmniVoiceClone": OmniVoiceClone,
    "OmniVoiceDesign": OmniVoiceDesign
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OmniVoiceModelLoader": "OmniVoice 模型加载器",
    "OmniVoiceClone": "OmniVoice 声音克隆",
    "OmniVoiceDesign": "OmniVoice 声音设计"
}