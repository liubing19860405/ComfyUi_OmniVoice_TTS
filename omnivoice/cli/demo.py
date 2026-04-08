"""
Gradio demo for OmniVoice.

支持语音克隆和声音设计。

使用方法：
    omnivoice-demo --model /path/to/checkpoint --port 8000
"""

import argparse
import logging
from typing import Any, Dict

import gradio as gr
import numpy as np
import torch

from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.utils.lang_map import LANG_NAMES, lang_display_name


def get_best_device():
    """自动检测最优设备：CUDA > MPS > CPU。"""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# 语言列表 — 支持600多种语言
# ---------------------------------------------------------------------------
_ALL_LANGUAGES = ["自动"] + sorted(lang_display_name(n) for n in LANG_NAMES)


# ---------------------------------------------------------------------------
# 声音设计指令模板
# ---------------------------------------------------------------------------
_CATEGORIES = {
    "性别": ["男", "女"],
    "年龄": [
        "儿童",
        "少年",
        "青年",
        "中年",
        "老年",
    ],
    "音调": [
        "极低音调",
        "低音调",
        "中音调",
        "高音调",
        "极高音调",
    ],
    "风格": ["耳语"],
    "英文口音": [
        "美式口音",
        "澳大利亚口音",
        "英国口音",
        "中国口音",
        "加拿大口音",
        "印度口音",
        "韩国口音",
        "葡萄牙口音",
        "俄罗斯口音",
        "日本口音",
    ],
    "中文方言": [
        "河南话",
        "陕西话",
        "四川话",
        "贵州话",
        "云南话",
        "桂林话",
        "济南话",
        "石家庄话",
        "甘肃话",
        "宁夏话",
        "青岛话",
        "东北话",
    ],
}

_ATTR_INFO = {
    "英文口音": "仅对英语口语有效。",
    "中文方言": "仅对中文语音有效。",
}

# ---------------------------------------------------------------------------
# 命令行参数解析
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="omnivoice-demo",
        description="启动 OmniVoice 的 Gradio 演示。",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="k2-fsa/OmniVoice",
        help="模型路径或 HuggingFace 仓库 ID。",
    )
    parser.add_argument(
        "--device", default=None, help="使用的设备，未指定则自动检测。"
    )
    parser.add_argument("--ip", default="0.0.0.0", help="服务器IP (默认: 0.0.0.0)。")
    parser.add_argument(
        "--port", type=int, default=7860, help="服务器端口 (默认: 7860)。"
    )
    parser.add_argument(
        "--root-path",
        default=None,
        help="反向代理使用的根路径。",
    )
    parser.add_argument(
        "--share", action="store_true", default=False, help="创建公共链接。"
    )
    return parser


# ---------------------------------------------------------------------------
# 构建演示界面
# ---------------------------------------------------------------------------

def build_demo(
    model: OmniVoice,
    checkpoint: str,
    generate_fn=None,
) -> gr.Blocks:

    sampling_rate = model.sampling_rate

    # 生成核心函数
    def _gen_core(
        text,
        language,
        ref_audio,
        instruct,
        num_step,
        guidance_scale,
        denoise,
        speed,
        duration,
        preprocess_prompt,
        postprocess_output,
        mode,
        ref_text=None,
    ):
        if not text or not text.strip():
            return None, "请输入要合成的文本。"

        gen_config = OmniVoiceGenerationConfig(
            num_step=int(num_step or 32),
            guidance_scale=float(guidance_scale) if guidance_scale is not None else 2.0,
            denoise=bool(denoise) if denoise is not None else True,
            preprocess_prompt=bool(preprocess_prompt),
            postprocess_output=bool(postprocess_output),
        )

        lang = language if (language and language != "自动") else None

        kw: Dict[str, Any] = dict(
            text=text.strip(), language=lang, generation_config=gen_config
        )

        if speed is not None and float(speed) != 1.0:
            kw["speed"] = float(speed)
        if duration is not None and float(duration) > 0:
            kw["duration"] = float(duration)

        if mode == "clone":
            if not ref_audio:
                return None, "请上传参考音频。"
            kw["voice_clone_prompt"] = model.create_voice_clone_prompt(
                ref_audio=ref_audio,
                ref_text=ref_text,
            )

        if mode == "design":
            if instruct and instruct.strip():
                kw["instruct"] = instruct.strip()

        try:
            audio = model.generate(**kw)
        except Exception as e:
            return None, f"错误：{type(e).__name__}：{e}"

        waveform = audio[0].squeeze(0).numpy()  # (T,)
        waveform = (waveform * 32767).astype(np.int16)
        return (sampling_rate, waveform), "合成完成。"

    # 允许外部包装函数
    _gen = generate_fn if generate_fn is not None else _gen_core

    # =====================================================================
    # 界面样式
    # =====================================================================
    css = """
    .gradio-container {max-width: 100% !important; font-size: 16px !important;}
    .compact-audio audio {height: 60px !important;}
    .compact-audio .waveform {min-height: 80px !important;}
    """

    # 复用：语言下拉框
    def _lang_dropdown(label="语种（可选）", value="自动"):
        return gr.Dropdown(
            label=label,
            choices=_ALL_LANGUAGES,
            value=value,
            allow_custom_value=False,
            interactive=True,
            info="保持为自动以自动检测语言。",
        )

    # 复用：生成设置面板
    def _gen_settings():
        with gr.Accordion("生成设置（可选）", open=False):
            sp = gr.Slider(
                0.7,
                1.3,
                value=1.0,
                step=0.05,
                label="语速",
                info="1.0 = 正常。大于1更快，小于1更慢。若设置了时长则忽略此项。",
            )
            du = gr.Number(
                value=None,
                label="时长（秒）",
                info=(
                    "留空以使用语速。"
                    " 设置固定时长将覆盖语速。"
                ),
            )
            ns = gr.Slider(
                4,
                64,
                value=32,
                step=1,
                label="推理步数",
                info="默认值：32。数值越低速度越快，数值越高质量越好。",
            )
            dn = gr.Checkbox(
                label="降噪",
                value=True,
                info="默认：开启。取消勾选以关闭降噪。",
            )
            gs = gr.Slider(
                0.0,
                4.0,
                value=2.0,
                step=0.1,
                label="指导尺度（CFG）",
                info="默认：2.0。",
            )
            pp = gr.Checkbox(
                label="预处理提示",
                value=True,
                info="对参考音频去除静音并修剪，自动为参考文本添加末尾标点。",
            )
            po = gr.Checkbox(
                label="后处理输出",
                value=True,
                info="从生成的音频中去除长时间静音。",
            )
        return ns, gs, dn, sp, du, pp, po

    with gr.Blocks(css=css, title="OmniVoice 演示") as demo:
        gr.Markdown(
            """
<div style="text-align: center; font-size: 48px; font-weight: bold; margin: 20px 0;">
OmniVoice 语音克隆和声音设计纯净版
</div>

支持**600种语言**的先进文本转语音模型，包含12种以上中国方言，极速、高自然度、零样本语音克隆。

"""
        )

        with gr.Tabs():
            # ==============================================================
            # 语音克隆
            # ==============================================================
            with gr.TabItem("语音克隆"):
                with gr.Row():
                    with gr.Column(scale=1):
                        vc_text = gr.Textbox(
                            label="待合成文本",
                            lines=4,
                            placeholder="请输入需要合成的文本...",
                        )
                        vc_ref_audio = gr.Audio(
                            label="参考音频",
                            type="filepath",
                            elem_classes="compact-audio",
                        )
                        gr.Markdown(
                            "<span style='font-size:0.85em;color:#888;'>"
                            "建议：3–10秒音频，音频过长会占用GPU显存！"
                            "</span>"
                        )
                        vc_ref_text = gr.Textbox(
                            label="参考音频文本（可选）",
                            lines=2,
                            placeholder="参考音频的文字记录，留空将通过ASR模型自动转录。",
                        )
                        vc_lang = _lang_dropdown("语种（可选）")
                        (
                            vc_ns,
                            vc_gs,
                            vc_dn,
                            vc_sp,
                            vc_du,
                            vc_pp,
                            vc_po,
                        ) = _gen_settings()
                        vc_btn = gr.Button("生成", variant="primary")
                    with gr.Column(scale=1):
                        vc_audio = gr.Audio(
                            label="合成结果",
                            type="numpy",
                        )
                        vc_status = gr.Textbox(label="状态", lines=2)

                def _clone_fn(
                    text, lang, ref_aud, ref_text, ns, gs, dn, sp, du, pp, po
                ):
                    return _gen(
                        text,
                        lang,
                        ref_aud,
                        None,
                        ns,
                        gs,
                        dn,
                        sp,
                        du,
                        pp,
                        po,
                        mode="clone",
                        ref_text=ref_text or None,
                    )

                vc_btn.click(
                    _clone_fn,
                    inputs=[
                        vc_text,
                        vc_lang,
                        vc_ref_audio,
                        vc_ref_text,
                        vc_ns,
                        vc_gs,
                        vc_dn,
                        vc_sp,
                        vc_du,
                        vc_pp,
                        vc_po,
                    ],
                    outputs=[vc_audio, vc_status],
                )

            # ==============================================================
            # 声音设计
            # ==============================================================
            with gr.TabItem("声音设计"):
                with gr.Row():
                    with gr.Column(scale=1):
                        vd_text = gr.Textbox(
                            label="待合成文本",
                            lines=4,
                            placeholder="请输入需要合成的文本...",
                        )
                        vd_lang = _lang_dropdown()

                        _AUTO = "自动"
                        vd_groups = []
                        for _cat, _choices in _CATEGORIES.items():
                            vd_groups.append(
                                gr.Dropdown(
                                    label=_cat,
                                    choices=[_AUTO] + _choices,
                                    value=_AUTO,
                                    info=_ATTR_INFO.get(_cat),
                                )
                            )

                        (
                            vd_ns,
                            vd_gs,
                            vd_dn,
                            vd_sp,
                            vd_du,
                            vd_pp,
                            vd_po,
                        ) = _gen_settings()
                        vd_btn = gr.Button("生成", variant="primary")
                    with gr.Column(scale=1):
                        vd_audio = gr.Audio(
                            label="合成结果",
                            type="numpy",
                        )
                        vd_status = gr.Textbox(label="状态", lines=2)

                def _build_instruct(groups):
                    selected = [g for g in groups if g and g != "自动"]
                    if not selected:
                        return None
                    return "，".join(selected)

                def _design_fn(text, lang, ns, gs, dn, sp, du, pp, po, *groups):
                    return _gen(
                        text,
                        lang,
                        None,
                        _build_instruct(groups),
                        ns,
                        gs,
                        dn,
                        sp,
                        du,
                        pp,
                        po,
                        mode="design",
                    )

                vd_btn.click(
                    _design_fn,
                    inputs=[
                        vd_text,
                        vd_lang,
                        vd_ns,
                        vd_gs,
                        vd_dn,
                        vd_sp,
                        vd_du,
                        vd_pp,
                        vd_po,
                    ]
                    + vd_groups,
                    outputs=[vd_audio, vd_status],
                )

    return demo


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s：%(message)s",
    )
    parser = build_parser()
    args = parser.parse_args(argv)

    device = args.device or get_best_device()

    checkpoint = args.model
    if not checkpoint:
        parser.print_help()
        return 0
    logging.info(f"正在加载模型：{checkpoint}，设备：{device}...")
    model = OmniVoice.from_pretrained(
        checkpoint,
        device_map=device,
        dtype=torch.float16,
        load_asr=True,
    )
    print("模型加载完成。")

    demo = build_demo(model, checkpoint)

    demo.queue().launch(
        server_name=args.ip,
        server_port=args.port,
        share=args.share,
        root_path=args.root_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())