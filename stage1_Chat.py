import argparse
import os
from typing import List

import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from data_provider.tokenization_utils import batch_tokenize_messages_list
from models.mol_llama import MolLLaMA
from utils.configuration_mol_llama import MolLLaMAConfig


DEFAULT_TRAIN_CONFIG = "/home/haoqian/Data/Molecule/Latent/configs/stage1/train_config.yaml"
DEFAULT_STAGE1_CKPT_DIR = "/home/haoqian/Data/Molecule/Latent/checkpoints/stage1"
DEFAULT_STAGE1_CKPT_PATH = "/home/haoqian/Data/Molecule/Latent/checkpoints/stage1/20260322_002518-naive-lr-4slots-dorfrz77/naive-lr-4slots-stepstep=00044036.ckpt"


def _pick_checkpoint(ckpt_path: str, ckpt_dir: str) -> str:
    if ckpt_path and os.path.exists(ckpt_path):
        return ckpt_path
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")
    cands = []
    for name in os.listdir(ckpt_dir):
        if name.endswith(".ckpt"):
            cands.append(os.path.join(ckpt_dir, name))
    if not cands:
        raise FileNotFoundError(f"No .ckpt files under {ckpt_dir}")
    cands.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return cands[0]


def _precision_to_dtype(precision: str):
    p = str(precision).lower()
    if "bf16" in p:
        return "bfloat16"
    if p in {"16", "fp16", "float16"}:
        return "float16"
    return "float32"


def _resolve_device(device_arg: str) -> str:
    if device_arg:
        return device_arg
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def _build_tokenizer(llm_model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
    mol_ids = tokenizer("<mol>", add_special_tokens=False).input_ids
    if len(mol_ids) != 1:
        tokenizer.add_special_tokens({"additional_special_tokens": ["<mol>"]})
        mol_ids = tokenizer("<mol>", add_special_tokens=False).input_ids
    tokenizer.mol_token_id = mol_ids[0]
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _parse_smiles(text: str) -> List[str]:
    parts = [x.strip() for x in text.split(",")]
    return [p for p in parts if p]


def _normalize_user_text(text: str) -> str:
    # Terminal inputs often contain escaped '\n' from copy/paste.
    return text.replace("\\n", "\n").strip()


def _prepare_text_batch(tokenizer, llama_version: str, user_prompt: str, use_smiles: bool, device: str):
    mol_prompt = "<mol><mol><mol><mol><mol><mol><mol><mol>"
    if use_smiles:
        # Expand every <mol> placeholder to match training-time 8 query slots.
        if "<mol>" in user_prompt:
            user_prompt = user_prompt.replace("<mol>", mol_prompt)
        else:
            user_prompt = f"{mol_prompt}\n{user_prompt}"
    messages = [[{"role": "user", "content": user_prompt}]]
    text_batch = batch_tokenize_messages_list(messages, tokenizer, llama_version, padding_side="left")
    return text_batch.to(device)


def _decode_new_tokens(tokenizer, outputs, text_batch):
    if outputs is None or outputs.numel() == 0:
        return ""
    prompt_len = int(text_batch.input_ids.shape[1])
    if outputs.shape[1] > prompt_len:
        gen_ids = outputs[:, prompt_len:]
    else:
        gen_ids = outputs
    text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
    if not text:
        text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()
    return text


def main():
    parser = argparse.ArgumentParser(description="Stage-I interactive chat")
    parser.add_argument("--train_config", default=DEFAULT_TRAIN_CONFIG)
    parser.add_argument("--checkpoint_path", default=DEFAULT_STAGE1_CKPT_PATH, help="Specific .ckpt path (default: last-v3)")
    parser.add_argument("--checkpoint_dir", default=DEFAULT_STAGE1_CKPT_DIR, help="Auto-pick latest .ckpt from this directory")
    parser.add_argument("--device", default="", help="e.g. cuda:0 or cpu")
    parser.add_argument("--precision", default="", help="Override precision (bf16-mixed/16/32)")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--min_new_tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--do_sample", action="store_true", help="Enable sampling; default uses greedy decoding")
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--num_latent_steps", type=int, default=4)
    parser.add_argument("--disable_latent_reasoning", action="store_true", help="Use plain Mol-LLaMA generation path")
    args = parser.parse_args()

    train_cfg = OmegaConf.load(args.train_config)
    model_cfg = getattr(train_cfg, "model", {})
    llm_model_name = str(getattr(model_cfg, "llm_model", "meta-llama/Llama-3.1-8B-Instruct"))
    precision = args.precision or str(getattr(train_cfg, "precision", "bf16-mixed"))
    torch_dtype = _precision_to_dtype(precision)
    device = _resolve_device(args.device)
    llama_version = "llama3" if "Llama-3" in llm_model_name else "llama2"

    ckpt_path = _pick_checkpoint(args.checkpoint_path, args.checkpoint_dir)
    print(f"[StageI Chat] loading checkpoint: {ckpt_path}")
    print(f"[StageI Chat] model={llm_model_name}, device={device}, precision={torch_dtype}")

    tokenizer = _build_tokenizer(llm_model_name)
    config = MolLLaMAConfig()
    config.llm_config.llm_model = llm_model_name
    config.graph_encoder_config.encoder_types = ["unimol", "moleculestm"]

    model = MolLLaMA(
        config=config,
        vocab_size=len(tokenizer),
        torch_dtype=torch_dtype,
        enable_flash=bool(getattr(train_cfg, "enable_flash", False)),
        use_latent_reasoning=not bool(args.disable_latent_reasoning),
        num_latent_steps=int(args.num_latent_steps),
    ).to(device)
    model.eval()
    model.load_from_ckpt(ckpt_path)

    if llama_version == "llama3":
        eos_token_id = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    else:
        eos_token_id = tokenizer.eos_token_id

    print("\n[StageI Chat] Ready. 输入 exit/quit 结束。\n")
    while True:
        user_text = _normalize_user_text(input("你: "))
        if user_text.lower() in {"exit", "quit"}:
            print("已退出。")
            break
        if not user_text:
            continue

        need_smiles = input("是否需要输入分子smiles? (y/n): ").strip().lower()
        smiles_list = []
        use_smiles = need_smiles in {"y", "yes", "1", "true"}
        if use_smiles:
            smiles_line = input("请输入SMILES（多个用逗号分隔）: ").strip()
            smiles_list = _parse_smiles(smiles_line)
            if not smiles_list:
                print("未检测到有效 SMILES，本轮按纯文本对话处理。")
                use_smiles = False

        text_batch = _prepare_text_batch(
            tokenizer=tokenizer,
            llama_version=llama_version,
            user_prompt=user_text,
            use_smiles=use_smiles,
            device=device,
        )

        with torch.no_grad():
            if use_smiles and len(smiles_list) > 0:
                outputs = model.generate_with_smiles(
                    smiles_list=smiles_list,
                    text_batch=text_batch,
                    do_sample=bool(args.do_sample),
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_new_tokens=args.max_new_tokens,
                    min_new_tokens=args.min_new_tokens,
                    repetition_penalty=args.repetition_penalty,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=eos_token_id,
                )
            else:
                outputs = model.generate(
                    graph_batch=None,
                    text_batch=text_batch,
                    do_sample=bool(args.do_sample),
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_new_tokens=args.max_new_tokens,
                    min_new_tokens=args.min_new_tokens,
                    repetition_penalty=args.repetition_penalty,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=eos_token_id,
                )

        text = _decode_new_tokens(tokenizer, outputs, text_batch)
        print(f"模型: {text}\n")


if __name__ == "__main__":
    main()
