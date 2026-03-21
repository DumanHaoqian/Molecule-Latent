import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, Optional

import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from transformers import AutoTokenizer

from data_provider.tokenization_utils import batch_tokenize_messages_list
from models.mol_llama import MolLLaMA

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


DEFAULT_MODEL_PATH = "/home/haoqian/Data/Molecule/Latent/checkpoints/hf/Mol-Llama-3.1-8B-Instruct"
DEFAULT_TEST_DIR = "/home/haoqian/Data/Molecule/Datasets/Test/01_mol_edit"
DEFAULT_OUTPUT_DIR = "/home/haoqian/Data/Molecule/Latent/eval_outputs/moledit_molllama"

MOLEDIT_TASK_FILES = {
    "add": "add.json",
    "delete": "delete.json",
    "sub": "sub.json",
}

GROUP_TO_SMARTS = {
    "benzene": "[cR1]1[cR1][cR1][cR1][cR1][cR1]1",
    "benzene_ring": "[cR1]1[cR1][cR1][cR1][cR1][cR1]1",
    "hydroxyl": "[OX2H]",
    "aldehyde": "[CX3H1](=O)[#6]",
    "ketone": "[#6][CX3](=O)[#6]",
    "carboxyl": "[CX3](=O)[OX2H1]",
    "ester": "[#6][CX3](=O)[OX2H0][#6]",
    "anhydride": "[CX3](=[OX1])[OX2][CX3](=[OX1])",
    "amine": "[NX3;H2,H1;!$(NC=O)]",
    "amide": "[NX3][CX3](=[OX1])[#6]",
    "nitro": "[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]",
    "halo": "[F,Cl,Br,I]",
    "thiol": "[#16X2H]",
    "thioether": "[SX2][CX4]",
    "disulfide": "[#16X2H0][#16X2H0]",
    "sulfoxide": "[$([#16X3]=[OX1]),$([#16X3+][OX1-])]",
    "sulfone": "[$([#16X4](=[OX1])=[OX1]),$([#16X4+2]([OX1-])[OX1-])]",
    "sulfide": "[#16X2H0]",
    "nitrile": "[NX1]#[CX2]",
    "borane": "[BX3]",
}

GROUP_ALIASES = {
    "benzene ring": "benzene_ring",
    "benzene-ring": "benzene_ring",
}

SMILES_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9@+\-\[\]\(\)=#$\\/%.]+")
ANSWER_TAG_PATTERN = re.compile(r"<answer>\s*([^<]+?)\s*</answer>", flags=re.IGNORECASE | re.DOTALL)


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


def _build_tokenizer(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    mol_ids = tokenizer("<mol>", add_special_tokens=False).input_ids
    if len(mol_ids) != 1:
        tokenizer.add_special_tokens({"additional_special_tokens": ["<mol>"]})
        mol_ids = tokenizer("<mol>", add_special_tokens=False).input_ids
    tokenizer.mol_token_id = mol_ids[0]
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _normalize_group(group: Optional[str]) -> Optional[str]:
    if not isinstance(group, str):
        return None
    g = group.strip().lower().replace(".", "")
    g = GROUP_ALIASES.get(g, g)
    g = g.replace(" ", "_")
    return g if g in GROUP_TO_SMARTS else None


def _safe_parse_meta(meta_value) -> Dict:
    if isinstance(meta_value, dict):
        return meta_value
    if isinstance(meta_value, str):
        try:
            obj = json.loads(meta_value)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}
    return {}


def _canonicalize_smiles(smiles: Optional[str]) -> Optional[str]:
    if not isinstance(smiles, str) or len(smiles.strip()) == 0:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def _extract_predicted_smiles(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None
    m = ANSWER_TAG_PATTERN.search(text)
    if m:
        tagged = m.group(1).strip()
        can = _canonicalize_smiles(tagged)
        if can is not None:
            return can

    tokens = SMILES_TOKEN_PATTERN.findall(text)
    tokens = sorted(set(tokens), key=len, reverse=True)
    for tok in tokens:
        can = _canonicalize_smiles(tok)
        if can is not None:
            return can
    return None


def _count_group(smiles: str, group: str) -> Optional[int]:
    smarts = GROUP_TO_SMARTS.get(group)
    if smarts is None:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    patt = Chem.MolFromSmarts(smarts)
    if patt is None:
        return None
    matches = mol.GetSubstructMatches(patt)
    if group == "sulfide":
        disulfide_patt = Chem.MolFromSmarts(GROUP_TO_SMARTS["disulfide"])
        disulfide_matches = mol.GetSubstructMatches(disulfide_patt)
        return max(0, len(matches) - len(disulfide_matches))
    return len(matches)


def _check_edit_correct(subtask: str, src: str, pred: str, add_group: Optional[str], remove_group: Optional[str]) -> bool:
    if subtask == "add":
        g = _normalize_group(add_group)
        if g is None:
            return False
        src_c = _count_group(src, g)
        pred_c = _count_group(pred, g)
        return src_c is not None and pred_c is not None and pred_c == src_c + 1
    if subtask == "delete":
        g = _normalize_group(remove_group)
        if g is None:
            return False
        src_c = _count_group(src, g)
        pred_c = _count_group(pred, g)
        return src_c is not None and pred_c is not None and pred_c == src_c - 1
    if subtask == "sub":
        g_add = _normalize_group(add_group)
        g_rm = _normalize_group(remove_group)
        if g_add is None or g_rm is None:
            return False
        src_add = _count_group(src, g_add)
        pred_add = _count_group(pred, g_add)
        src_rm = _count_group(src, g_rm)
        pred_rm = _count_group(pred, g_rm)
        return (
            src_add is not None and pred_add is not None and pred_add == src_add + 1
            and src_rm is not None and pred_rm is not None and pred_rm == src_rm - 1
        )
    return False


def _morgan_similarity(smiles_a: Optional[str], smiles_b: Optional[str]) -> float:
    if smiles_a is None or smiles_b is None:
        return 0.0
    mol_a = Chem.MolFromSmiles(smiles_a)
    mol_b = Chem.MolFromSmiles(smiles_b)
    if mol_a is None or mol_b is None:
        return 0.0
    fp_a = AllChem.GetMorganFingerprintAsBitVect(mol_a, radius=2, nBits=2048)
    fp_b = AllChem.GetMorganFingerprintAsBitVect(mol_b, radius=2, nBits=2048)
    return float(DataStructs.TanimotoSimilarity(fp_a, fp_b))


def _prepare_text_batch(tokenizer, llama_version: str, user_prompt: str, device: str):
    mol_prompt = "<mol><mol><mol><mol><mol><mol><mol><mol>"
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


def _load_model_and_tokenizer(args):
    torch_dtype = _precision_to_dtype(args.precision)
    device = _resolve_device(args.device)
    llama_version = "llama3" if "Llama-3" in args.pretrained_model_name_or_path else "llama2"

    tokenizer = _build_tokenizer(args.pretrained_model_name_or_path)
    model = MolLLaMA.from_pretrained(
        args.pretrained_model_name_or_path,
        vocab_size=len(tokenizer),
        torch_dtype=torch_dtype,
        enable_flash=bool(args.enable_flash),
    ).to(device)
    model.eval()

    if llama_version == "llama3":
        eos_token_id = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    else:
        eos_token_id = tokenizer.eos_token_id
    return model, tokenizer, device, llama_version, eos_token_id


def run_eval(args):
    model, tokenizer, device, llama_version, eos_token_id = _load_model_and_tokenizer(args)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"
    metrics_path = out_dir / "metrics.json"

    results = []
    metrics = {"overall": {}, "by_task": {}}

    for subtask, filename in MOLEDIT_TASK_FILES.items():
        test_file = Path(args.test_dir) / filename
        if not test_file.exists():
            raise FileNotFoundError(f"Missing test file: {test_file}")
        with test_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise RuntimeError(f"{test_file} must be a JSON list.")
        if args.max_samples_per_task > 0:
            data = data[: args.max_samples_per_task]

        iterable = data
        if tqdm is not None:
            iterable = tqdm(data, desc=f"Eval {subtask}", unit="sample")

        correct = 0
        valid = 0
        exact = 0
        sim_sum = 0.0
        total = len(data)

        for row in iterable:
            q = str(row.get("query") or "").strip()
            sample_id = row.get("id")
            meta = _safe_parse_meta(row.get("meta"))
            source_smiles = _canonicalize_smiles(meta.get("molecule"))
            ref_smiles = _canonicalize_smiles(meta.get("reference"))

            text_batch = _prepare_text_batch(tokenizer, llama_version, q, device)
            with torch.no_grad():
                if source_smiles is not None:
                    outputs = model.generate_with_smiles(
                        smiles_list=[source_smiles],
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

            decoded = _decode_new_tokens(tokenizer, outputs, text_batch)
            pred_smiles = _extract_predicted_smiles(decoded)
            is_valid = pred_smiles is not None
            is_exact = bool(pred_smiles is not None and ref_smiles is not None and pred_smiles == ref_smiles)
            is_correct = False
            if pred_smiles is not None and source_smiles is not None:
                is_correct = _check_edit_correct(
                    subtask=subtask,
                    src=source_smiles,
                    pred=pred_smiles,
                    add_group=meta.get("added_group"),
                    remove_group=meta.get("removed_group"),
                )
            similarity = _morgan_similarity(pred_smiles, ref_smiles)

            valid += int(is_valid)
            exact += int(is_exact)
            correct += int(is_correct)
            sim_sum += similarity

            results.append(
                {
                    "sample_id": sample_id,
                    "task": subtask,
                    "query": q,
                    "source_smiles": source_smiles,
                    "reference_smiles": ref_smiles,
                    "pred_smiles": pred_smiles,
                    "raw_model_output": decoded,
                    "added_group": meta.get("added_group"),
                    "removed_group": meta.get("removed_group"),
                    "is_valid": is_valid,
                    "is_exact_match": is_exact,
                    "is_edit_correct": is_correct,
                    "morgan_similarity_to_reference": similarity,
                }
            )

        metrics["by_task"][subtask] = {
            "total": total,
            "valid_rate": (valid / total) if total > 0 else 0.0,
            "exact_match_rate": (exact / total) if total > 0 else 0.0,
            "correct_rate": (correct / total) if total > 0 else 0.0,
            "avg_morgan_similarity": (sim_sum / total) if total > 0 else 0.0,
        }

    overall_total = len(results)
    overall_valid = sum(int(x["is_valid"]) for x in results)
    overall_exact = sum(int(x["is_exact_match"]) for x in results)
    overall_correct = sum(int(x["is_edit_correct"]) for x in results)
    overall_sim = sum(float(x["morgan_similarity_to_reference"]) for x in results)
    metrics["overall"] = {
        "total": overall_total,
        "valid_rate": (overall_valid / overall_total) if overall_total > 0 else 0.0,
        "exact_match_rate": (overall_exact / overall_total) if overall_total > 0 else 0.0,
        "correct_rate": (overall_correct / overall_total) if overall_total > 0 else 0.0,
        "avg_morgan_similarity": (overall_sim / overall_total) if overall_total > 0 else 0.0,
    }
    metrics["run_config"] = {
        "model_path": args.pretrained_model_name_or_path,
        "test_dir": str(args.test_dir),
        "do_sample": bool(args.do_sample),
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "max_new_tokens": int(args.max_new_tokens),
        "min_new_tokens": int(args.min_new_tokens),
        "precision": str(args.precision),
        "enable_flash": bool(args.enable_flash),
    }

    with pred_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"Saved predictions: {pred_path}")
    print(f"Saved metrics: {metrics_path}")
    print(json.dumps(metrics["overall"], ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Inference + evaluation for MolEdit using pure Mol-LLaMA.")
    parser.add_argument("--pretrained_model_name_or_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--test_dir", default=DEFAULT_TEST_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="")
    parser.add_argument("--precision", default="bf16-mixed")
    parser.add_argument("--enable_flash", action="store_true")
    parser.add_argument("--max_samples_per_task", type=int, default=0, help="0 means all")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--min_new_tokens", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    args = parser.parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()
