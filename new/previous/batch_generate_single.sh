#!/usr/bin/env bash
set -euo pipefail

# ====== 配置区 ======
MODEL_PATH="./save/humanml_trans_dec_512_bert/model000200000.pt"
PROMPT_FILE="/root/autodl-tmp/motion-diffusion-model/assets/test_30_prompt.txt"
BASE_OUTDIR="./save/263_batch_single_outputs"

# 每条 prompt 只生成 1 个样本、1 个 repetition
NUM_SAMPLES=1
NUM_REPETITIONS=1

# 可选：如果你想换成 trans_dec 模型，就改成：
# MODEL_PATH="./save/humanml_trans_dec_512_bert/model000200000.pt"

# ====== 检查 ======
if [[ ! -f "$MODEL_PATH" ]]; then
  echo "[ERROR] model not found: $MODEL_PATH"
  exit 1
fi

if [[ ! -f "$PROMPT_FILE" ]]; then
  echo "[ERROR] prompt file not found: $PROMPT_FILE"
  exit 1
fi

mkdir -p "$BASE_OUTDIR"

echo "[INFO] model: $MODEL_PATH"
echo "[INFO] prompts: $PROMPT_FILE"
echo "[INFO] output root: $BASE_OUTDIR"
echo

# ====== 主循环 ======
idx=0
while IFS= read -r line || [[ -n "$line" ]]; do
  # 去掉首尾空白
  prompt="$(echo "$line" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"

  # 跳过空行
  if [[ -z "$prompt" ]]; then
    continue
  fi

  # 跳过注释行（以 # 开头）
  if [[ "$prompt" =~ ^# ]]; then
    continue
  fi

  idx=$((idx + 1))
  case_id=$(printf "%04d" "$idx")

  # 生成一个适合文件夹名的短标签
  short_name=$(echo "$prompt" \
    | tr '[:upper:]' '[:lower:]' \
    | sed 's/[^a-z0-9]/_/g' \
    | sed 's/__*/_/g' \
    | sed 's/^_//;s/_$//' \
    | cut -c1-50)

  if [[ -z "$short_name" ]]; then
    short_name="prompt"
  fi

  outdir="${BASE_OUTDIR}/${case_id}_${short_name}"
  mkdir -p "$outdir"

  # 把原始 prompt 也保存进去，方便回看
  printf "%s\n" "$prompt" > "${outdir}/prompt.txt"

  echo "[INFO] (${case_id}) generating:"
  echo "       $prompt"
  echo "       -> $outdir"

  python -u -m sample.generate \
    --model_path "$MODEL_PATH" \
    --text_prompt "$prompt" \
    --num_samples "$NUM_SAMPLES" \
    --num_repetitions "$NUM_REPETITIONS" \
    --output_dir "$outdir" \
    2>&1 | tee "${outdir}/generate.log"

  echo "[INFO] done: ${outdir}"
  echo
done < "$PROMPT_FILE"

echo "[INFO] all done. total prompts processed: $idx"