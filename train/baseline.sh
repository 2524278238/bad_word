export CUDA_VISIBLE_DEVICES="0"


python baseline.py \
    --model_path /public/home/xiangyuduan/bli/blidata/models/hf/Llama-3.1-8B \
    --seed 42 \
    --src /public/home/xiangyuduan/lyt/basedata/zhen/test_src.zh \
    --ref /public/home/xiangyuduan/lyt/basedata/zhen/test_ref.en \
    --t30 /public/home/xiangyuduan/lyt/rStar/data/Trigger/test_all.json \
    --write_path /data/lyt/a100/model/llama3_all/test2200.en \
    --batch_size 16 \
    --num_beams 1 \
    --max_new_tokens 150 \
    --do_sample False \
    --temperature 0.0 \
    --top_p 0.00001 \
    --top_k 1 \