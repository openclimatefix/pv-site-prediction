config=$1
exp_name=$2
num_proc=16
# num_test_samples=100000
poetry run python psp/scripts/train_model.py -n $exp_name \
    -c $config \
    -b 1 \
    -w $num_proc \
    --force \
    --no_infer
poetry run python psp/scripts/eval_model.py -n $exp_name \
    -w $num_proc \
    -l 2000
