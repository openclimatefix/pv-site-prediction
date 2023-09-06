# The script run_exp.sh can be used to train and then evaluate a model, for example
#./run_exp.sh exp_config_to_use name_for_exp
# Current set to evalutate 2000 samples

config=$1
exp_name=$2
num_proc=16
poetry run python psp/scripts/train_model.py -n $exp_name \
    -c $config \
    -b 1 \
    -w $num_proc \
    --force \
    --no_infer
poetry run python psp/scripts/eval_model.py -n $exp_name \
    -w $num_proc \
    -l 2000
