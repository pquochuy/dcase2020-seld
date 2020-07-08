# run experiments on the development data
# validation performance is the last entry in current_best_valid.txt
# test performance is the last entry in current_best_test.txt
CUDA_VISIBLE_DEVICES="0,-1" python3 train_seld.py --out_dir "./dcase2020_dev_foa/" --task_id 4 --augment 1 --learning_rate 0.0002 --decay_rate 0.8 --training_epoch 10000
CUDA_VISIBLE_DEVICES="0,-1" python3 train_seld.py --out_dir "./dcase2020_dev_mic/" --task_id 2 --augment 1 --learning_rate 0.0002 --decay_rate 0.8 --training_epoch 10000
