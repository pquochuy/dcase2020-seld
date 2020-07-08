# submission 1
CUDA_VISIBLE_DEVICES="0,-1" python3 train_seld.py --out_dir "./submission1/" --task_id 5 --augment 1 --learning_rate 0.0002 --decay_rate 0.8 --training_epoch 10000
CUDA_VISIBLE_DEVICES="0,-1" python3 test_seld.py --out_dir "./submission1/" --dcase_dir "output" --task_id 5 --augment 1 --learning_rate 0.0002 --decay_rate 0.8 --training_epoch 10000

# submission 2
CUDA_VISIBLE_DEVICES="0,-1" python3 train_seld.py --out_dir "./submission2/" --task_id 3 --augment 1 --learning_rate 0.0002 --decay_rate 0.8 --training_epoch 10000
CUDA_VISIBLE_DEVICES="0,-1" python3 test_seld.py --out_dir "./submission2/" --dcase_dir "output" --task_id 3 --augment 1 --learning_rate 0.0002 --decay_rate 0.8 --training_epoch 10000

# submission 3
CUDA_VISIBLE_DEVICES="0,-1" python3 train_seld_all.py --out_dir "./submission3/" --task_id 5 --augment 1 --learning_rate 0.0002 --decay_rate 0.8 --training_epoch 10000
CUDA_VISIBLE_DEVICES="0,-1" python3 test_seld.py --out_dir "./submission3/" --dcase_dir "output" --task_id 5 --augment 1 --learning_rate 0.0002 --decay_rate 0.8 --training_epoch 10000

# submission 4
CUDA_VISIBLE_DEVICES="0,-1" python3 train_seld_all.py --out_dir "./submission4/" --task_id 3 --augment 1 --learning_rate 0.0002 --decay_rate 0.8 --training_epoch 10000
CUDA_VISIBLE_DEVICES="0,-1" python3 test_seld.py --out_dir "./submission4/" --dcase_dir "output" --task_id 3 --augment 1 --learning_rate 0.0002 --decay_rate 0.8 --training_epoch 10000
