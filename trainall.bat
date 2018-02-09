python train_mlp_nets.py -i datasets/ds_4d_grid_sampling1.csv -od runs/4d_gs_1k/ -vvvvv -epochs 500
python train_nets.py -i datasets/ds_4d_random_sampling1.csv -od runs/4d_rs_1k/ -vvvvv -epochs 500
python train_nets.py -i datasets/3d_grid_sampling1.csv -od runs/3d_gs_1k/ -vvvvv -epochs 500
python train_nets.py -i datasets/3d_random_sampling1.csv -od runs/3d_rs_1k/ -vvvvv -epochs 500