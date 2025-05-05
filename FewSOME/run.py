import os

if __name__ == '__main__':
   commands = []
   seeds =[1001, 3472, 128037, 875688, 71530]
   ns = [100, 200, 300]
   for seed in seeds:
      for n in ns:
               if (n == 100) | (n == 200):
                 alp = 0.6
                 lr = 1e-5
               else:
                 alp = 0.01
                 lr = 1e-4
               model_name = 'model_N_' + str(n) + '_lr_' + str(lr) + '_alpha_' + str(alp) + '_seed_' + str(seed)
               commands.append("python3 main.py --device cuda:0 --eval_epoch 0 --model_name " + model_name + " --model_type RESNET --batch_size 30  --num_ref " + str(n) + " --seed " + str(seed) + " --epochs 100 --alpha " + str(alp) + "  --lr " + str(lr) + " --smart_samp 0 --k 1 --weight_init_seed " + str(seed) + " --biases 0")


for run in commands:
    os.system(run)