## Federated Learning with Reduced Information Leakage and Computation

This repo contains PyTorch implementation of **Upcycled-FL**. Upcycled-FL is a simple yet effective strategy that applies first-order approximation at every even round to reduce the data accesses and improve the privacy guarantees. For more details, please check our paper [Federated Learning with Reduced Information Leakage and Computation](https://openreview.net/forum?id=ZJ4A3xhADV).

### Usage

We have included all the baselines used in the paper within this repository.

To run the code:

```
python -u train.py --baseline <baseline_name> --algorithm <Upcycled or Vanilla> --seed <random_seed> --dataset <dataset_name>  --epochs <number_epochs> --frac <fraction of devices> --local_ep <number_epoch_clients> --lr <learning_rate> --upcycled_param <even_step_upcycled_parameter> --clip <output_perturbation_parameter> --sigma <output_perturbation_parameter> --alpha <objective_perturbation_parameter> --local_bs <batch_size> --straggler <straggler_ratio> --gpu <gpu_id> |  tee <your_log_path>
```

Taking Upcycled-FedAvg as an example:

#### Non-private

To compare convergence, we conduct experiments under the similar training time.

```
python -u train.py --baseline avg --algorithm Upcycled --seed 2 --dataset synthetic_0.5_0.5  --epochs 160 --frac 0.3 --local_ep 10 --lr 5e-3 --upcycled_param 0.7 --clip 0 --sigma 0 --alpha 0 --local_bs 32 --straggler 0.3 --gpu 0 
python -u train.py --baseline avg --algorithm Vanilla --seed 2 --dataset synthetic_0.5_0.5  --epochs 80 --frac 0.3 --local_ep 10 --lr 5e-3 --upcycled_param 0 --clip 0 --sigma 0 --alpha 0 --local_bs 32 --straggler 0.3 --gpu 0 
```

#### Output Perturbation

```
python -u train.py --baseline avg --algorithm Upcycled --seed 2 --dataset synthetic_0.5_0.5  --epochs 80 --frac 1 --local_ep 10 --lr 7e-2 --upcycled_param 0.3 --clip 10 --sigma 0.8 --alpha 0 --local_bs 32 --gpu 0 
python -u train.py --baseline avg --algorithm Vanilla --seed 2 --dataset synthetic_0.5_0.5  --epochs 80 --frac 1 --local_ep 10 --lr 7e-2 --upcycled_param 0 --clip 10 --sigma 1 --alpha 0 --local_bs 32  --gpu 0 
```

We provide *clip* and *sigma* configuration in train.py to guarentee stricter privacy for Upcycled-FL.

#### Objective Perturbation

```
python -u train.py --baseline avg --algorithm Upcycled --seed 2 --dataset synthetic_0.5_0.5  --epochs 80 --frac 1 --local_ep 10 --lr 1e-3 --upcycled_param 0.7 --clip 0 --sigma 0 --alpha 20 --local_bs 32 --gpu 0 
python -u train.py --baseline avg --algorithm Vanilla --seed 2 --dataset synthetic_0.5_0.5  --epochs 80 --frac 1 --local_ep 10 --lr 1e-3 --upcycled_param 0 --clip 0 --sigma 0 --alpha 10 --local_bs 32  --gpu 0 
```

For objective Perturbation, please keep *clip* and *sigma* as 0.

### Citing

TODO

### Contact

If you have any questions, you can contact me via email at tanxuwei99@gmail.com. You can also open an issue here, but please note that this repo is managed by a public account, so I might not see it immediately.

### Acknowledgement

Our code is based on existing FL repos. We sincerely appreciate the following github repos:

[https://github.com/litian96/FedProx]()

[https://github.com/KarhouTam/FL-bench]()

[https://github.com/DataSysTech/FedTune]()
