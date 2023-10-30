# Code Instruction for GoMARL

This instruction hosts the PyTorch implementation of GoMARL accompanying the paper "**Automatic Grouping for Efficient Cooperative Multi-Agent Reinforcement Learning**" (NeurIPS, 2023). GoMARL is a domain-agnostic method that learns automatic grouping for efficient cooperation by promoting intra- and inter-group coordination.

The implementation is based on the frameworks [PyMARL](https://github.com/oxwhirl/pymarl) and [PyMARL2](https://github.com/hijkzzz/pymarl2) with environments [SMAC](https://github.com/oxwhirl/smac) and [Gfootball](https://github.com/google-research/football). All of our SMAC experiments are based on the latest PyMARL2 utilizing SC2.4.6.10. The underlying dynamics are sufficiently different, so you cannot compare runs across various versions.


## Setup

Set up the working environment: 

```shell
pip3 install -r requirements.txt
```

Set up the StarCraftII game core (SC2.4.6.10): 

```shell
bash install_sc2.sh
```

Set up the Gfootball environment:

```shell
bash install_gfootball.sh
```

## Training

To train `GoMARL` on the `3s5z_vs_3s6z`(SMAC) scenario: 

```shell
python3 src/main.py --config=group --env-config=sc2 with env_args.map_name=3s5z_vs_3s6z
```

Change the `map_name` accordingly for other SMAC scenarios (e.g., `MMM2`). 

To train `GoMARL` on the `academy_3_vs_1_with_keeper`(Gfootball) scenario:

```
python3 src/main.py --config=group --env-config=academy_3_vs_1_with_keeper
```

Change the `env-config` accordingly for other Gfootball scenarios (e.g., `academy_counterattack_easy`). 

All results will be saved in the `results` folder. 

The config file `src/config/algs/group.yaml` contains default hyperparameters for GoMARL.


## Evaluation

### TensorBoard

One could set `use_tensorboard` to `True` in `src/config/default.yaml`, and the training tensorboards will be saved in the `results/tb_logs` directory, containing useful info such as test battle win rate during training. 

### Saving models

Same as PyMARL, set `save_model` to `True` in `src/config/default.yaml`, and the learned model during training will be saved in the `results/models/` directory. The frequency for saving models can be adjusted by setting the parameter `save_model_interval`.

### Loading models

Saved models can be loaded by adjusting the `checkpoint_path` parameter in `src/config/default.yaml`. For instance, to load the model under path `result/model/[timesteps]/agent.th`, set `checkpoint_path` to `result/model/[timesteps]`.

### Saving Starcraft II Replay

The learned model loaded from `checkpoint_path` can be evaluated by setting `evaluate` to `True` in `src/config/default.yaml`. To save the Starcraft II replays, please make sure the configuration `save_replay` is set to `True`, and use the `episode_runner`.

Check out [PyMARL](https://github.com/oxwhirl/pymarl) documentation for more information.

## See Also

See [SMAC](https://github.com/oxwhirl/smac), [PyMARL2](https://github.com/hijkzzz/pymarl2), [PyMARL](https://github.com/oxwhirl/pymarl) and [Gfootball](https://github.com/google-research/football) for additional instructions.
