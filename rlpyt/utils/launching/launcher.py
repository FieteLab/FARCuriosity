import os
import sys
import subprocess
import time
import json
import argparse
from six.moves import shlex_quote
import GPUtil
import psutil
import torch

# Runners
from rlpyt.runners.minibatch_rl import MinibatchRl, MinibatchRlEval

# Policies
from rlpyt.agents.pg.atari import AtariFfAgent, AtariLstmAgent

# Samplers
from rlpyt.samplers.parallel.cpu.collectors import CpuResetCollector, CpuWaitResetCollector
from rlpyt.samplers.parallel.gpu.collectors import GpuResetCollector, GpuWaitResetCollector
from rlpyt.samplers.parallel.cpu.sampler import CpuSampler
from rlpyt.samplers.parallel.gpu.sampler import GpuSampler

# Environments
from rlpyt.samplers.collections import TrajInfo
from rlpyt.envs.atari.atari_env import AtariEnv, AtariTrajInfo

# Learning Algorithms
from rlpyt.algos.pg.ppo import PPO
from rlpyt.algos.pg.a2c import A2C

# Utils
from rlpyt.utils.logging.context import logger_context


with open('./global.json') as global_params:
    params = json.load(global_params)
    _RESULTS_DIR = params['local_resultsdir']
    _ATARI_ENVS = params['envs']['atari_envs']



def get_logdir(pretrain, args):
    name = '_'.join([args.alg, args.curiosity_alg, args.env])
    if args.fragmentation:
        name = f'Frag_{name}_{args.frag_criteria}{args.frag_th}_{args.cos_th}C{args.cos_th_min}R{args.recall_th}'
        if args.use_feature:
            name = name + '_feat'
    if os.path.isdir(f'{_RESULTS_DIR}/{name}/run_0'):
        runs = os.listdir(f'{_RESULTS_DIR}/{name}')
        try:
            runs.remove('tmp')
        except ValueError:
            pass
        try:
            runs.remove('.DS_Store')
        except ValueError:
            pass
        sorted_runs = sorted(runs, key=lambda run: int(run.split('_')[-1]))
        run_id = int(sorted_runs[-1].split('_')[-1]) + 1
    else:
        run_id = 0
        os.makedirs(os.path.join(_RESULTS_DIR, name, f'run_{run_id}'))
    log_dir = os.path.join(_RESULTS_DIR, name, f'run_{run_id}')
    return log_dir


def get_agent(args, initial_model_state_dict, log_dir):
    model_args = dict(curiosity_kwargs=dict(curiosity_alg=args.curiosity_alg))

    # fragmentation
    model_args['fragmentation_kwargs'] = dict(fragmentation=args.fragmentation)
    if args.fragmentation:
        model_args['fragmentation_kwargs']['use_recall'] = args.recall
        model_args['fragmentation_kwargs']['threshold'] = args.frag_th
        model_args['fragmentation_kwargs']['recall_threshold'] = args.recall_th
        model_args['fragmentation_kwargs']['mem_size'] = args.mem_size
        model_args['fragmentation_kwargs']['num_envs'] = args.num_envs
        model_args['fragmentation_kwargs']['frag_obs_scale'] = args.frag_obs_scale # used when obs similarity is used.
        model_args['fragmentation_kwargs']['device'] = args.sample_mode
        model_args['fragmentation_kwargs']['cos_th'] = args.cos_th
        model_args['fragmentation_kwargs']['cos_th_min'] = min(args.cos_th_min, args.cos_th)
        model_args['fragmentation_kwargs']['use_feature'] = args.use_feature
        model_args['fragmentation_kwargs']['frag_criteria'] = args.frag_criteria
    if args.curiosity_alg == 'icm':
        model_args['curiosity_kwargs']['feature_encoding'] = args.feature_encoding
        model_args['curiosity_kwargs']['batch_norm'] = args.batch_norm
        model_args['curiosity_kwargs']['prediction_beta'] = args.prediction_beta
        model_args['curiosity_kwargs']['forward_loss_wt'] = args.forward_loss_wt
        model_args['curiosity_kwargs']['forward_model'] = args.forward_model
        model_args['curiosity_kwargs']['feature_space'] = args.feature_space
        model_args['curiosity_kwargs']['fix_features'] = args.fix_features
        model_args['curiosity_kwargs']['dual_value'] = args.dual_value
        model_args['curiosity_kwargs']['dual_policy'] = args.dual_policy
    elif args.curiosity_alg == 'micm':
        model_args['curiosity_kwargs']['feature_encoding'] = args.feature_encoding
        model_args['curiosity_kwargs']['batch_norm'] = args.batch_norm
        model_args['curiosity_kwargs']['prediction_beta'] = args.prediction_beta
        model_args['curiosity_kwargs']['forward_loss_wt'] = args.forward_loss_wt
        model_args['curiosity_kwargs']['forward_model'] = args.forward_model
        model_args['curiosity_kwargs']['ensemble_mode'] = args.ensemble_mode
        model_args['curiosity_kwargs']['device'] = args.sample_mode
        model_args['curiosity_kwargs']['dual_value'] = args.dual_value
        model_args['curiosity_kwargs']['dual_policy'] = args.dual_policy
    elif args.curiosity_alg == 'disagreement':
        model_args['curiosity_kwargs']['feature_encoding'] = args.feature_encoding
        model_args['curiosity_kwargs']['ensemble_size'] = args.ensemble_size
        model_args['curiosity_kwargs']['batch_norm'] = args.batch_norm
        model_args['curiosity_kwargs']['prediction_beta'] = args.prediction_beta
        model_args['curiosity_kwargs']['forward_loss_wt'] = args.forward_loss_wt
        model_args['curiosity_kwargs']['device'] = args.sample_mode
        model_args['curiosity_kwargs']['forward_model'] = args.forward_model
        model_args['curiosity_kwargs']['dual_value'] = args.dual_value
        model_args['curiosity_kwargs']['dual_policy'] = args.dual_policy
    elif args.curiosity_alg == 'ndigo':
        model_args['curiosity_kwargs']['feature_encoding'] = args.feature_encoding
        model_args['curiosity_kwargs']['pred_horizon'] = args.pred_horizon
        model_args['curiosity_kwargs']['prediction_beta'] = args.prediction_beta
        model_args['curiosity_kwargs']['batch_norm'] = args.batch_norm
        model_args['curiosity_kwargs']['device'] = args.sample_mode
        model_args['curiosity_kwargs']['dual_policy'] = args.dual_policy
    elif args.curiosity_alg == 'rnd':
        model_args['curiosity_kwargs']['feature_encoding'] = args.feature_encoding
        model_args['curiosity_kwargs']['prediction_beta'] = args.prediction_beta
        model_args['curiosity_kwargs']['drop_probability'] = args.drop_probability
        model_args['curiosity_kwargs']['gamma'] = args.discount_ri
        model_args['curiosity_kwargs']['device'] = args.sample_mode
        model_args['curiosity_kwargs']['dual_value'] = args.dual_value
        model_args['curiosity_kwargs']['dual_policy'] = args.dual_policy

    if args.lstm:
        agent = AtariLstmAgent(
                    initial_model_state_dict=initial_model_state_dict,
                    model_kwargs=model_args,
                    no_extrinsic=args.no_extrinsic,
                    dual_policy=args.dual_policy
                    )
    else:
        agent = AtariFfAgent(initial_model_state_dict=initial_model_state_dict)
    return agent


def get_algo(args, initial_algo_state_dict, initial_optim_state_dict, log_dir):   
    if args.alg == 'ppo':
        algo = PPO(
                discount=args.discount,
                discount_ri=getattr(args, 'discount_ri', 0.0),
                learning_rate=args.lr,
                value_loss_coeff=args.v_loss_coeff,
                entropy_loss_coeff=args.entropy_loss_coeff,
                OptimCls=torch.optim.Adam,
                optim_kwargs=None,
                clip_grad_norm=args.grad_norm_bound,
                initial_optim_state_dict=initial_optim_state_dict, # is None is not reloading a checkpoint
                gae_lambda=args.gae_lambda,
                minibatches=args.minibatches, # if recurrent: batch_B needs to be at least equal, if not recurrent: batch_B*batch_T needs to be at least equal to this
                epochs=args.epochs,
                ratio_clip=args.ratio_clip,
                linear_lr_schedule=args.linear_lr,
                normalize_advantage=args.normalize_advantage,
                normalize_reward=args.normalize_reward,
                normalize_extreward=args.normalize_extreward,
                normalize_intreward=args.normalize_intreward,
                rescale_extreward=args.rescale_extreward,
                rescale_intreward=args.rescale_intreward,
                dual_value=getattr(args, 'dual_value', False),
                dual_policy=args.dual_policy,
                dual_policy_noint=args.dual_policy_noint,
                dual_policy_weighting=args.dual_policy_weighting,
                dpw_formulation=args.dpw_formulation,
                utility_noworkers=args.utility_noworkers,
                kl_lambda=args.kl_lambda,
                kl_clamp=args.kl_clamp,
                util_clamp=args.util_clamp,
                util_detach=args.util_detach,
                kl_detach=args.kl_detach,
                importance_sample=args.importance_sample,
                curiosity_decay=getattr(args, 'curiosity_decay', False),
                initial_algo_state_dict=initial_algo_state_dict,
                curiosity_type=args.curiosity_alg,
                )
    elif args.alg == 'a2c':
        algo = A2C(
                discount=args.discount,
                learning_rate=args.lr,
                value_loss_coeff=args.v_loss_coeff,
                entropy_loss_coeff=args.entropy_loss_coeff,
                OptimCls=torch.optim.Adam,
                optim_kwargs=None,
                clip_grad_norm=args.grad_norm_bound,
                initial_optim_state_dict=initial_optim_state_dict,
                gae_lambda=args.gae_lambda,
                normalize_advantage=args.normalize_advantage
                )
    return algo


def get_envs(args, log_dir):
    # environment setup
    traj_info_cl = TrajInfo # environment specific - potentially overriden below
    if args.env not in _ATARI_ENVS:
        raise Exception()
    env_cl = AtariEnv
    traj_info_cl = AtariTrajInfo
    env_args = dict(
        game=args.env,
        no_extrinsic=args.no_extrinsic,
        no_negative_reward=args.no_negative_reward,
        normalize_obs=args.normalize_obs,
        normalize_obs_steps=10000,
        downsampling_scheme='classical',
        record_freq=args.record_freq,
        record_dir=log_dir,
        horizon=args.max_episode_steps,
        score_multiplier=args.score_multiplier,
        repeat_action_probability=args.repeat_action_probability,
        fire_on_reset=args.fire_on_reset
        )

    return traj_info_cl, env_cl, env_args


def start_experiment(args, use_jaynes=False):

    log_dir = get_logdir(args.pretrain, args)
    print(log_dir)
    args_json = json.dumps(vars(args), indent=4)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    with open(log_dir + '/arguments.json', 'w') as jsonfile:
        jsonfile.write(args_json)

    # W&B logging
    if args.use_wandb:
      import wandb
      wandb.init(name=log_dir, project=args.prj_name, entity=args.entity, sync_tensorboard=True, dir=log_dir, resume=True, config=args)
      wandb.config = vars(args)

    '''
    Overwrite the _RESULT_DIR so that we can set result dir from the launching script
    '''
    if args.result_dir:
      _RESULTS_DIR = args.result_dir

    if not use_jaynes:
      '''
      Jaynes monut code is not a git directory, but jaynes provides its own git record.
      '''
      with open(log_dir + '/git.txt', 'w') as git_file:
          branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).strip().decode('utf-8')
          commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('utf-8')
          git_file.write('{}/{}'.format(branch, commit))

    config = dict(env_id=args.env)
    affinity = dict(cuda_idx=0, workers_cpus=psutil.Process().cpu_affinity()[:args.num_cpus])

    # potentially reload models
    initial_optim_state_dict = None
    initial_model_state_dict = None
    initial_algo_state_dict = None
    if args.pretrain != 'None':
        os.system(f"find {log_dir} -name '*.json' -delete") # clean up json files for video recorder
        if '.pkl' not in args.pretrain:
            try:
                checkpoint = torch.load(os.path.join(args.pretrain, 'params.pkl'))
            except:
                checkpoint = torch.load(os.path.join(args.pretrain, 'params_old.pkl'))
        else:
            checkpoint = torch.load(args.pretrain)
        initial_optim_state_dict = checkpoint['optimizer_state_dict']
        initial_model_state_dict = checkpoint['agent_state_dict']
        initial_algo_state_dict = checkpoint.get('algo_state_dict', None)
        
    encoder_pretrain = getattr(args, 'encoder_pretrain', 'None')
    if encoder_pretrain != 'None' and encoder_pretrain != None:
        try:
            checkpoint = torch.load(os.path.join(_RESULTS_DIR, encoder_pretrain, 'params.pkl'))
        except:
            checkpoint = torch.load(os.path.join(_RESULTS_DIR, encoder_pretrain, 'params_old.pkl'))

    # ----------------------------------------------------- POLICY ----------------------------------------------------- #
    agent = get_agent(args, initial_model_state_dict, log_dir)
    algo = get_algo(args, initial_algo_state_dict, initial_optim_state_dict, log_dir)
    traj_info_cl, env_cl, env_args = get_envs(args, log_dir)

    if args.sample_mode == 'gpu':
        if args.lstm:
            collector_class = GpuWaitResetCollector
        else:
            collector_class = GpuResetCollector
        sampler = GpuSampler(
            EnvCls=env_cl,
            env_kwargs=env_args,
            eval_env_kwargs=env_args,
            batch_T=args.timestep_limit,
            batch_B=args.num_envs,
            max_decorrelation_steps=0,
            TrajInfoCls=traj_info_cl,
            eval_n_envs=args.eval_envs,
            eval_max_steps=args.eval_max_steps,
            eval_max_trajectories=args.eval_max_traj,
            record_freq=args.record_freq,
            log_dir=log_dir,
            CollectorCls=collector_class
        )
    else:
        if args.lstm:
            collector_class = CpuWaitResetCollector
        else:
            collector_class = CpuResetCollector
        sampler = CpuSampler(
            EnvCls=env_cl,
            env_kwargs=env_args,
            eval_env_kwargs=env_args,
            batch_T=args.timestep_limit, # timesteps in a trajectory episode
            batch_B=args.num_envs, # environments distributed across workers
            max_decorrelation_steps=0,
            TrajInfoCls=traj_info_cl,
            eval_n_envs=args.eval_envs,
            eval_max_steps=args.eval_max_steps,
            eval_max_trajectories=args.eval_max_traj,
            record_freq=args.record_freq,
            log_dir=log_dir,
            CollectorCls=collector_class
            )

    # ----------------------------------------------------- RUNNER ----------------------------------------------------- #
    if args.eval_envs > 0:
        runner = MinibatchRlEval(
            algo=algo,
            agent=agent,
            sampler=sampler,
            n_steps=args.iterations,
            affinity=affinity,
            log_interval_steps=args.log_interval,
            log_dir=log_dir,
            pretrain=args.pretrain
            )
    else:
        runner = MinibatchRl(
            algo=algo,
            agent=agent,
            sampler=sampler,
            n_steps=args.iterations,
            affinity=affinity,
            log_interval_steps=args.log_interval,
            log_dir=log_dir,
            pretrain=args.pretrain,
            model_save_freq=args.model_save_freq
            )
    
    with logger_context(log_dir, config, snapshot_mode="last", use_summary_writer=True):
        runner.train()

    if args.use_wandb:
      wandb.finish()
