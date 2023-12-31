import numpy as np
import torch

from rlpyt.algos.pg.base import PolicyGradientAlgo, OptInfo
from rlpyt.agents.base import AgentInputs, AgentInputsRnn, IcmAgentCuriosityInputs, NdigoAgentCuriosityInputs, RndAgentCuriosityInputs
from rlpyt.utils.tensor import valid_mean
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.buffer import buffer_to, buffer_method
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.misc import iterate_mb_idxs
from rlpyt.utils.averages import RunningMeanStd, RewardForwardFilter
from rlpyt.utils.grad_utils import plot_grad_flow

LossInputs = namedarraytuple("LossInputs", ["agent_inputs", 
                                            "agent_curiosity_inputs", 
                                            "action", 
                                            "return_", 
                                            "return_int_", 
                                            "advantage", 
                                            "advantage_ext",
                                            "advantage_int",
                                            "valid", 
                                            "old_dist_info",
                                            "old_dist_ext_info",
                                            "old_dist_int_info"])

class PPO(PolicyGradientAlgo):
    """
    Proximal Policy Optimization algorithm.  Trains the agent by taking
    multiple epochs of gradient steps on minibatches of the training data at
    each iteration, with advantages computed by generalized advantage
    estimation.  Uses clipped likelihood ratios in the policy loss.
    """

    def __init__(
            self,
            discount=0.99,
            discount_ri=0.99,
            learning_rate=0.001,
            value_loss_coeff=1.,
            entropy_loss_coeff=0.01,
            OptimCls=torch.optim.Adam,
            optim_kwargs=None,
            clip_grad_norm=1.,
            initial_optim_state_dict=None,
            gae_lambda=1,
            minibatches=4,
            epochs=4,
            ratio_clip=0.1,
            linear_lr_schedule=True,
            normalize_advantage=False,
            normalize_reward=False,
            normalize_extreward=False,
            normalize_intreward=False,
            rescale_extreward=False,
            rescale_intreward=False,
            dual_value=False,
            dual_policy='default',
            dual_policy_noint=False,
            dual_policy_weighting='none',
            dpw_formulation='inverse',
            utility_noworkers=False,
            kl_lambda=1.0,
            kl_clamp=0.0,
            util_clamp=0.2,
            util_detach='none',
            kl_detach='none',
            importance_sample=0.,
            curiosity_decay=False,
            initial_algo_state_dict=None,
            curiosity_type='none',
            ):
        """Saves input settings."""
        if optim_kwargs is None:
            optim_kwargs = dict()
        save__init__args(locals())
        if self.normalize_reward:
            self.reward_ff = RewardForwardFilter(discount)
            self.reward_rms = RunningMeanStd()
        if self.normalize_extreward:
            self.extreward_ff = RewardForwardFilter(discount)
            self.extreward_rms = RunningMeanStd()
        if self.normalize_intreward:
            self.intreward_ff = RewardForwardFilter(discount)
            self.intreward_rms = RunningMeanStd()
        if self.dual_policy_weighting != 'none':
            self.advantage_int_weighted = np.array(0.0)
        self.intrinsic_rewards = None
        self.normalized_extreward = None
        self.normalized_intreward = None
        self.rescaled_extreward = None
        self.rescaled_intreward = None
        self.extint_ratio = None


    def algo_state_dict(self):
        if self.curiosity_type == 'none':
            return {}
        return {"curiosity_decay" : self.agent.model.curiosity_model.decay}

    def load_algo_state_dict(self, algo_state_dict):
        self.agent.model.curiosity_model.decay = algo_state_dict["curiosity_decay"]
        
    def initialize(self, *args, **kwargs):
        """
        Extends base ``initialize()`` to initialize learning rate schedule, if
        applicable.
        """
        super().initialize(*args, **kwargs)
        if self.curiosity_decay:
            self.curiosity_decay_rate = (1-0.00001)/self.n_itr
            if self.initial_algo_state_dict is not None:
                self.load_algo_state_dict(self.initial_algo_state_dict)
            
        self._batch_size = self.batch_spec.size // self.minibatches  # For logging.
        if self.linear_lr_schedule:
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer=self.optimizer,
                lr_lambda=lambda itr: (self.n_itr - itr) / self.n_itr)  # Step once per itr.
            self._ratio_clip = self.ratio_clip  # Save base value.

    def optimize_agent(self, itr, samples):
        """
        Train the agent, for multiple epochs over minibatches taken from the
        input samples.  Organizes agent inputs from the training data, and
        moves them to device (e.g. GPU) up front, so that minibatches are
        formed within device, without further data transfer.
        """
        recurrent = self.agent.recurrent
        agent_inputs = AgentInputs(  # Move inpu1ts to device once, index there.
            observation=samples.env.observation,
            prev_action=samples.agent.prev_action,
            prev_reward=samples.env.prev_reward,
        )
        agent_inputs = buffer_to(agent_inputs, device=self.agent.device)

        init_rnn_state = None
        if recurrent:
            # Leave in [B,N,H] for slicing to minibatches.
            init_rnn_state = samples.agent.agent_info.prev_rnn_state[0]  # T=0.

        dual_policy_weights = None
        if self.dual_policy_weighting != 'none':
            if init_rnn_state is not None:
                # [B,N,H] --> [N,B,H] (for cudnn).
                init_rnn_state = buffer_method(init_rnn_state, "transpose", 0, 1)
                init_rnn_state = buffer_method(init_rnn_state, "contiguous")
                with torch.no_grad():
                    dist_info, dist_ext_info, dist_int_info, value, value_int, _rnn_state = self.agent(*agent_inputs, init_rnn_state)
            else:
                with torch.no_grad():
                    dist_info, dist_ext_info, dist_int_info, value, value_int = self.agent(*agent_inputs) # uses __call__ instead of step() because rnn state is included here
            with torch.no_grad():
                dist_og = self.agent.distribution
                if self.dual_policy_weighting == 'ext_first':
                    kl_scores = dist_og.kl(dist_ext_info, dist_int_info)
                    if self.dpw_formulation == 'inverse':
                        dual_policy_weights = 1/torch.clamp(kl_scores, min=0.00000000001, max=100000.0)
                    elif self.dpw_formulation == 'exp':
                        dual_policy_weights = torch.exp(-kl_scores)
                if self.dual_policy_weighting == 'int_first':
                    kl_scores = dist_og.kl(dist_int_info, dist_ext_info)
                    if self.dpw_formulation == 'inverse':
                        dual_policy_weights = 1/torch.clamp(kl_scores, min=0.00000000001, max=100000.0)
                    elif self.dpw_formulation == 'exp':
                        dual_policy_weights = torch.exp(-kl_scores)

        if hasattr(self.agent, "update_obs_rms"):
            self.agent.update_obs_rms(agent_inputs.observation)

        if self.curiosity_type != 'none':
            if hasattr(self.agent.model.curiosity_model, 'current_idx'):
                current_idx = self.agent.model.curiosity_model.current_idx
            else:
                current_idx = None
    
        # get idx.
        return_, return_int_, advantage, advantage_ext, advantage_int, valid, intrinsic_rewards = self.process_returns(samples, self.dual_value, dual_policy_weights)

        if self.curiosity_type in {'icm', 'micm', 'disagreement'}:
            agent_curiosity_inputs = IcmAgentCuriosityInputs(
                observation=samples.env.observation.clone(),
                next_observation=samples.env.next_observation.clone(),
                action=samples.agent.action.clone(),
                valid=valid
            )
            agent_curiosity_inputs = buffer_to(agent_curiosity_inputs, device=self.agent.device)
        elif self.curiosity_type == 'ndigo':
            agent_curiosity_inputs = NdigoAgentCuriosityInputs(
                observation=samples.env.observation.clone(),
                prev_actions=samples.agent.prev_action.clone(),
                actions=samples.agent.action.clone(),
                valid=valid
            )
            agent_curiosity_inputs = buffer_to(agent_curiosity_inputs, device=self.agent.device)
        elif 'rnd' in self.curiosity_type:
            agent_curiosity_inputs = RndAgentCuriosityInputs(
                next_observation=samples.env.next_observation.clone(),
                valid=valid
            )
            agent_curiosity_inputs = buffer_to(agent_curiosity_inputs, device=self.agent.device)
        elif self.curiosity_type == 'none':
            agent_curiosity_inputs = None
        loss_inputs = LossInputs(  # So can slice all.
            agent_inputs=agent_inputs,
            agent_curiosity_inputs=agent_curiosity_inputs,
            action=samples.agent.action,
            return_=return_,
            return_int_=return_int_,
            advantage=advantage,
            advantage_ext=advantage_ext,
            advantage_int=advantage_int,
            valid=valid,
            old_dist_info=samples.agent.agent_info.dist_info,
            old_dist_ext_info=samples.agent.agent_info.dist_ext_info,
            old_dist_int_info=samples.agent.agent_info.dist_int_info

        )

        if recurrent:
            # Leave in [B,N,H] for slicing to minibatches.
            init_rnn_state = samples.agent.agent_info.prev_rnn_state[0]  # T=0.

        T, B = samples.env.reward.shape[:2]

        opt_info = OptInfo(*([] for _ in range(len(OptInfo._fields))))

        # If recurrent, use whole trajectories, only shuffle B; else shuffle all.
        batch_size = B if self.agent.recurrent else T * B
        mb_size = batch_size // self.minibatches

        for _ in range(self.epochs):
            for idxs in iterate_mb_idxs(batch_size, mb_size, shuffle=True):
                T_idxs = slice(None) if recurrent else idxs % T
                B_idxs = idxs if recurrent else idxs // T
                self.optimizer.zero_grad()
                rnn_state = init_rnn_state[B_idxs] if recurrent else None


                # NOTE: if not recurrent, will lose leading T dim, should be OK.
                loss, pi_loss, pi_ext_loss, pi_int_loss, value_loss, value_int_loss, entropy_loss, entropy_ext_loss, entropy_int_loss, entropy, entropy_ext, entropy_int, perplexity, perplexity_ext, perplexity_int, curiosity_losses, utility_nw, kl_constraint = self.loss(*loss_inputs[T_idxs, B_idxs], rnn_state, indices=B_idxs)


                loss.backward()
                count = 0
                
                grad_norm = torch.nn.utils.clip_grad_norm_(self.agent.parameters(), self.clip_grad_norm)
                self.optimizer.step()
                
                # Tensorboard summaries
                opt_info.loss.append(loss.item())
                opt_info.pi_loss.append(pi_loss.item())
                opt_info.pi_ext_loss.append(pi_ext_loss.item())
                opt_info.pi_int_loss.append(pi_int_loss.item())
                opt_info.value_loss.append(value_loss.item())
                opt_info.value_int_loss.append(value_int_loss.item())
                opt_info.entropy_loss.append(entropy_loss.item())
                opt_info.entropy_ext_loss.append(entropy_ext_loss.item())
                opt_info.entropy_int_loss.append(entropy_int_loss.item())
                opt_info.utility_nw.append(utility_nw.item())
                opt_info.kl_constraint.append(kl_constraint.item())

                intrinsic = False
                if self.curiosity_type in {'icm', 'micm'}:
                    inv_loss, forward_loss = curiosity_losses
                    opt_info.inv_loss.append(inv_loss.item())
                    opt_info.forward_loss.append(forward_loss.item())
                    intrinsic = True
                elif self.curiosity_type == 'disagreement':
                    forward_loss = curiosity_losses
                    opt_info.forward_loss.append(forward_loss.item())
                    intrinsic = True
                elif self.curiosity_type == 'ndigo':
                    forward_loss = curiosity_losses
                    opt_info.forward_loss.append(forward_loss.item())
                    intrinsic = True
                elif 'rnd' in self.curiosity_type:
                    forward_loss = curiosity_losses
                    opt_info.forward_loss.append(forward_loss.item())
                    intrinsic = True
                if intrinsic:
                    opt_info.intrinsic_rewards.append(self.intrinsic_rewards.flatten())
                    opt_info.extint_ratio.append(self.extint_ratio.flatten())

                if self.normalize_extreward:
                    opt_info.normalized_extreward.append(self.normalized_extreward.flatten())
                if self.normalize_intreward:
                    opt_info.normalized_intreward.append(self.normalized_intreward.flatten())
                if self.rescale_extreward:
                    opt_info.rescaled_extreward.append(self.rescaled_extreward.flatten())
                if self.rescale_intreward:
                    opt_info.rescaled_intreward.append(self.rescaled_intreward.flatten())
                opt_info.entropy.append(entropy.item())
                opt_info.entropy_ext.append(entropy_ext.item())
                opt_info.entropy_int.append(entropy_int.item())
                opt_info.perplexity.append(perplexity.item())
                opt_info.perplexity_ext.append(perplexity_ext.item())
                opt_info.perplexity_int.append(perplexity_int.item())
                self.update_counter += 1
        

        opt_info.return_.append(torch.mean(return_.detach()).detach().clone().item())
        opt_info.return_int_.append(torch.mean(return_int_.detach()).detach().clone().item())
        opt_info.advantage.append(torch.mean(advantage.detach()).detach().clone().item())
        if self.dual_policy_weighting != 'none':
            opt_info.dual_policy_weights.append(dual_policy_weights.clone().data.numpy().flatten())
            opt_info.kl_scores.append(kl_scores.clone().data.numpy().flatten())
            opt_info.advantage_int_weighted.append(self.advantage_int_weighted.flatten())
        if self.dual_value:
            opt_info.advantage_ext.append(torch.mean(advantage_ext.detach()).detach().clone().item())
            opt_info.advantage_int.append(torch.mean(advantage_int.detach()).detach().clone().item())
        opt_info.valpred.append(torch.mean(samples.agent.agent_info.value.detach()).detach().clone().item())

        if self.linear_lr_schedule:
            self.lr_scheduler.step()
            self.ratio_clip = self._ratio_clip * (self.n_itr - itr) / self.n_itr

        if self.curiosity_decay:
            self.agent.model.curiosity_model.decay -= self.curiosity_decay_rate

        layer_info = dict() # empty dict to store model layer weights for tensorboard visualizations
        
        return opt_info, layer_info


    def loss(self, agent_inputs, agent_curiosity_inputs, action, return_, return_int_, advantage, advantage_ext, advantage_int, valid, old_dist_info,
            old_dist_ext_info=None, old_dist_int_info=None, init_rnn_state=None, indices=None):
        """
        Compute the training loss: policy_loss + value_loss + entropy_loss
        Policy loss: min(likelhood-ratio * advantage, clip(likelihood_ratio, 1-eps, 1+eps) * advantage)
        Value loss:  0.5 * (estimated_value - return) ^ 2
        Calls the agent to compute forward pass on training data, and uses
        the ``agent.distribution`` to compute likelihoods and entropies.  Valid
        for feedforward or recurrent agents.
        """
        if init_rnn_state is not None:
            # [B,N,H] --> [N,B,H] (for cudnn).
            init_rnn_state = buffer_method(init_rnn_state, "transpose", 0, 1)
            init_rnn_state = buffer_method(init_rnn_state, "contiguous")
            dist_info, dist_ext_info, dist_int_info, value, value_int, _rnn_state = self.agent(*agent_inputs, init_rnn_state)
        else:
            dist_info, dist_ext_info, dist_int_info, value, value_int = self.agent(*agent_inputs) # uses __call__ instead of step() because rnn state is included here

        # combined policy
        dist = self.agent.distribution
        ratio = dist.likelihood_ratio(action, old_dist_info=old_dist_info, new_dist_info=dist_info)
        surr_1 = ratio * advantage
        clipped_ratio = torch.clamp(ratio, 1. - self.ratio_clip, 1. + self.ratio_clip)
        surr_2 = clipped_ratio * advantage
        surrogate = torch.min(surr_1, surr_2)
        pi_loss = - valid_mean(surrogate, valid)

        value_error = 0.5 * (value - return_) ** 2
        value_loss = self.value_loss_coeff * valid_mean(value_error, valid)

        entropy = dist.mean_entropy(dist_info, valid)
        perplexity = dist.mean_perplexity(dist_info, valid)
        entropy_loss = - self.entropy_loss_coeff * entropy

        if self.dual_policy != 'default':
            # extrinsic policy
            dist_ext = self.agent.distribution_ext
            ratio_ext = dist_ext.likelihood_ratio(action, old_dist_info=old_dist_ext_info, new_dist_info=dist_ext_info)
            surr_1_ext = ratio_ext * advantage_ext
            clipped_ratio_ext = torch.clamp(ratio_ext, 1. - self.ratio_clip, 1. + self.ratio_clip)
            surr_2_ext = clipped_ratio_ext * advantage_ext
            surrogate_ext = torch.min(surr_1_ext, surr_2_ext)
            if self.importance_sample > 0.:
                ext_importance = dist_ext.likelihood_ratio(action, old_dist_info=dist_info, new_dist_info=dist_ext_info, stopgrad='both')
                ext_importance = torch.clamp(ext_importance, 1. - self.importance_sample, 1. + self.importance_sample).clone().detach()
                surrogate_ext *= ext_importance
            pi_ext_loss = - valid_mean(surrogate_ext, valid)

            entropy_ext = dist_ext.mean_entropy(dist_ext_info, valid)
            perplexity_ext = dist_ext.mean_perplexity(dist_ext_info, valid)
            entropy_ext_loss = - self.entropy_loss_coeff * entropy_ext

            # intrinsic policy
            if self.dual_policy_noint:
                pi_int_loss = torch.tensor([0.0])
                entropy_int = torch.tensor([0.0])
                perplexity_int = torch.tensor([0.0])
                entropy_int_loss = torch.tensor([0.0])
            else:
                dist_int = self.agent.distribution_int
                ratio_int = dist_int.likelihood_ratio(action, old_dist_info=old_dist_int_info, new_dist_info=dist_int_info)
                surr_1_int = ratio_int * advantage_int
                clipped_ratio_int = torch.clamp(ratio_int, 1. - self.ratio_clip, 1. + self.ratio_clip)
                surr_2_int = clipped_ratio_int * advantage_int
                surrogate_int = torch.min(surr_1_int, surr_2_int)
                pi_int_loss = - valid_mean(surrogate_int, valid)

                entropy_int = dist_int.mean_entropy(dist_int_info, valid)
                perplexity_int = dist_int.mean_perplexity(dist_int_info, valid)
                entropy_int_loss = - self.entropy_loss_coeff * entropy_int

        if self.dual_value:
            value_int_error = 0.5 * (value_int - return_int_) ** 2
            value_int_loss = self.value_loss_coeff * valid_mean(value_int_error, valid)
            loss = pi_loss + value_loss + value_int_loss + entropy_loss
            if self.dual_policy in {'combined', 'ext', 'int'}:
                loss += pi_ext_loss + entropy_ext_loss
                if self.dual_policy_noint == False:
                    loss += pi_int_loss + entropy_int_loss
        else:
            value_int_loss = torch.tensor([0.0])
            loss = pi_loss + value_loss + entropy_loss

        if self.utility_noworkers:
            explore_ratio = dist.likelihood_ratio(action, old_dist_info=dist_ext_info, new_dist_info=dist_info, stopgrad=self.util_detach)
            clipped_explore_ratio = torch.clamp(explore_ratio, min=1.0-self.util_clamp, max=1.0+self.util_clamp)
            utility_nw = clipped_explore_ratio * advantage_ext
            utility_nw = valid_mean(utility_nw, valid)
            if self.kl_clamp > 0.0:
                kl_constraint = torch.clamp(dist.kl(dist_info, dist_ext_info, detach=self.kl_detach), min=0.0, max=self.kl_clamp)
            else:
                kl_constraint = dist.kl(dist_info, dist_ext_info, detach=self.kl_detach)
            kl_constraint = - self.kl_lambda * valid_mean(kl_constraint, valid)
            loss += utility_nw
            loss += kl_constraint
        else:
            utility_nw = torch.tensor([0.0])
            kl_constraint = torch.tensor([0.0])

        if self.curiosity_type in {'icm', 'micm'}: 
            inv_loss, forward_loss = self.agent.curiosity_loss(self.curiosity_type, *agent_curiosity_inputs, indices=indices)
            loss += inv_loss
            loss += forward_loss
            curiosity_losses = (inv_loss, forward_loss)
        elif self.curiosity_type == 'disagreement':
            forward_loss = self.agent.curiosity_loss(self.curiosity_type, *agent_curiosity_inputs)
            loss += forward_loss
            curiosity_losses = (forward_loss)
        elif self.curiosity_type == 'ndigo':
            forward_loss = self.agent.curiosity_loss(self.curiosity_type, *agent_curiosity_inputs)
            loss += forward_loss
            curiosity_losses = (forward_loss)
        elif 'rnd' in self.curiosity_type:
            forward_loss = self.agent.curiosity_loss(self.curiosity_type, *agent_curiosity_inputs, indices=indices)
            loss += forward_loss
            curiosity_losses = (forward_loss)
        else:
            curiosity_losses = None

        if self.dual_value and self.dual_policy in {'combined', 'ext', 'int'}:
            pass  
        elif self.dual_value:
            pi_ext_loss = torch.tensor([0.0])
            pi_int_loss = torch.tensor([0.0])
            entropy_ext_loss = torch.tensor([0.0])
            entropy_ext = torch.tensor([0.0])
            perplexity_ext = torch.tensor([0.0])
            entropy_int_loss = torch.tensor([0.0])
            entropy_int = torch.tensor([0.0])
            perplexity_int = torch.tensor([0.0])
        else:
            pi_ext_loss = torch.tensor([0.0])
            pi_int_loss = torch.tensor([0.0])
            entropy_ext_loss = torch.tensor([0.0])
            entropy_ext = torch.tensor([0.0])
            perplexity_ext = torch.tensor([0.0])
            entropy_int_loss = torch.tensor([0.0])
            entropy_int = torch.tensor([0.0])
            perplexity_int = torch.tensor([0.0])
            value_int_loss = torch.tensor([0.0])
        
        return loss, pi_loss, pi_ext_loss, pi_int_loss, value_loss, value_int_loss, entropy_loss, entropy_ext_loss, entropy_int_loss, entropy, entropy_ext, entropy_int, perplexity, perplexity_ext, perplexity_int, curiosity_losses, utility_nw, kl_constraint




