import copy
import torch
from torch import nn
from torch.optim import SGD
from torchviz import make_dot
import numpy as np
from PIL import ImageColor, Image, ImageDraw, ImageFont

import networks
import tools

to_np = lambda x: x.detach().cpu().numpy()


class RewardEMA(object):
    """running mean and std"""

    def __init__(self, device, alpha=1e-2):
        self.device = device
        self.values = torch.zeros((2,)).to(device)
        self.alpha = alpha
        self.range = torch.tensor([0.05, 0.95]).to(device)

    def __call__(self, x):
        flat_x = torch.flatten(x.detach())
        x_quantile = torch.quantile(input=flat_x, q=self.range)
        self.values = self.alpha * x_quantile + (1 - self.alpha) * self.values
        scale = torch.clip(self.values[1] - self.values[0], min=1.0)
        offset = self.values[0]
        return offset.detach(), scale.detach()


class WorldModel(nn.Module):
    def __init__(self, step, config):
        super(WorldModel, self).__init__()
        self._step = step
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        self.encoder = networks.ConvEncoder(
            config.grayscale,
            config.cnn_depth,
            config.act,
            config.norm,
            config.encoder_kernels,
        )
        if config.size[0] == 64 and config.size[1] == 64:
            embed_size = (
                (64 // 2 ** (len(config.encoder_kernels))) ** 2
                * config.cnn_depth
                * 2 ** (len(config.encoder_kernels) - 1)
            )
        else:
            raise NotImplemented(f"{config.size} is not applicable now")
        self.dynamics = networks.RSSM(
            config.dyn_stoch,
            config.dyn_deter,
            config.dyn_hidden,
            config.dyn_input_layers,
            config.dyn_output_layers,
            config.dyn_rec_depth,
            config.dyn_shared,
            config.dyn_discrete,
            config.act,
            config.norm,
            config.dyn_mean_act,
            config.dyn_std_act,
            config.dyn_temp_post,
            config.dyn_min_std,
            config.dyn_cell,
            config.unimix_ratio,
            config.num_actions,
            embed_size,
            config.device,
        )
        self.heads = nn.ModuleDict()
        channels = 1 if config.grayscale else 3
        shape = (channels,) + config.size
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
        self.heads["image"] = networks.ConvDecoder(
            feat_size,  # pytorch version
            config.cnn_depth,
            config.act,
            config.norm,
            shape,
            config.decoder_kernels,
        )
        if config.reward_head == "twohot_symlog":
            self.heads["reward"] = networks.DenseHead(
                feat_size,  # pytorch version
                (255,),
                config.reward_layers,
                config.units,
                config.act,
                config.norm,
                dist=config.reward_head,
                outscale=0.0,
            )
        else:
            self.heads["reward"] = networks.DenseHead(
                feat_size,  # pytorch version
                [],
                config.reward_layers,
                config.units,
                config.act,
                config.norm,
                dist=config.reward_head,
                outscale=0.0,
            )
        if config.pred_discount:
            self.heads["discount"] = networks.DenseHead(
                feat_size,  # pytorch version
                [],
                config.discount_layers,
                config.units,
                config.act,
                config.norm,
                dist="binary",
            )
        if config.cost_head == "twohot_symlog":
            self.heads["cost"] = networks.DenseHead(
                feat_size,  # pytorch version
                (255,),
                config.cost_layers,
                config.units,
                config.act,
                config.norm,
                dist=config.cost_head,
            )

        for name in config.grad_heads:
            assert name in self.heads, name
        self._model_opt = tools.Optimizer(
            "model",
            self.parameters(),
            config.model_lr,
            config.opt_eps,
            config.grad_clip,
            config.weight_decay,
            opt=config.opt,
            use_amp=self._use_amp,
        )
        self._scales = dict(reward=config.reward_scale, discount=config.discount_scale, cost=config.cost_scale)

    def _train(self, data):
        # action (batch_size, batch_length, act_dim)
        # image (batch_size, batch_length, h, w, ch)
        # reward (batch_size, batch_length)
        # discount (batch_size, batch_length)
        data = self.preprocess(data)

        with tools.RequiresGrad(self):
            with torch.cuda.amp.autocast(self._use_amp):
                embed = self.encoder(data)
                post, prior = self.dynamics.observe(embed, data["action"])
                kl_free = tools.schedule(self._config.kl_free, self._step)
                kl_lscale = tools.schedule(self._config.kl_lscale, self._step)
                kl_rscale = tools.schedule(self._config.kl_rscale, self._step)
                kl_loss, kl_value, loss_lhs, loss_rhs = self.dynamics.kl_loss(
                    post, prior, self._config.kl_forward, kl_free, kl_lscale, kl_rscale
                )
                losses = {}
                likes = {}
                for name, head in self.heads.items():
                    grad_head = name in self._config.grad_heads
                    feat = self.dynamics.get_feat(post)
                    feat = feat if grad_head else feat.detach()
                    pred = head(feat)
                    like = pred.log_prob(data[name])
                    likes[name] = like                  
                    losses[name] = -torch.mean(like) * self._scales.get(name, 1.0)
                model_loss = sum(losses.values()) + kl_loss
            metrics = self._model_opt(model_loss, self.parameters())

        metrics.update({f"{name}_loss": to_np(loss) for name, loss in losses.items()})
        metrics["kl_free"] = kl_free
        metrics["kl_lscale"] = kl_lscale
        metrics["kl_rscale"] = kl_rscale
        metrics["loss_lhs"] = to_np(loss_lhs)
        metrics["loss_rhs"] = to_np(loss_rhs)
        metrics["kl"] = to_np(torch.mean(kl_value))
        with torch.cuda.amp.autocast(self._use_amp):
            metrics["prior_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(prior).entropy())
            )
            metrics["post_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(post).entropy())
            )
            context = dict(
                embed=embed,
                feat=self.dynamics.get_feat(post),
                kl=kl_value,
                postent=self.dynamics.get_dist(post).entropy(),
            )
        post = {k: v.detach() for k, v in post.items()}
        return post, context, metrics

    def preprocess(self, obs):
        obs = obs.copy()
        obs["image"] = torch.Tensor(obs["image"]) / 255.0 - 0.5
        # (batch_size, batch_length) -> (batch_size, batch_length, 1)
        obs["reward"] = torch.Tensor(obs["reward"]).unsqueeze(-1)
        obs["cost"] = torch.Tensor(obs["cost"]).unsqueeze(-1)
        if "discount" in obs:
            obs["discount"] *= self._config.discount
            # (batch_size, batch_length) -> (batch_size, batch_length, 1)
            obs["discount"] = torch.Tensor(obs["discount"]).unsqueeze(-1)
        obs = {k: torch.Tensor(v).to(self._config.device) for k, v in obs.items()}
        return obs

    def video_pred(self, data):
        data = self.preprocess(data)
        embed = self.encoder(data)

        states, _ = self.dynamics.observe(embed[:6, :5], data["action"][:6, :5])
        recon = self.heads["image"](self.dynamics.get_feat(states)).mode()[:6]
        reward_post = self.heads["reward"](self.dynamics.get_feat(states)).mode()[:6]
        init = {k: v[:, -1] for k, v in states.items()}
        prior = self.dynamics.imagine(data["action"][:6, 5:], init)
        openl = self.heads["image"](self.dynamics.get_feat(prior)).mode()
        reward_prior = self.heads["reward"](self.dynamics.get_feat(prior)).mode()
        # observed image is given until 5 steps
        model = torch.cat([recon[:, :5], openl], 1)
        truth = data["image"][:6] + 0.5
        model = model + 0.5
        error = (model - truth + 1.0) / 2.0

        return torch.cat([truth, model, error], 2)


class ImagBehavior(nn.Module):
    def __init__(self, config, world_model, stop_grad_actor=True, reward=None, start_lagrange = 1):
        super(ImagBehavior, self).__init__()
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        self._world_model = world_model
        self._stop_grad_actor = stop_grad_actor
        self._reward = reward

        budget = 25
        self.last_cost = None
        self.gamma_c = 0.995
        self.steps = 1000
        self.budget_undiscounted = budget
        self.budget = budget*(1 - self.gamma_c ** (1000)) / (1 - self.gamma_c)/(1000)
        self.raw_lag = torch.tensor([np.log(np.exp(start_lagrange)-1)], requires_grad=True, device=torch.device('cuda'), dtype=torch.float32)
        with torch.no_grad():
            self.lagrange = torch.nn.functional.softplus(self.raw_lag)

        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
        self.actor = networks.ActionHead(
            feat_size,  # pytorch version
            config.num_actions,
            config.actor_layers,
            config.units,
            config.act,
            config.norm,
            config.actor_dist,
            config.actor_init_std,
            config.actor_min_std,
            config.actor_max_std,
            config.actor_temp,
            outscale=1.0,
        )  # action_dist -> action_disc?
        if config.value_head == "twohot_symlog":
            self.value = networks.DenseHead(
                feat_size,  # pytorch version
                (255,),
                config.value_layers,
                config.units,
                config.act,
                config.norm,
                config.value_head,
                outscale=0.0,
            )
        else:
            self.value = networks.DenseHead(
                feat_size,  # pytorch version
                [],
                config.value_layers,
                config.units,
                config.act,
                config.norm,
                config.value_head,
                outscale=0.0,
            )
        if config.cost_head == "twohot_symlog":
            self.cost = networks.DenseHead(
                feat_size,  # pytorch version
                (255,),
                config.value_layers,
                config.units,
                config.act,
                config.norm,
                config.cost_head,
                outscale=0.0,
            )
        if config.slow_value_target:
            self._slow_value = copy.deepcopy(self.value)
            self._updates = 0
        kw = dict(wd=config.weight_decay, opt=config.opt, use_amp=self._use_amp)
        self._actor_opt = tools.Optimizer(
            "actor",
            self.actor.parameters(),
            config.actor_lr,
            config.ac_opt_eps,
            config.actor_grad_clip,
            **kw,
        )
        self._value_opt = tools.Optimizer(
            "value",
            self.value.parameters(),
            config.value_lr,
            config.ac_opt_eps,
            config.value_grad_clip,
            **kw,
        )
        self._cost_opt = tools.Optimizer(
            "cost",
            self.cost.parameters(),
            config.cost_lr,
            config.ac_opt_eps,
            config.cost_grad_clip,
            **kw,
        )
        self.optim_lagrange = SGD([self.raw_lag], lr=2e-4)
        if self._config.reward_EMA:
            self.reward_ema = RewardEMA(device=self._config.device)
        if self._config.cost_EMA:
            self.cost_ema = RewardEMA(device=self._config.device)

    def _train(
        self,
        start,
        objective=None,
        action=None,
        reward=None,
        cost_objective=None,
        imagine=None,
        tape=None,
        repeats=None,
    ):
        objective = objective or self._reward
        self._update_slow_target()
        metrics = {}

        with tools.RequiresGrad(self.actor):
            with torch.cuda.amp.autocast(self._use_amp):
                imag_feat, imag_state, imag_action = self._imagine(
                    start, self.actor, self.cost, self._config.imag_horizon, repeats, self.last_cost
                )
                # print('imag feat dim', imag_feat.size())
                # print('image state dim', [len(values) for values in imag_state.values()])
                # print(imag_state['logit'].size())
                # print('imag action dim', imag_action.size())
                reward = objective(imag_feat, imag_state, imag_action)
                cost = cost_objective(imag_feat, imag_state, imag_action)
                _lag, lag_loss = self.update_lag()
                if _lag is not None:
                    metrics["lagrange"] = to_np(_lag)
                    metrics["lagrange_loss"] = to_np(lag_loss)
                self.last_cost = cost
                actor_ent = self.actor(imag_feat).entropy()
                state_ent = self._world_model.dynamics.get_dist(imag_state).entropy()
                # this target is not scaled
                # slow is flag to indicate whether slow_target is used for lambda-return
                reward_target, weights, reward_base = self._compute_reward_target(
                    imag_feat, imag_state, imag_action, reward, cost, actor_ent, state_ent
                )
                cost_target, cost_weights, cost_base = self._compute_cost_target(
                    imag_feat, imag_state, imag_action, reward, cost, actor_ent, state_ent
                )
                actor_loss, mets = self._compute_actor_loss(
                    imag_feat,
                    imag_state,
                    imag_action,
                    reward_target,
                    cost_target,
                    actor_ent,
                    state_ent,
                    weights,
                    reward_base,
                    cost_base,
                )
                metrics.update(mets)
                value_input = imag_feat
                cost_input = imag_feat

        with tools.RequiresGrad(self.value):
            with torch.cuda.amp.autocast(self._use_amp):
                value = self.value(value_input[:-1].detach())
                reward_target = torch.stack(reward_target, dim=1)
                # (time, batch, 1), (time, batch, 1) -> (time, batch)
                value_loss = -value.log_prob(reward_target.detach())
                slow_target = self._slow_value(value_input[:-1].detach())
                if self._config.slow_value_target:
                    value_loss = value_loss - value.log_prob(
                        slow_target.mode().detach()
                    )
                if self._config.value_decay:
                    value_loss += self._config.value_decay * value.mode()
                # (time, batch, 1), (time, batch, 1) -> (1,)
                value_loss = torch.mean(weights[:-1] * value_loss[:, :, None])

        with tools.RequiresGrad(self.cost):
            with torch.cuda.amp.autocast(self._use_amp):
                cost_value = self.cost(cost_input[:-1].detach())
                cost_target = torch.stack(cost_target, dim=1)
                # (time, batch, 1), (time, batch, 1) -> (time, batch)
                cost_loss = -cost_value.log_prob(cost_target.detach())
                slow_target = self._slow_value(cost_input[:-1].detach())
                if self._config.slow_value_target:
                    cost_loss = cost_loss - cost_value.log_prob(
                        slow_target.mode().detach()
                    )
                if self._config.value_decay:
                    cost_loss += self._config.value_decay * cost_value.mode()
                # (time, batch, 1), (time, batch, 1) -> (1,)
                cost_loss = torch.mean(weights[:-1] * cost_loss[:, :, None])

        metrics.update(tools.tensorstats(value.mode(), "value"))
        metrics.update(tools.tensorstats(cost_value.mode(), "cost_critic"))
        metrics.update(tools.tensorstats(reward_target, "target"))
        metrics.update(tools.tensorstats(cost_target, "cost_target"))
        metrics.update(tools.tensorstats(reward, "imag_reward"))
        metrics.update(tools.tensorstats(cost, "imag_cost"))
        metrics.update(tools.tensorstats(imag_action, "imag_action"))
        metrics["actor_ent"] = to_np(torch.mean(actor_ent))
        metrics["cost_critic_loss"] = to_np(cost_loss)
        with tools.RequiresGrad(self):
            metrics.update(self._actor_opt(actor_loss, self.actor.parameters()))
            metrics.update(self._value_opt(value_loss, self.value.parameters()))
            metrics.update(self._cost_opt(cost_loss, self.cost.parameters()))
        return imag_feat, imag_state, imag_action, weights, metrics

    def _imagine(self, start, policy, cost_critic, horizon, repeats=None, last_cost=None):
        # horizon: 15
        # start = dict(stoch, deter, logit)
        # start["stoch"] (16, 63, 32, 32)
        # start["deter"] (16, 63, 512)
        # start["logit"] (16, 63, 32, 32)
        dynamics = self._world_model.dynamics
        if repeats:
            raise NotImplemented("repeats is not implemented in this version")
        flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
        start = {k: flatten(v) for k, v in start.items()}

        def step(prev, _):
            state, _, _ = prev
            feat = dynamics.get_feat(state)
            inp = feat.detach() if self._stop_grad_actor else feat
            action = policy(inp).sample()
            succ = dynamics.img_step(state, action, sample=self._config.imag_sample)
            return succ, feat, action

        feat = 0 * dynamics.get_feat(start)
        action = policy(feat).mode()
        # Is this action deterministic or stochastic?
        # action = policy(feat).sample()
        succ, feats, actions = tools.static_scan(
            step, [torch.arange(horizon)], (start, feat, action)
        )
        states = {k: torch.cat([start[k][None], v[:-1]], 0) for k, v in succ.items()}
        if repeats:
            raise NotImplemented("repeats is not implemented in this version")

        return feats, states, actions

    def update_lag(self):
        if self.last_cost is None:
            return None, None
        last_cost = self.last_cost.detach()
        cost_baseline = self.budget_undiscounted/self.steps * torch.flatten(last_cost).size()[0]
        loss_lag = (torch.nn.functional.softplus(self.raw_lag)/torch.nn.functional.softplus(self.raw_lag).detach() * (cost_baseline-torch.abs(torch.sum(torch.flatten(last_cost)))))
        # torch.abs used to improve stability at the start of training
        self.optim_lagrange.zero_grad()
        loss_lag.backward(retain_graph=False)
        torch.nn.utils.clip_grad_norm_(self.raw_lag, 1000)
        self.optim_lagrange.step()
        with torch.no_grad():
            self.lagrange = torch.nn.functional.softplus(self.raw_lag)
        return self.lagrange, loss_lag


    def _compute_cost_target(
        self, imag_feat, imag_state, imag_action, reward, cost, actor_ent, state_ent
    ):
        if "discount" in self._world_model.heads:
            inp = self._world_model.dynamics.get_feat(imag_state)
            discount = self._world_model.heads["discount"](inp).mean
        else:
            discount = self._config.discount * torch.ones_like(cost)
        if self._config.future_entropy and self._config.actor_entropy() > 0:
            cost += self._config.actor_entropy() * actor_ent
        if self._config.future_entropy and self._config.actor_state_entropy() > 0:
            cost += self._config.actor_state_entropy() * state_ent
        value = self.cost(imag_feat).mode()
        # value(15, 960, ch)
        # action(15, 960, ch)
        # discount(15, 960, ch)
        target = tools.lambda_return(
            cost[:-1],
            value[:-1],
            discount[:-1],
            bootstrap=value[-1],
            lambda_=self._config.discount_lambda,
            axis=0,
        )
        weights = torch.cumprod(
            torch.cat([torch.ones_like(discount[:1]), discount[:-1]], 0), 0
        ).detach()
        return target, weights, value[:-1]

    def _compute_reward_target(
        self, imag_feat, imag_state, imag_action, reward, cost, actor_ent, state_ent
    ):
        if "discount" in self._world_model.heads:
            inp = self._world_model.dynamics.get_feat(imag_state)
            discount = self._world_model.heads["discount"](inp).mean
        else:
            discount = self._config.discount * torch.ones_like(reward)
        if self._config.future_entropy and self._config.actor_entropy() > 0:
            reward += self._config.actor_entropy() * actor_ent
        if self._config.future_entropy and self._config.actor_state_entropy() > 0:
            reward += self._config.actor_state_entropy() * state_ent
        value = self.value(imag_feat).mode()
        # value(15, 960, ch)
        # action(15, 960, ch)
        # discount(15, 960, ch)
        target = tools.lambda_return(
            reward[:-1],
            value[:-1],
            discount[:-1],
            bootstrap=value[-1],
            lambda_=self._config.discount_lambda,
            axis=0,
        )
        weights = torch.cumprod(
            torch.cat([torch.ones_like(discount[:1]), discount[:-1]], 0), 0
        ).detach()
        return target, weights, value[:-1]

    def _compute_actor_loss(
        self,
        imag_feat,
        imag_state,
        imag_action,
        reward_target,
        cost_target,
        actor_ent,
        state_ent,
        weights,
        reward_base,
        cost_base,
    ):
        lam = 1
        metrics = {}
        inp = imag_feat.detach() if self._stop_grad_actor else imag_feat
        policy = self.actor(inp)
        actor_ent = policy.entropy()
        # Q-val for actor is not transformed using symlog
        reward_target = torch.stack(reward_target, dim=1)
        cost_target = torch.stack(cost_target, dim=1)
        if self._config.reward_EMA:
            offset, scale = self.reward_ema(reward_target)
            normed_reward_target = (reward_target - offset) / scale
            normed_reward_base = (reward_base - offset) / scale
            reward_adv = normed_reward_target - normed_reward_base
            metrics["reward_adv"] = to_np(torch.mean(weights[:-1]*reward_adv))
            metrics.update(tools.tensorstats(normed_reward_target, "normed_reward_target"))
            metrics.update(tools.tensorstats(normed_reward_base, "normed_reward_base"))
            values = self.reward_ema.values
            metrics["reward_EMA_005"] = to_np(values[0])
            metrics["reward_EMA_095"] = to_np(values[1])
        
        if self._config.cost_EMA:
            offset, scale = self.cost_ema(cost_target)
            normed_cost_target = (cost_target - offset) / scale
            normed_cost_base = (cost_base - offset) / scale
            cost_adv = normed_cost_target - normed_cost_base
            metrics["cost_adv"] = to_np(torch.mean(weights[:-1]*cost_adv))
            metrics.update(tools.tensorstats(normed_cost_target, "normed_cost_target"))
            metrics.update(tools.tensorstats(normed_cost_base, "normed_cost_base"))
            values = self.cost_ema.values
            metrics["cost_EMA_005"] = to_np(values[0])
            metrics["cost_EMA_095"] = to_np(values[1])
        else: 
            cost_adv = cost_target - cost_base
            metrics["cost_adv"] = to_np(torch.mean(weights[:-1]*cost_adv))

        if self._config.imag_gradient == "dynamics":
            actor_target = reward_adv
        elif self._config.imag_gradient == "reinforce":
            actor_target = (
                policy.log_prob(imag_action)[:-1][:, :, None]
                * (reward_target - self.value(imag_feat[:-1]).mode()).detach()
            )
        elif self._config.imag_gradient == "both":
            actor_target = (
                policy.log_prob(imag_action)[:-1][:, :, None]
                * (reward_target - self.value(imag_feat[:-1]).mode()).detach()
            )
            mix = self._config.imag_gradient_mix()
            #print(mix)
            actor_target = mix * reward_target + (1 - mix) * actor_target
            metrics["imag_gradient_mix"] = mix
        else:
            raise NotImplementedError(self._config.imag_gradient)
        
        if not self._config.future_entropy and (self._config.actor_entropy() > 0):
            actor_entropy = self._config.actor_entropy() * actor_ent[:-1][:, :, None]
            actor_target += actor_entropy
            #cost_adv += actor_entropy
            metrics["actor_entropy"] = to_np(torch.mean(actor_entropy))
        if not self._config.future_entropy and (self._config.actor_state_entropy() > 0):
            state_entropy = self._config.actor_state_entropy() * state_ent[:-1]
            actor_target += state_entropy
            metrics["actor_state_entropy"] = to_np(torch.mean(state_entropy))
        if type(self.lagrange) == torch.Tensor:
            l = self.lagrange.detach().item()
        else:
            l = self.lagrange
        actor_loss = -torch.mean(weights[:-1] * actor_target) + l * torch.mean(weights[:-1] * cost_adv)
        metrics["actor_loss"] = to_np(actor_loss)
        return actor_loss, metrics

    def _update_slow_target(self):
        if self._config.slow_value_target:
            if self._updates % self._config.slow_target_update == 0:
                mix = self._config.slow_target_fraction
                for s, d in zip(self.value.parameters(), self._slow_value.parameters()):
                    d.data = mix * s.data + (1 - mix) * d.data
            self._updates += 1
