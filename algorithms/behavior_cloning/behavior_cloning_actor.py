import torch
import torch.nn as nn

from ..utils.mlp import MLPBase
from ..utils.gru import GRULayer
from ..utils.act import ACTLayer
from ..utils.utils import check


class BehaviorCloningActor(nn.Module):
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        super(BehaviorCloningActor, self).__init__()
        # 同样的网络结构
        self.use_recurrent_policy = args.use_recurrent_policy
        self.base = MLPBase(obs_space, args.hidden_size, args.activation_id, args.use_feature_normalization)
        input_size = self.base.output_size
        if args.use_recurrent_policy:
            self.rnn = GRULayer(input_size, args.recurrent_hidden_size, args.recurrent_hidden_layers)
            input_size = self.rnn.output_size
        self.act = ACTLayer(act_space, input_size, args.act_hidden_size, args.activation_id, args.gain)
        self.to(device)

    def forward(self, obs, rnn_states, masks, deterministic=False):
        obs = check(obs).to(torch.float32)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        actor_features = self.base(obs)
        if self.use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)
        actions, action_log_probs = self.act(actor_features, deterministic=True)
        return actions, action_log_probs, rnn_states

