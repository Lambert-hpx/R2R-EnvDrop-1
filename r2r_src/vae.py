import torch
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

from param import args
from speaker import Speaker
from agent import Seq2SeqAgent
import model
from distribution import Normal,Categorical
import utils

writer = SummaryWriter(log_dir=args.save_path+"/log")

# following the Experimental Details in SeCTER
class StateEncoder(nn.Module):
    '''
    Encode state sequence as z distribution(dist for short)
    The encoder is a two-layer bidirectional-LSTM with 300 hidden units
    We mean-pool over LSTM outputs over time before a linear transformation
    '''
    def __init__(self, obs_dim, latent_dim, hidden_dim=600, bidirectional=True, num_layers=2, min_var=1e-4, dropout_ratio=args.dropout):
        super(StateEncoder, self).__init__()
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        if bidirectional:
            print("Using Bidir in EncoderLSTM")
        self.num_directions = 2 if bidirectional else 1
        self.num_layers = num_layers
        self.lstm = nn.LSTM(obs_dim, hidden_dim // self.num_directions, self.num_layers, batch_first=True, dropout=dropout_ratio, bidirectional=bidirectional)
        self.attention_layer = model.SoftDotAttention(hidden_dim, obs_dim)
        self.post_lstm = nn.LSTM(hidden_dim, hidden_dim // self.num_directions, self.num_layers, batch_first=True, dropout=dropout_ratio, bidirectional=bidirectional)
        self.mean_network = model.MLP(hidden_dim, latent_dim)
        self.log_var_network = model.MLP(hidden_dim, latent_dim)
        # self.min_log_var = Variable(np.log(np.array([min_var])).astype(np.float32))
        self.min_log_var = torch.from_numpy(np.log(np.array([min_var])).astype(np.float32)).cuda()

    def forward(self, action_embeds, feature, lengths):
        ''' Expects input state sequence as (batch, seq_len). Also requires a
            list of lengths for dynamic batching. '''
        x = action_embeds
        ctx, _ = self.lstm(x)
        # Att and Handle with the shape
        batch_size, max_length, _ = ctx.size()
        x, _ = self.attention_layer(                        # Attend to the feature map
            ctx.contiguous().view(-1, self.hidden_dim),    # (batch, length, hidden) --> (batch x length, hidden)
            feature.view(batch_size * max_length, -1, self.obs_dim),        # (batch, length, # of images, obs_dim) --> (batch x length, # of images, obs_dim)
        )
        x = x.view(batch_size, max_length, -1)
        # Post LSTM layer
        x, _ = self.post_lstm(x)

        # need mean pooling for path_len
        x = torch.mean(x, dim=1)
        mean, log_var = self.mean_network(x), self.log_var_network(x)
        log_var = torch.max(self.min_log_var, log_var)
        dist = Normal(mean=mean, log_var=log_var)
        return dist

class StateDecoder(nn.Module):
    '''
    Decodes z as state sequence
    Single-layer LSTM, 256 hidden units
    '''
    # TODO: trajectories length T=19, plan over K=2048
    # random latent sequences. horizon H = 380/950 ?
    # Here obs_dim=2048+128(geo_feat_size)
    def __init__(self, obs_dim, latent_dim, view_num, path_len=2, hidden_dim=256, bidirectional=False, num_layers=1, dropout_ratio=args.dropout):
        super(StateDecoder, self).__init__()
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.path_len = path_len
        self.hidden_dim = hidden_dim
        self.num_directions = 2 if bidirectional else 1
        self.num_layers = num_layers
        self.view_num = view_num
        self.view_atten = model.SelfAttention(obs_dim+latent_dim, hidden_dim)
        self.fc1 = nn.Linear(obs_dim+latent_dim, hidden_dim)
        self.relu1 = nn.LeakyReLU()
        self.h_size = 1
        if bidirectional:
            self.h_size += 1
        self.h_size *= num_layers
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers,
                batch_first=True, dropout=dropout_ratio,
                bidirectional=bidirectional)
        self.mean_network = model.MLP(hidden_dim, obs_dim*view_num)
        self.log_var_network = model.Parameter(obs_dim*view_num, init=np.log(0.1))

    def init_state(self, bs):
        '''Initialize to zero cell states and hidden states.'''
        self.hidden = (Variable(torch.zeros(self.h_size, bs, self.hidden_dim).cuda()),
                       Variable(torch.zeros(self.h_size, bs, self.hidden_dim)).cuda())
        self.lengths = Variable(torch.ones(bs)).cuda()

    def step(self, x):
        # Torch uses (bs, 1, latent_dim) for recurrent input
        # Return (bs, 1, obs_dim)
        assert len(x.size())==3, "Input shape must be (bs, 1, latent_dim)"
        assert x.size()[1] == 1, "Path len must be 1"
        seq_x = pack_padded_sequence(x, self.lengths, batch_first=True)
        packed_output, self.hidden = self.lstm(seq_x, self.hidden)
        output, input_sizes = pad_packed_sequence(packed_output, batch_first=True)
        mean, log_var = self.mean_network(output), self.log_var_network(output)
        return mean.squeeze(1), log_var.squeeze(1)

    def forward(self, z, s0):
        '''
        z: (bs, input_dim)
        s0: initial obs (bs, view_dim, obs_dim) # view_dim = 36
        '''
        bs = z.size()[0]
        self.init_state(bs)
        if s0 is None:
            s0 = Variable(torch.zeros(bs, self.obs_dim)) # init input
        (bs, view_dim, obs_dim) = s0.shape
        z = z.unsqueeze(1).repeat(1, view_dim, 1) # (bs, view_dim, latent_dim)
        means, log_vars = [], []
        x = s0
        # NOTE: decode states by step 1 since we have fixed start point s0
        for i in range(self.path_len):
            # TODO: layernorm instead of concat: x=LayerNorm(A x) + LayerNorm(B z)
            x = torch.cat([x, z], -1) # TODO: balance energy between x,z
            x, attn = self.view_atten(x)
            x = self.relu1(self.fc1(x)) # TODO: relu is necessary or not ?
            x = x.unsqueeze(1) # bs, len, dim
            mean, log_var = self.step(x)
            x = mean.view(bs, self.view_num, -1)
            means.append(mean)
            log_vars.append(log_var)
        means = torch.stack(means, 1).view(bs, -1)
        log_vars = torch.stack(log_vars, 1).view(bs, -1)
        dist = Normal(means, log_var=log_vars)
        return dist

class PolicyDecoder(nn.Module):
    '''Single MLP predict action by (z, state)'''
    # TODO: auto encoder for compress observation
    def __init__(self, obs_dim, latent_dim, view_dim, path_len, hidden_dim=256):
        super(PolicyDecoder, self).__init__()
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.view_dim = view_dim
        self.path_len = path_len
        self.feat_attn_layer = model.SoftDotAttention(latent_dim, obs_dim)
        # self.cand_attn_layer = model.SoftDotAttention(latent_dim, obs_dim)
        self.cand_attn_layer = model.SoftDotAttention(latent_dim, obs_dim)

    def forward(self, z, cand_feats, cand_masks=None):
        (bs, path_len, cand_num, obs_dim) = cand_feats.size()
        z = z.unsqueeze(1).repeat(1, path_len, 1).view(bs*path_len, -1) # (bs*path_len, cand_num, latent_dim)
        cand_feats = cand_feats.view(bs*path_len, cand_num, obs_dim)
        _, prob = self.cand_attn_layer(z, cand_feats)
        dist = Categorical(prob)
        return dist
        # (bs, path_len, view_dim, obs_dim) = img_feats.size()
        # x = img_feats.view(-1, view_dim, obs_dim)
        # z = z.unsqueeze(1).unsqueeze(1).repeat(1, path_len, view_dim, 1).view(bs*path_len, view_dim, -1) # (bs*path_len, view_dim, latent_dim)
        # x = torch.cat([x, z], -1) # TODO: balance energy between x,z
        # x, attn = self.view_atten(x) # (bs*path_len, hidden_dim)
        # x = self.relu1(self.fc1(x))
        # (bs, path_len, candidate_num, obs_dim) = candidate_feats.size()
        # candidate_feats = candidate_feats.view(bs*path_len, candidate_num, obs_dim)
        # candidate_feats = self.relu2(self.fc2(candidate_feats))
        # if candidate_masks is not None:
        #     candidate_masks = candidate_masks.view(bs*path_len, candidate_num)
        # # _, logit = self.pred_layer(x, candidate_feats, mask=candidate_masks, output_prob=False) # output logit rather than prob
        # _, prob = self.pred_layer(x, candidate_feats, mask=candidate_masks)
        # dist = Categorical(prob)
        # # return dist
        # return prob

class BaseVAE(nn.Module):
    env_actions = {
        'left': (0,-1, 0), # left
        'right': (0, 1, 0), # right
        'up': (0, 0, 1), # up
        'down': (0, 0,-1), # down
        'forward': (1, 0, 0), # forward
        '<end>': (0, 0, 0), # <end>
        '<start>': (0, 0, 0), # <start>
        '<ignore>': (0, 0, 0)  # <ignore>
    }
    def __init__(self, env, tok, obs_dim, latent_dim, loss_type="mse", view_num=args.view_num, path_len=2):
        super(BaseVAE, self).__init__()
        self.env = env
        self.tok = tok
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.path_len = path_len
        self.view_num = view_num
        self.loss_type=loss_type
        self.encoder = StateEncoder(obs_dim, latent_dim)
        self.decoder = StateDecoder(obs_dim, latent_dim, view_num, path_len=path_len)
        self.policy = PolicyDecoder(obs_dim, latent_dim, view_num, path_len)
        self.unit_n = Normal(Variable(torch.zeros(1, latent_dim)).cuda(),
                             log_var=Variable(torch.zeros(1, latent_dim)).cuda())
        self.encoder_optimizer = args.optimizer(self.encoder.parameters(), lr=args.lr)
        self.decoder_optimizer = args.optimizer(self.decoder.parameters(), lr=args.lr)
        self.policy_optimizer = args.optimizer(self.policy.parameters(), lr=args.lr)
        self.vae_loss_weight = args.vae_loss_weight
        self.vae_kl_weight = args.vae_kl_weight
        # self.bc_weight = 100
        self.vae_bc_weight = args.vae_bc_weight
        self.iter = None # training iteration indicator

    def forward(self, train=True):
        """
        train state-decoder with
        """
        if train:
            self.encoder.train()
            self.decoder.train()
            self.policy.train()
        else:
            self.encoder.eval()
            self.decoder.eval()
            self.policy.eval()
        obs = self.env._get_obs()
        batch_size = len(obs)
        (img_feats, can_feats, teacher_actions, candidate_feats, candidate_masks), lengths = self.from_shortest_path()
        # check mask for all non-zero values
        assert bool((torch.sum(candidate_feats*candidate_masks.unsqueeze(3).repeat(1,1,1,2176).float())==0).cpu().numpy())
        z_dist = self.encoder(can_feats, img_feats, lengths)
        h_t = torch.zeros(1, batch_size, args.rnn_dim).cuda()
        c_t = torch.zeros(1, batch_size, args.rnn_dim).cuda()
        z = z_dist.sample()
        s0 = img_feats[:,0,:,:] # init s0
        # forward state decoder
        state_dist = self.decoder.forward(z, s0.detach())
        # forward policy decoder
        # NOTE: don't predict last action(use STOP as label may misleading)
        action_dist = self.policy.forward(z, candidate_feats[:,:-1,:,:].contiguous(), candidate_masks[:,:-1,:].contiguous()) # dist shape(bs*path_len, num_classes)
        # compute loss
        y_state = img_feats[:,1:,:,:].view(batch_size,-1) # gt for state decoder
        # behavior clone loss
        y_action = teacher_actions[:,:-1,:].contiguous().view(batch_size*self.path_len, -1)
        bcloss = self.vae_bc_weight * torch.sum(-action_dist.log_likelihood(y_action.float()))
        # mse loss for state decoder
        mse = self.vae_loss_weight * torch.sum(torch.pow(state_dist.mle - y_state, 2).mean(-1))
        neg_ll = self.vae_loss_weight * torch.sum(-state_dist.log_likelihood(y_state) / self.path_len)
        kl = self.vae_kl_weight * torch.sum(z_dist.kl(self.unit_n))
        # print(bcloss, mse, neg_ll, kl)
        writer.add_scalar('Loss/bcloss', bcloss.detach().cpu().numpy(), self.iter)
        writer.add_scalar('Loss/mse', mse.detach().cpu().numpy(), self.iter)
        writer.add_scalar('Loss/kl', kl.detach().cpu().numpy(), self.iter)
        if self.loss_type == 'mse':
            loss =  mse + kl + bcloss
        elif self.loss_type=="ll":
            loss =  neg_ll + kl + bcloss
        else:
            raise Exception("loss type undefined")
        # return mse, neg_ll, kl, bcloss, z_dist
        return loss

    def train(self, test_dataset=None, max_iter=args.iters, save_step=1000, print_step=1, plot_step=1, record_stats=False):
        # speaker = Speaker(train_env, listner, tok)
        # listner = Seq2SeqAgent(train_env, "", tok, args.maxAction)
        # TODO: train a VAE for state-state&inst-inst
        # TODO: train a policy decoder, RL
        if args.speaker is not None:
            print("Load the speaker from %s." % args.speaker)
            speaker.load(args.speaker)
        for _iter in range(1, max_iter + 1):
            self.iter = _iter
            self.env.reset()
            self.encoder_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()
            self.policy_optimizer.zero_grad()
            loss = self.forward(train=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm(self.encoder.parameters(), 40.)
            torch.nn.utils.clip_grad_norm(self.encoder.parameters(), 40.)
            torch.nn.utils.clip_grad_norm(self.policy.parameters(), 40.)
            self.encoder_optimizer.step()
            self.decoder_optimizer.step()
            self.policy_optimizer.step()

            if _iter % save_step == 0:
                print("Save models at iter %d" % _iter)
                self.save(_iter)
            # TODO: test every epoch

        return

    def save(self, itr):
        torch.save(self.encoder.state_dict(), args.save_path + '/encoder_%d.pkl' %itr)
        torch.save(self.decoder.state_dict(), args.save_path + '/decoder_%d.pkl' % itr)
        torch.save(self.policy.state_dict(), args.save_path + '/policy_%d.pkl' %itr)

    def load(self, itr):
        self.encoder.load_state_dict(torch.load(args.save_path + '/encoder_%d.pkl' %itr))
        self.decoder.load_state_dict(torch.load(args.save_path + '/decoder_%d.pkl' %itr))
        self.policy.load_state_dict(torch.load(args.save_path + '/policy_%d.pkl' %itr))

    def from_shortest_path(self, viewpoints=None, get_first_feat=False):
        """
        :param viewpoints: [[], [], ....(batch_size)]. Only for dropout viewpoint
        :param get_first_feat: whether output the first feat
        :return:
        """
        obs = self.env._get_obs()
        ended = np.array([False] * len(obs)) # Indices match permuation of the model, not env
        length = np.zeros(len(obs), np.int64)
        img_feats = []
        can_feats = []
        teacher_actions = []
        teacher_actions_1h = []
        candidate_feats = []
        candidate_masks = []
        first_feat = np.zeros((len(obs), self.obs_dim), np.float32)
        for i, ob in enumerate(obs):
            first_feat[i, -args.angle_feat_size:] = utils.angle_feature(ob['heading'], ob['elevation'])
        first_feat = torch.from_numpy(first_feat).cuda()
        while not ended.all():
            if viewpoints is not None:
                for i, ob in enumerate(obs):
                    viewpoints[i].append(ob['viewpoint'])
            teacher_action = self._teacher_action(obs, ended)
            teacher_action = teacher_action.cpu().numpy()
            # TODO: why last teacher action not -1
            teacher_actions.append(teacher_action.copy())
            candidate_length = [len(ob['candidate']) + 1 for ob in obs]       # +1 is for the end
            candidate_feat = np.zeros((len(obs), max(candidate_length), self.obs_dim))
            # NOTE: The candidate_feat at len(ob['candidate']) is the feature for the END, which is zero in my implementation
            for i, ob in enumerate(obs):
                for j, c in enumerate(ob['candidate']):
                    candidate_feat[i, j, :] = c['feature']
            candidate_feats.append(torch.Tensor(candidate_feat).cuda())
            candidate_masks.append(utils.length2mask(candidate_length))
            img_feats.append(self._feature_variable(obs))
            for i, act in enumerate(teacher_action):
                if act < 0 or act == len(obs[i]['candidate']):  # Ignore or Stop
                    teacher_action[i] = -1                      # Stop Action
            can_feats.append(self._candidate_variable(obs, teacher_action))
            self.make_equiv_action(teacher_action, obs)
            length += (1 - ended)
            ended[:] = np.logical_or(ended, (teacher_action == -1))
            obs = self.env._get_obs()
            # TODO: heading random ?
            # TODO: policy decoder behavior clone
            # TODO: state decoder mse
            # TODO: state decoder weight = 0 ?

        assert len(teacher_actions)==len(candidate_feats)==len(candidate_masks)
        _max=0
        for i in range(len(candidate_feats)):
            _max = max(_max, candidate_feats[i].shape[1])
        shape_list = np.array(candidate_feats[0].shape)
        shape_list[1] = 1
        feat_pad_vec = torch.zeros(tuple(shape_list)).cuda()
        shape_list = np.array(candidate_masks[0].shape)
        shape_list[1] = 1
        mask_pad_vec = torch.ones(tuple(shape_list)).bool().cuda()
        for i in range(len(candidate_feats)):
            diff = _max - candidate_feats[i].shape[1]
            diff2 = _max - candidate_masks[i].shape[1]
            assert diff == diff2
            if diff > 0:
                candidate_feats[i] = torch.cat([candidate_feats[i], feat_pad_vec.repeat(1,diff,1)], dim=1)
                candidate_masks[i] = torch.cat([candidate_masks[i], mask_pad_vec.repeat(1,diff)], dim=1)
            # convert teacher actions to one-hot vectors
            teacher_actions_1h.append(torch.nn.functional.one_hot(torch.LongTensor(teacher_actions[i]), num_classes=_max).cuda())

        img_feats = torch.stack(img_feats, 1).contiguous()  # batch_size, max_len, 36, 2052
        can_feats = torch.stack(can_feats, 1).contiguous()  # batch_size, max_len, 2052
        teacher_actions_1h = torch.stack(teacher_actions_1h, 1).contiguous()
        candidate_feats = torch.stack(candidate_feats, 1).contiguous()
        candidate_masks = torch.stack(candidate_masks, 1).contiguous()
        if get_first_feat:
            return (img_feats, can_feats, first_feat), length
        else:
            return (img_feats, can_feats, teacher_actions_1h, candidate_feats, candidate_masks), length
        # NOTE: the last teacher_actions are all STOP(verified as below)
        # torch.all(torch.eq(
        #     torch.nn.functional.one_hot(
        #         torch.sum(1-candidate_masks[:,-1,:].long(), dim=1)-1, num_classes=14
        #         ),teacher_actions_1h[:,-1,:]
        #     )

    def _candidate_variable(self, obs, actions):
        candidate_feat = np.zeros((len(obs), self.obs_dim), dtype=np.float32)
        for i, (ob, act) in enumerate(zip(obs, actions)):
            if act == -1:  # Ignore or Stop --> Just use zero vector as the feature
                pass
            else:
                c = ob['candidate'][act]
                candidate_feat[i, :] = c['feature'] # Image feat
        return torch.from_numpy(candidate_feat).cuda()

    def make_equiv_action(self, a_t, perm_obs, perm_idx=None, traj=None):
        def take_action(i, idx, name):
            if type(name) is int:       # Go to the next view
                self.env.env.sims[idx].makeAction(name, 0, 0)
            else:                       # Adjust
                self.env.env.sims[idx].makeAction(*self.env_actions[name])
            state = self.env.env.sims[idx].getState()
            if traj is not None:
                traj[i]['path'].append((state.location.viewpointId, state.heading, state.elevation))
        if perm_idx is None:
            perm_idx = range(len(perm_obs))
        for i, idx in enumerate(perm_idx):
            action = a_t[i]
            if action != -1:            # -1 is the <stop> action
                select_candidate = perm_obs[i]['candidate'][action]
                src_point = perm_obs[i]['viewIndex']
                trg_point = select_candidate['pointId']
                src_level = (src_point) // 12   # The point idx started from 0
                trg_level = (trg_point) // 12
                while src_level < trg_level:    # Tune up
                    take_action(i, idx, 'up')
                    src_level += 1
                    # print("UP")
                while src_level > trg_level:    # Tune down
                    take_action(i, idx, 'down')
                    src_level -= 1
                    # print("DOWN")
                while self.env.env.sims[idx].getState().viewIndex != trg_point:    # Turn right until the target
                    take_action(i, idx, 'right')
                    # print("RIGHT")
                    # print(self.env.env.sims[idx].getState().viewIndex, trg_point)
                assert select_candidate['viewpointId'] == \
                       self.env.env.sims[idx].getState().navigableLocations[select_candidate['idx']].viewpointId
                take_action(i, idx, select_candidate['idx'])

    def _feature_variable(self, obs):
        ''' Extract precomputed features into variable. '''
        features = np.empty((len(obs), args.views, self.obs_dim), dtype=np.float32)
        for i, ob in enumerate(obs):
            features[i, :, :] = ob['feature']   # Image feat
        return Variable(torch.from_numpy(features), requires_grad=False).cuda()

    def _teacher_action(self, obs, ended):
        """
        Extract teacher actions into variable.
        :param obs: The observation.
        :param ended: Whether the action seq is ended
        :return:
        """
        a = np.zeros(len(obs), dtype=np.int64)
        for i, ob in enumerate(obs):
            if ended[i]:                                            # Just ignore this index
                a[i] = args.ignoreid
            else:
                for k, candidate in enumerate(ob['candidate']):
                    if candidate['viewpointId'] == ob['teacher']:   # Next view point
                        a[i] = k
                        break
                else:   # Stop here
                    assert ob['teacher'] == ob['viewpoint']         # The teacher action should be "STAY HERE"
                    a[i] = len(ob['candidate'])
        return torch.from_numpy(a).cuda()
