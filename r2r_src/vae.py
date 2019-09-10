import torch
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


from param import args
from speaker import Speaker
from agent import Seq2SeqAgent
import model
from distribution import Normal


# following the Experimental Details in SeCTER
class StateEncoder(nn.Module):
    '''
    Encode state sequence as z distribution(dist for short)
    The encoder is a two-layer bidirectional-LSTM with 300 hidden units
    We mean-pool over LSTM outputs over time before a linear transformation
    '''
    def __init__(self, latent_dim, hidden_dim=300, bidirectional=True, num_layers=2, min_var=1e-4):
        self.hidden_dim = hidden_dim
        if bidirectional:
            print("Using Bidir in EncoderLSTM")
        self.num_directions = 2 if bidirectional else 1
        self.num_layers = num_layers
        input_size = embedding_size
        self.lstm = nn.LSTM(input_size, hidden_size, self.num_layers,
                            batch_first=True, dropout=dropout_ratio,
                            bidirectional=bidirectional)
        self.mean_network = model.MLP(2*hidden_dim, latent_dim)
        self.log_var_network = model.MLP(2*hidden_dim, latent_dim)
        self.min_log_var = Variable(np.log(np.array([min_var])).astype(np.float32))

    def init_state(self, inputs):
        ''' Initialize to zero cell states and hidden states.'''
        batch_size = inputs.size(0)
        h0 = Variable(torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size
        ), requires_grad=False)
        c0 = Variable(torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size
        ), requires_grad=False)

        return h0.cuda(), c0.cuda()

    def forward(self, inputs, lengths):
        ''' Expects input state sequence as (batch, seq_len). Also requires a
            list of lengths for dynamic batching. '''
        h0, c0 = self.init_state(inputs)
        packed_embeds = pack_padded_sequence(inputs, lengths, batch_first=True)
        enc_h, (enc_h_t, enc_c_t) = self.lstm(packed_embeds, (h0, c0))
        # if self.num_directions == 2:    # The size of enc_h_t is (num_layers * num_directions, batch, hidden_size)
        #     h_t = torch.cat((enc_h_t[-1], enc_h_t[-2]), 1)
        #     c_t = torch.cat((enc_c_t[-1], enc_c_t[-2]), 1)
        # else:
        #     h_t = enc_h_t[-1]
        #     c_t = enc_c_t[-1] # (batch, hidden_size)
        ctx, _ = pad_packed_sequence(enc_h, batch_first=True)
        # mean pooling
        ctx_mean = torch.mean(ctx, dim=1)
        mean, log_var = self.mean_network(ctx_mean), self.log_var_network(ctx_mean)
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
    def __init__(self, obs_dim, latent_dim, path_len, hidden_dim=256, bidirectional=False, num_layers=1):
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.path_len = path_len
        self.hidden_dim = hidden_dim
        self.num_directions = 2 if bidirectional else 1
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, self.num_layers,
                batch_first=True, dropout=dropout_ratio,
                bidirectional=bidirectional)
        self.mean_network = model.MLP(hidden_dim, obs_dim)
        self.log_var_network = model.Parameter(obs_dim, init=np.log(0.1))

    def init_state(self, inputs):
        ''' Initialize to zero cell states and hidden states.'''
        batch_size = inputs.size(0)
        h0 = Variable(torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size
        ), requires_grad=False)
        c0 = Variable(torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size
        ), requires_grad=False)

        return h0.cuda(), c0.cuda()

    def step(self, x):
        # Torch uses (path_len, bs, input_dim) for recurrent input
        # Return (1, bs, output_dim)
        assert x.size()[0] == 1, "Path len must be 1"
        out = self.lstm(x).squeeze(0)
        mean, log_var = self.mean_network(out), self.log_var_network(out)
        #log_var = torch.max(self.min_log_var, log_var)
        return mean.unsqueeze(0), log_var.unsqueeze(0)

    def forward(self, z, s0):
        '''
        z: (bs, input_dim)
        s0: initial obs (bs, obs_dim)
        '''
        h0, c0 = self.init_state(z)
        packed_embeds = pack_padded_sequence(z, lengths, batch_first=True)
        if s0 is None:
            s0 = Variable(torch.zeros(bs, self.output_dim)) # init input
        z = z.unsqueeze(0) # (1, bs, latent_dim)
        s0 = s0.unsqueeze(0) # (1, bs, obs_dim)
        means, log_vars = [], []
        x = s0
        for i in range(self.path_len):
            x = torch.cat([x, z], -1)
            mean, log_var = self.step(x)
            x = mean
            #x = Variable(torch.randn(mean.size())) * torch.exp(log_var) + mean
            means.append(mean.squeeze(dim=0))
            log_vars.append(log_var.squeeze(dim=0))
        means = torch.stack(means, 1).view(bs, -1)
        log_vars = torch.stack(log_vars, 1).view(bs, -1)
        dist = Normal(means, log_var=log_vars)
        return dist

class PolicyDecoder(nn.Module):
    '''Single MLP predict action by (z, state)'''
    # TODO: auto encoder for compress observation
    def __init__(self, obs_dim, latent_dim, action_dim)
        self.policy_network = MLP(obs_dim+latent_dim, action_dim,
            hidden_sizes=(400, 300, 200), hidden_act=nn.ReLU, final_act=nn.Softmax)
        self.action_dim = action_dim

    def forward(self, x):
        prob = self.prob_network(x)
        dist = Categorical(prob)
        return dist

class BaseVAE():
    def __init__(self, obs_dim, latent_dim, path_len, action_dim, lr):
        self.obs_dim
        self.latent_dim
        self.path_len
        self.action_dim
        self.encoder = StateEncoder(latent_dim)
        self.decoder = StateDecoder(obs_dim, latent_dim, path_len)
        self.policy = PolicyDecoder(obs_dim, latent_dim, action_dim)
        self.encoder_optimizer = args.optimizer(self.encoder.parameters(), lr=args.lr)
        self.decoder_optimizer = args.optimizer(self.decoder.parameters(), lr=args.lr)
        self.policy_optimizer = args.optimizer(self.policy.parameters(), lr=args.lr)

    def forward(self, batch_state, batch_act, train=True):
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
        (img_feats, can_feats), lengths = self.from_shortest_path()

        x = batch_state
        z_dist = self.encode(Variable(x))
        z = z_dist.sample()
        y_dist = self.decode(x, z)
        # y = x[:, self.step_dim:].contiguous() # TODO: what is x[:, self.step_dim:]
        # TODO: debug reshape
        import pdb; pdb.set_trace()
        xview = x.view((x.size()[0], -1, self.obs_dim)).clone()
        zexpand = z.unsqueeze(1).expand(*xview.size()[:2], z.size()[-1])
        xz = torch.cat(( Variable(xview), zexpand), -1)
        xz_view = xz.view((-1, xz.size()[-1]))
        dist = self.policy.forward(xz_view)
        act_view = batch_act.view((-1, self.action_dim))
        bcloss = -dist.log_likelihood(Variable(act_view))

        mse = torch.pow(y_dist.mle - x, 2).mean(-1)
        neg_ll = - y_dist.log_likelihood(x) / self.decoder.path_len
        kl = z_dist.kl(self.unit_n)
        return mse, neg_ll, kl, bcloss, z_dist

    def forward(self):
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
        (img_feats, can_feats), lengths = self.from_shortest_path()


    def encode(self, x):
        # TODO: debug reshape
        x = x.view((x.size()[0], -1, self.obs_dim)).clone()
        x = x.transpose(0, 1)
        return self.encoder.forward(x)

    def decode(self, x, z):
        if self.decoder.recurrent():
            # initial_input = x[:, :self.obs_dim].contiguous().clone()
            initial_input = x.clone()
            output = self.decoder.forward(z, initial_input=Variable(initial_input))
            return output
        else:
            return self.decoder.forward(z)

    def train(self, dataset, tok, test_dataset=None, max_epochs=10000, save_step=1000, print_step=1, plot_step=1, record_stats=False):
        # speaker = Speaker(train_env, listner, tok)
        # listner = Seq2SeqAgent(train_env, "", tok, args.maxAction)
        # TODO: train a VAE for state-state&inst-inst
        # TODO: train a policy decoder
        if args.speaker is not None:
            print("Load the speaker from %s." % args.speaker)
            speaker.load(args.speaker)
        for epoch in range(1, max_epochs + 1):
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
            self.policy.step()
            # stats = self.train_epoch(dataset, epoch)
            # test = self.train_epoch(test_dataset, epoch, train=False)
            # for k, v in test.items():
            #     stats['V ' + k] = v
            # stats['Test RL'] = self.test_pd(test_dataset)

            # if epoch % print_step == 0:
            #     with logger.prefix('itr #%d | ' % epoch):
            #         self.print_diagnostics(stats)

            # if epoch % plot_step == 0:
            #     self.plot_compare(dataset, epoch)
            #     self.plot_interp(dataset, epoch)
            #     self.plot_compare(test_dataset, epoch, save_dir='test')
            #     self.plot_random(dataset, epoch)

            # if epoch % save_step == 0 and logger.get_snapshot_dir() is not None:
            #     self.save(logger.get_snapshot_dir() + '/snapshots/', epoch)

            # if record_stats:
            #     with logger.prefix('itr #%d | ' % epoch):
            #         self.log_diagnostics(stats)
            #         logger.dump_tabular()

        return stats



    def save(self, snapshot_dir, itr):
        import os
        if not os.path.exists(snapshot_dir):
            os.makedirs(snapshot_dir)
        torch.save(self.encoder.state_dict_lst(), snapshot_dir + '/encoder_%d.pkl' %itr)
        torch.save(self.policy.state_dict_lst(), snapshot_dir + '/policy_%d.pkl' %itr)
        torch.save(self.decoder.state_dict_lst(), snapshot_dir + '/decoder_%d.pkl' % itr)

    def load(self, snapshot_dir, itr):
        self.policy.load_state_dict(torch.load(snapshot_dir + '/policy_%d.pkl' %itr))
        self.encoder.load_state_dict(torch.load(snapshot_dir + '/encoder_%d.pkl' %itr))
        self.decoder.load_state_dict(torch.load(snapshot_dir + '/decoder_%d.pkl' % itr))

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
        first_feat = np.zeros((len(obs), self.feature_size+args.angle_feat_size), np.float32)
        for i, ob in enumerate(obs):
            first_feat[i, -args.angle_feat_size:] = utils.angle_feature(ob['heading'], ob['elevation'])
        first_feat = torch.from_numpy(first_feat).cuda()
        while not ended.all():
            if viewpoints is not None:
                for i, ob in enumerate(obs):
                    viewpoints[i].append(ob['viewpoint'])
            img_feats.append(self.listener._feature_variable(obs))
            teacher_action = self._teacher_action(obs, ended)
            teacher_action = teacher_action.cpu().numpy()
            for i, act in enumerate(teacher_action):
                if act < 0 or act == len(obs[i]['candidate']):  # Ignore or Stop
                    teacher_action[i] = -1                      # Stop Action
            can_feats.append(self._candidate_variable(obs, teacher_action))
            self.make_equiv_action(teacher_action, obs)
            length += (1 - ended)
            ended[:] = np.logical_or(ended, (teacher_action == -1))
            obs = self.env._get_obs()
        img_feats = torch.stack(img_feats, 1).contiguous()  # batch_size, max_len, 36, 2052
        can_feats = torch.stack(can_feats, 1).contiguous()  # batch_size, max_len, 2052
        if get_first_feat:
            return (img_feats, can_feats, first_feat), length
        else:
            return (img_feats, can_feats), length

#    def loss_generator(self, dataset):
#        kl_weight = self.compute_kl_weight(0)
#        for batch_idx, (batch, target) in enumerate(dataset.dataloader):
#            mse, neg_ll, kl, bcloss, z_dist = self.forward(batch)
#            mse = mse.mean(0)
#            neg_ll = neg_ll.mean(0)
#            kl = kl.mean(0)
#            bcloss = bcloss.mean(0)
#            if self.loss_type == 'mse':
#                loss = self.vae_loss_weight * mse + kl_weight * kl + bcloss * self.bc_weight
#            elif self.loss_type == 'll':
#                loss = self.vae_loss_weight * neg_ll + kl_weight * kl + bcloss * self.bc_weight
#            else:
#                raise Exception('undefined loss type: '+self.loss_type+", expected mse or ll")
#
#            stats = {
#                'MSE': mse,
#                'Total Loss': loss,
#                'LL': neg_ll,
#                'KL Loss': kl,
#                'BC Loss': bcloss
#            }
#            yield loss, stats

    # def train_epoch(self, dataset, epoch=0, train=True, max_steps=1e99):
    #     full_stats = dict([('MSE',0), ('Total Loss', 0), ('LL', 0), ('KL Loss', 0),
    #                        ('BC Loss', 0)])
    #     n_batch = 0
    #     self.optimizer.zero_grad()
    #     for loss, stats in self.loss_generator(dataset):
    #         if train:
    #             loss.backward()
    #             self.optimizer.step()

    #         for k in stats.keys():
    #             full_stats[k] += get_numpy(stats[k])[0]
    #         n_batch += 1
    #         if n_batch >= max_steps:
    #             break
    #         self.optimizer.zero_grad()

    #     for k in full_stats.keys():
    #         full_stats[k] /= n_batch

    #     return full_stats


