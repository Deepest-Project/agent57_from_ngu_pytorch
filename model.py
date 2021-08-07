import torch
import torch.nn as nn
import torch.nn.functional as F

from config import config


class R2D2(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(R2D2, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.lstm = nn.LSTM(input_size=num_inputs, hidden_size=config.hidden_size, batch_first=True)
        self.fc = nn.Linear(config.hidden_size, 128)
        self.fc_adv = nn.Linear(128 + 1, num_outputs)
        self.fc_val = nn.Linear(128 + 1, 1)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, x, hidden, beta):
        # x [batch_size,s equence_length, num_inputs]
        batch_size = x.size()[0]
        sequence_length = x.size()[1]
        out, hidden = self.lstm(x, hidden)  # out = [batch_size, config.hidden_size]

        out = F.relu(self.fc(out))  # ( batch_size, config.hidden_size)

        # out = torch.cat([out, torch.tensor(beta, dtype=torch.float32, device=self.device).reshape([out, 1, 1])], dim=2)
        out = torch.cat([out, torch.ones([*out.shape[:-1], 1]) * beta], dim=-1)
        adv = self.fc_adv(out)
        adv = adv.view(batch_size, sequence_length, self.num_outputs)
        val = self.fc_val(out)
        val = val.view(batch_size, sequence_length, 1)

        qvalue = val + (adv - adv.mean(dim=-1, keepdim=True))

        return qvalue, hidden

    @classmethod
    def get_td_error(cls, online_net, target_net, batch, lengths, beta):
        """
        batch.shape = [B]
        batch.state = [B, eps_len, *observation.shape]
        batch.action
        batch.reward
        lengths.shape = [B, 1]
        """

        def slice_burn_in(item):
            return item[:, config.burn_in_length:, :]

        batch_size = torch.stack(batch.state).size()[0]
        states = torch.stack(batch.state).view(batch_size, config.sequence_length, online_net.num_inputs)
        next_states = torch.stack(batch.next_state).view(batch_size, config.sequence_length, online_net.num_inputs)
        actions = torch.stack(batch.action).view(batch_size, config.sequence_length, -1).long()
        rewards = torch.stack(batch.reward).view(batch_size, config.sequence_length, -1)
        masks = torch.stack(batch.mask).view(batch_size, config.sequence_length, -1)
        steps = torch.stack(batch.step).view(batch_size, config.sequence_length, -1)
        rnn_state = torch.stack(batch.rnn_state).view(batch_size, config.sequence_length, 2, -1)

        [h0, c0] = rnn_state[:, 0, :, :].transpose(0, 1)
        h0 = h0.unsqueeze(0).detach()
        c0 = c0.unsqueeze(0).detach()

        [h1, c1] = rnn_state[:, 1, :, :].transpose(0, 1)
        h1 = h1.unsqueeze(0).detach()
        c1 = c1.unsqueeze(0).detach()

        pred, _ = online_net(states, (h0, c0), beta)
        next_pred, _ = target_net(next_states, (h1, c1), beta)

        next_pred_online, _ = online_net(next_states, (h1, c1), beta)

        pred = slice_burn_in(pred)
        next_pred = slice_burn_in(next_pred)
        actions = slice_burn_in(actions)
        rewards = slice_burn_in(rewards)
        masks = slice_burn_in(masks)
        steps = slice_burn_in(steps)
        next_pred_online = slice_burn_in(next_pred_online)

        pred = pred.gather(2, actions)

        _, next_pred_online_action = next_pred_online.max(2)

        target = rewards + masks * pow(config.gamma, steps) * next_pred.gather(2, next_pred_online_action.unsqueeze(2))

        td_error = pred - target.detach()

        for idx, length in enumerate(lengths):
            td_error[idx][length - config.burn_in_length:][:] = 0

        return td_error  # [B, 1]

    @classmethod
    def train_model(cls, online_net, target_net, optimizer, batch, lengths, beta):
        td_error = cls.get_td_error(online_net, target_net, batch, lengths, beta)

        loss = pow(td_error, 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss, td_error

    def get_action(self, state, hidden, beta):
        state = state.unsqueeze(0).unsqueeze(0)

        qvalue, hidden = self.forward(state, hidden, beta)

        _, action = torch.max(qvalue, 2)
        return action.numpy()[0][0], hidden


class R2D2_agent57:
    def __init__(self, num_inputs, num_outputs):
        self.R2D2_int = R2D2(num_inputs, num_outputs)
        self.R2D2_ext = R2D2(num_inputs, num_outputs)

    def forward(self, x, hidden, beta):
        q_int = self.R2D2_int.forward(x, hidden, beta)
        q_ext = self.R2D2_ext.forward(x, hidden, beta)
        q_final = R2D2_agent57.h_function(beta * R2D2_agent57.h_inv(q_int) + R2D2_agent57.h_inv(q_ext))
        return q_final

    @classmethod
    def h_function(cls, z, epsilon=0.001):
        return torch.sign(z) * (torch.sqrt(torch.abs(z) + 1) - 1) + epsilon * z

    @classmethod
    def h_inv(cls, z, epsilon=0.001):
        return torch.sign(z)((torch.sqrt(1 + 4 * epsilon * (torch.abs(z) + 1 + epsilon)) - 1) / (2 * epsilon) - 1)

    @classmethod
    def get_td_error(cls, online_net, target_net, batch, lengths, beta):
        """
        batch.shape = [B]
        batch.state = [B, eps_len, *observation.shape]
        batch.action
        batch.reward
        lengths.shape = [B, 1]
        """

        def slice_burn_in(item):
            return item[:, config.burn_in_length:, :]

        batch_size = torch.stack(batch.state).size()[0]
        states = torch.stack(batch.state).view(batch_size, config.sequence_length, online_net.num_inputs)
        next_states = torch.stack(batch.next_state).view(batch_size, config.sequence_length, online_net.num_inputs)
        actions = torch.stack(batch.action).view(batch_size, config.sequence_length, -1).long()
        rewards = torch.stack(batch.reward).view(batch_size, config.sequence_length, -1)
        masks = torch.stack(batch.mask).view(batch_size, config.sequence_length, -1)
        steps = torch.stack(batch.step).view(batch_size, config.sequence_length, -1)
        rnn_state = torch.stack(batch.rnn_state).view(batch_size, config.sequence_length, 2, -1)

        [h0, c0] = rnn_state[:, 0, :, :].transpose(0, 1)
        h0 = h0.unsqueeze(0).detach()
        c0 = c0.unsqueeze(0).detach()

        [h1, c1] = rnn_state[:, 1, :, :].transpose(0, 1)
        h1 = h1.unsqueeze(0).detach()
        c1 = c1.unsqueeze(0).detach()

        pred, _ = online_net(states, (h0, c0), beta)
        next_pred, _ = target_net(next_states, (h1, c1), beta)

        next_pred_online, _ = online_net(next_states, (h1, c1), beta)

        pred = slice_burn_in(pred)
        next_pred = slice_burn_in(next_pred)
        actions = slice_burn_in(actions)
        rewards = slice_burn_in(rewards)
        masks = slice_burn_in(masks)
        steps = slice_burn_in(steps)
        next_pred_online = slice_burn_in(next_pred_online)

        pred = pred.gather(2, actions)

        _, next_pred_online_action = next_pred_online.max(2)

        target = rewards + masks * pow(config.gamma, steps) * next_pred.gather(2, next_pred_online_action.unsqueeze(2))

        td_error = pred - target.detach()

        for idx, length in enumerate(lengths):
            td_error[idx][length - config.burn_in_length:][:] = 0

        return td_error  # [B, 1]

    @classmethod
    def train_model(cls, online_net, target_net, optimizer, batch, lengths, beta):
        td_error = cls.get_td_error(online_net, target_net, batch, lengths, beta)

        loss = pow(td_error, 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss, td_error

    def get_action(self, state, hidden, beta):
        state = state.unsqueeze(0).unsqueeze(0)

        qvalue, hidden = self.forward(state, hidden, beta)

        _, action = torch.max(qvalue, 2)
        return action.numpy()[0][0], hidden
