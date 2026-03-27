# double_dqn.py

import torch
import torch.nn.functional as F


def ddqn_update(
    online_net,
    target_net,
    optimizer,
    batch,
    gamma=0.99
):

    states, actions, rewards, next_states, dones = batch

    q_values = online_net(states)

    q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():

        next_actions = online_net(next_states).argmax(1)

        next_q_values = target_net(next_states)

        next_q = next_q_values.gather(
            1,
            next_actions.unsqueeze(1)
        ).squeeze(1)

        target = rewards + gamma * next_q * (1 - dones)

    loss = F.smooth_l1_loss(q_value, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()