import torch


def calc_rewards(t, s, term):
    # calcualate rewards for any that just finished
    # it will calculate three reward values:
    # agent 1 (as proporition of max agent 1), agent 2 (as proportion of max agent 2), prosocial (as proportion of max prosocial)
    # in the non-prosocial setting, we need all three:
    # - first two for training
    # - next one for evaluating Table 1, in the paper
    # in the prosocial case, we'll skip calculating the individual agent rewards, possibly/probably

    # assert prosocial, 'not tested for not prosocial currently'

    agent = t % 2
    batch_size = term.size()[0]
    utility = s.utilities[:, agent]
    type_constr = torch.cuda if s.pool.is_cuda else torch
    rewards_batch = type_constr.FloatTensor(batch_size, 3).fill_(0)  # each row is: {one, two, combined}
    if t == 0:
        # on first timestep theres no actual proposal yet, so score zero if terminate
        return rewards_batch

    reward_eligible_mask = term.view(batch_size).clone().byte()
    if reward_eligible_mask.max() == 0:
        # if none of them accepted proposal, by terminating
        return rewards_batch

    exceeded_pool, _ = ((s.last_proposal - s.pool) > 0).max(1)
    if exceeded_pool.max() > 0:
        reward_eligible_mask[exceeded_pool.nonzero().long().view(-1)] = 0
        if reward_eligible_mask.max() == 0:
            # all eligible ones exceeded pool
            return rewards_batch

    proposer = 1 - agent
    accepter = agent
    proposal = torch.zeros(batch_size, 2, 3).long()
    proposal[:, proposer] = s.last_proposal
    proposal[:, accepter] = s.pool - s.last_proposal
    max_utility, _ = s.utilities.max(1)

    reward_eligible_idxes = reward_eligible_mask.nonzero().long().view(-1)
    for b in reward_eligible_idxes:
        raw_rewards = torch.FloatTensor(2).fill_(0)
        for i in range(2):
            raw_rewards[i] = s.utilities[b, i].cpu().dot(proposal[b, i].cpu())

        scaled_rewards = torch.FloatTensor(3).fill_(0)

        # we always calculate the prosocial reward
        actual_prosocial = raw_rewards.sum()
        available_prosocial = max_utility[b].cpu().dot(s.pool[b].cpu())
        if available_prosocial != 0:
            scaled_rewards[2] = actual_prosocial / available_prosocial

        for i in range(2):
            max_agent = s.utilities[b, i].cpu().dot(s.pool[b].cpu())
            if max_agent != 0:
                scaled_rewards[i] = raw_rewards[i] / max_agent

        rewards_batch[b] = scaled_rewards
    return rewards_batch
