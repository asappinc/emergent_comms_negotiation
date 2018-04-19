import torch
import numpy as np
import math
import ecn
from pytest import approx


import sampling
import rewards_lib


def test_rewards_t0():
    t = 0
    batch_size = 128
    torch.manual_seed(123)
    np.random.seed(123)
    s = ecn.State(**sampling.generate_batch(batch_size))
    agent = 0 if t  % 2 == 0 else 1
    term = torch.from_numpy(np.random.choice(2, batch_size)).long()
    rewards = rewards_lib.calc_rewards(s=s, t=t, term=term)
    assert rewards.size() == (batch_size, 3)
    assert rewards.abs().sum() == 0


def test_rewards_t1():
    t = 1
    batch_size = 97
    torch.manual_seed(123)
    np.random.seed(123)
    s = ecn.State(**sampling.generate_batch(batch_size))
    agent = 0 if t  % 2 == 0 else 1
    term = torch.from_numpy(np.random.choice(2, batch_size)).long()
    rewards = rewards_lib.calc_rewards(s=s, t=t, term=term)
    # print('alive_games', alive_games)
    for b in range(batch_size):
        # game = alive_games[b]
        assert rewards[b].tolist() == [0, 0, 0] or term[b] == 1


def test_single_game_noterm():
    t = 1
    batch_size = 1
    torch.manual_seed(123)
    np.random.seed(123)
    s = ecn.State(**sampling.generate_batch(batch_size))
    agent = 0 if t  % 2 == 0 else 1
    term = torch.ByteTensor([0])
    rewards = rewards_lib.calc_rewards(s=s, t=t, term=term)
    assert rewards[0].tolist() == [0, 0, 0]


def test_single_game_term_ideal():
    t = 1
    batch_size = 1
    torch.manual_seed(123)
    np.random.seed(123)
    s = ecn.State(**sampling.generate_batch(batch_size))

    s.pool = torch.LongTensor([[3, 7, 2]])
    s.utilities = torch.LongTensor([[[5,4,3], [3,4,5]]])
    # last proposal means agent 0's, and we are now on agent 1, who is accepintg it
    s.last_proposal = torch.LongTensor([[3, 7, 0]])

    agent = 0 if t  % 2 == 0 else 1
    term = torch.ByteTensor([1])
    rewards = rewards_lib.calc_rewards(s=s, t=t, term=term)
    assert rewards[0,0] == approx((3 * 5 + 7 * 4) / ( 3 * 5 + 7 * 4 + 2 * 3))
    assert rewards[0,1] == approx((2 * 5) / ( 3 * 3 + 7 * 4 + 2 * 5))
    assert rewards[0,2] == 1.0


def test_single_game_term_ideal2():
    t = 1
    batch_size = 1
    torch.manual_seed(123)
    np.random.seed(123)
    s = ecn.State(**sampling.generate_batch(batch_size))

    s.pool = torch.LongTensor([[3, 7, 2]])
    s.utilities = torch.LongTensor([[[5,4,3], [3,4,5]]])
    # last proposal means agent 0's, and we are now on agent 1, who is accepintg it
    s.last_proposal = torch.LongTensor([[3, 0, 0]])

    agent = 0 if t  % 2 == 0 else 1
    term = torch.ByteTensor([1])
    rewards = rewards_lib.calc_rewards(s=s, t=t, term=term)

    assert rewards[0,0] == approx((3 * 5) / ( 3 * 5 + 7 * 4 + 2 * 3))
    assert rewards[0,1] == approx((7 * 4 + 2 * 5) / ( 3 * 3 + 7 * 4 + 2 * 5))

    assert rewards[0,2] == 1.0


def test_single_game_term_ideal3():
    t = 1
    batch_size = 1
    torch.manual_seed(123)
    np.random.seed(123)
    s = ecn.State(**sampling.generate_batch(batch_size))

    s.pool = torch.LongTensor([[3, 7, 2]])
    s.utilities = torch.LongTensor([[[5,4,3], [3,4,5]]])
    # last proposal means agent 0's, and we are now on agent 1, who is accepintg it
    s.last_proposal = torch.LongTensor([[3, 2, 0]])

    agent = 0 if t  % 2 == 0 else 1
    term = torch.ByteTensor([1])
    rewards = rewards_lib.calc_rewards(s=s, t=t, term=term)

    assert rewards[0,0] == approx((3 * 5 + 4 * 2) / (3 * 5 + 7 * 4 + 2 * 3))
    assert rewards[0,1] == approx((5 * 4 + 2 * 5) / (3 * 3 + 7 * 4 + 2 * 5))

    assert rewards[0,2] == 1.0


def test_single_game_term_nonideal1():
    t = 1
    batch_size = 1
    torch.manual_seed(123)
    np.random.seed(123)
    s = ecn.State(**sampling.generate_batch(batch_size))

    s.pool = torch.LongTensor([[3, 7, 2]])
    s.utilities = torch.LongTensor([[[5,4,3], [3,4,5]]])
    # last proposal means agent 0's, and we are now on agent 1, who is accepintg it
    s.last_proposal = torch.LongTensor([[0, 2, 0]])

    total_available = 3 * 5 + 7 * 4 + 2 * 5
    print('total_available', total_available)
    actual = 3 * 3 + 7 * 4 + 5 * 2
    print('actual', actual)
    ratio = actual / total_available
    print('ratio', ratio)

    agent = 0 if t  % 2 == 0 else 1
    term = torch.ByteTensor([1])
    rewards = rewards_lib.calc_rewards(s=s, t=t, term=term)

    assert rewards[0,0] == approx((2 * 4) / (3 * 5 + 7 * 4 + 2 * 3))
    assert rewards[0,1] == approx((3 * 3 + 5 * 4 + 2 * 5) / (3 * 3 + 7 * 4 + 2 * 5))
    assert rewards[0,2] == approx(ratio)


def test_single_game_term_exceeds_pool():
    t = 1
    batch_size = 1
    torch.manual_seed(123)
    np.random.seed(123)
    s = ecn.State(**sampling.generate_batch(batch_size))

    s.pool = torch.LongTensor([[3, 7, 2]])
    s.utilities = torch.LongTensor([[[5,4,3], [3,4,5]]])
    # last proposal means agent 0's, and we are now on agent 1, who is accepintg it
    s.last_proposal = torch.LongTensor([[0, 2, 3]])

    agent = 0 if t  % 2 == 0 else 1
    term = torch.ByteTensor([1])
    rewards = rewards_lib.calc_rewards(s=s, t=t, term=term)
    assert rewards[0].tolist() == [0, 0, 0]


def test_single_game_term_exceeds_withinpool():
    t = 1
    batch_size = 1
    torch.manual_seed(123)
    np.random.seed(123)
    s = ecn.State(**sampling.generate_batch(batch_size))

    s.pool = torch.LongTensor([[3, 7, 2]])
    s.utilities = torch.LongTensor([[[5,4,3], [3,4,5]]])
    # last proposal means agent 0's, and we are now on agent 1, who is accepintg it
    s.last_proposal = torch.LongTensor([[0, 0, 0]])

    total_available = 3 * 5 + 7 * 4 + 2 * 5
    print('total_available', total_available)
    actual = 3 * 3 + 7 * 4 + 2 * 5
    print('actual', actual)
    ratio = actual / total_available
    print('ratio', ratio)

    agent = 0 if t  % 2 == 0 else 1
    term = torch.ByteTensor([1])
    rewards = rewards_lib.calc_rewards(s=s, t=t, term=term)

    assert rewards[0,0] == approx((0) / (3 * 5 + 7 * 4 + 2 * 3))
    assert rewards[0,1] == approx((3 * 3 + 7 * 4 + 2 * 5) / (3 * 3 + 7 * 4 + 2 * 5))
    assert rewards[0,2] == approx(ratio)


def test_single_game_term_exceeds_withinpool2():
    t = 1
    batch_size = 1
    torch.manual_seed(123)
    np.random.seed(123)
    s = ecn.State(**sampling.generate_batch(batch_size))

    s.pool = torch.LongTensor([[3, 7, 2]])
    s.utilities = torch.LongTensor([[[5,4,3], [3,4,5]]])
    # last proposal means agent 0's, and we are now on agent 1, who is accepintg it
    s.last_proposal = torch.LongTensor([[3, 7, 2]])

    total_available = 3 * 5 + 7 * 4 + 2 * 5
    print('total_available', total_available)
    actual = 3 * 5 + 7 * 4 + 2 * 3
    print('actual', actual)
    ratio = actual / total_available
    print('ratio', ratio)

    agent = 0 if t  % 2 == 0 else 1
    term = torch.ByteTensor([1])
    rewards = rewards_lib.calc_rewards(s=s, t=t, term=term)

    assert rewards[0,0] == approx((3 * 5 + 7 * 4 + 2 * 3) / (3 * 5 + 7 * 4 + 2 * 3))
    assert rewards[0,1] == approx((0) / (3 * 3 + 7 * 4 + 2 * 5))
    assert rewards[0,2] == approx(ratio)


def test_single_game_term_t2():
    t = 2
    batch_size = 1
    torch.manual_seed(123)
    np.random.seed(123)
    s = ecn.State(**sampling.generate_batch(batch_size))

    s.pool = torch.LongTensor([[3, 7, 2]])
    s.utilities = torch.LongTensor([[[5,4,3], [3,4,5]]])
    s.last_proposal = torch.LongTensor([[3, 0, 0]])

    total_available = 3 * 5 + 7 * 4 + 2 * 5
    print('total_available', total_available)
    # so, the  proposer is the second agent, ie agent 1
    # so, the proposer, agent 1, will take: 3 0 0
    # accepter, agent 0, will take 0 7 2
    actual = 0 * 5 + 7 * 4 + 2 * 3 + \
            3 * 3
    print('actual', actual)
    ratio = actual / total_available
    print('ratio', ratio)

    agent = 0 if t  % 2 == 0 else 1
    term = torch.ByteTensor([1])
    rewards = rewards_lib.calc_rewards(s=s, t=t, term=term)

    assert rewards[0,0] == approx((0 * 5 + 7 * 4 + 2 * 3) / (3 * 5 + 7 * 4 + 2 * 3))
    assert rewards[0,1] == approx((3 * 3 + 0 * 4 + 0 * 5) / (3 * 3 + 7 * 4 + 2 * 5))
    assert rewards[0,2] == approx(ratio)


def test_single_game_term_t2_batch3():
    t = 2
    batch_size = 3
    torch.manual_seed(123)
    np.random.seed(123)
    s = ecn.State(**sampling.generate_batch(batch_size))

    s.pool = torch.from_numpy(np.random.choice(10, (batch_size, 3))).long()
    s.pool[1] = torch.LongTensor([3, 7, 2])

    s.utilities = torch.from_numpy(np.random.choice(10, (batch_size, 2, 3))).long()
    s.utilities[1] = torch.LongTensor([[5,4,3], [3,4,5]])

    s.last_proposal = torch.from_numpy(np.random.choice(10, (batch_size, 3))).long()
    s.last_proposal[1] = torch.LongTensor([3, 0, 0])

    term = torch.ByteTensor([0, 1, 0])
    # since only one terminated, reward should be for simply the hard-coded ones above
    # all others should be zero

    s.pool[0] = s.last_proposal[0]
    s.pool[2] = s.last_proposal[2]

    # make rewards for 0 and 2 1.0
    s.utilities[0][1] = torch.max(s.utilities[0], 0)[0].view(1, 3)
    s.utilities[2][1] = torch.max(s.utilities[2], 0)[0].view(1, 3)

    total_available = 3 * 5 + 7 * 4 + 2 * 5
    print('total_available', total_available)
    # so, the  proposer is the second agent, ie agent 1
    # so, the proposer, agent 1, will take: 3 0 0
    # accepter, agent 0, will take 0 7 2
    actual = 0 * 5 + 7 * 4 + 2 * 3 + \
            3 * 3
    print('actual', actual)
    ratio = actual / total_available
    print('ratio', ratio)

    agent = 0 if t  % 2 == 0 else 1
    rewards = rewards_lib.calc_rewards(s=s, t=t, term=term)

    assert rewards[1,0] == approx((0 * 5 + 7 * 4 + 2 * 3) / (3 * 5 + 7 * 4 + 2 * 3))
    assert rewards[1,1] == approx((3 * 3 + 0 * 4 + 0 * 5) / (3 * 3 + 7 * 4 + 2 * 5))

    assert rewards[0].tolist() == [0.0, 0.0, 0]
    assert rewards[1,2] == approx(ratio)
    assert rewards[2].tolist() == [0.0, 0.0, 0]


def test_single_game_term_t2_batch3_2term():
    t = 2
    batch_size = 3
    torch.manual_seed(123)
    np.random.seed(123)
    s = ecn.State(**sampling.generate_batch(batch_size))

    s.pool = torch.from_numpy(np.random.choice(10, (batch_size, 3))).long()
    s.pool[1] = torch.LongTensor([3, 7, 2])

    s.utilities = torch.from_numpy(np.random.choice(10, (batch_size, 2, 3))).long()
    s.utilities[1] = torch.LongTensor([[5,4,3], [3,4,5]])

    s.last_proposal = torch.from_numpy(np.random.choice(10, (batch_size, 3))).long()
    s.last_proposal[1] = torch.LongTensor([3, 0, 0])

    term = torch.ByteTensor([0, 1, 1])
    # second should have reward too, and shouldnt affect the reward we are calcing

    s.pool[0] = s.last_proposal[0]
    s.pool[2] = s.last_proposal[2]

    # make rewards for 0 and 2 1.0
    s.utilities[0][1] = torch.max(s.utilities[0], 0)[0].view(1, 3)
    s.utilities[2][1] = torch.max(s.utilities[2], 0)[0].view(1, 3)

    total_available = 3 * 5 + 7 * 4 + 2 * 5
    print('total_available', total_available)
    # so, the  proposer is the second agent, ie agent 1
    # so, the proposer, agent 1, will take: 3 0 0
    # accepter, agent 0, will take 0 7 2
    actual = 0 * 5 + 7 * 4 + 2 * 3 + \
            3 * 3
    print('actual', actual)
    ratio = actual / total_available
    print('ratio', ratio)

    agent = 0 if t  % 2 == 0 else 1
    rewards = rewards_lib.calc_rewards(s=s, t=t, term=term)

    assert rewards[1,0] == approx((0 * 5 + 7 * 4 + 2 * 3) / (3 * 5 + 7 * 4 + 2 * 3))
    assert rewards[1,1] == approx((3 * 3 + 0 * 4 + 0 * 5) / (3 * 3 + 7 * 4 + 2 * 5))

    assert rewards[0].tolist() == [0.0, 0.0, 0]
    assert rewards[1,2] == approx(ratio)
    assert rewards[2,2] == 1.0


def test_single_game_term_t2_batch3_2termb():
    t = 2
    batch_size = 3
    torch.manual_seed(123)
    np.random.seed(123)
    s = ecn.State(**sampling.generate_batch(batch_size))

    s.pool = torch.from_numpy(np.random.choice(10, (batch_size, 3))).long()
    s.pool[1] = torch.LongTensor([3, 7, 2])

    s.utilities = torch.from_numpy(np.random.choice(10, (batch_size, 2, 3))).long()
    s.utilities[1] = torch.LongTensor([[5,4,3], [3,4,5]])

    s.last_proposal = torch.from_numpy(np.random.choice(10, (batch_size, 3))).long()
    s.last_proposal[1] = torch.LongTensor([3, 0, 0])

    term = torch.ByteTensor([1, 1, 0])
    # zeroth should have reward too, and shouldnt affect the reward we are calcing
    s.pool[0] = s.last_proposal[0]
    s.pool[2] = s.last_proposal[2]

    # make rewards for 0 and 2 1.0
    s.utilities[0][1] = torch.max(s.utilities[0], 0)[0].view(1, 3)
    s.utilities[2][1] = torch.max(s.utilities[2], 0)[0].view(1, 3)

    total_available = 3 * 5 + 7 * 4 + 2 * 5
    print('total_available', total_available)
    # so, the  proposer is the second agent, ie agent 1
    # so, the proposer, agent 1, will take: 3 0 0
    # accepter, agent 0, will take 0 7 2
    actual = 0 * 5 + 7 * 4 + 2 * 3 + \
            3 * 3
    print('actual', actual)
    ratio = actual / total_available
    print('ratio', ratio)

    agent = 0 if t  % 2 == 0 else 1
    rewards = rewards_lib.calc_rewards(s=s, t=t, term=term)
    assert rewards[0][2] == 1.0
    assert rewards[1,2] == approx(ratio)
    assert rewards[2].tolist() == [0.0, 0.0, 0]

    assert rewards[1,0] == approx((0 * 5 + 7 * 4 + 2 * 3) / (3 * 5 + 7 * 4 + 2 * 3))
    assert rewards[1,1] == approx((3 * 3 + 0 * 4 + 0 * 5) / (3 * 3 + 7 * 4 + 2 * 5))


def test_single_game_term_t2_batch3_3term():
    t = 2
    batch_size = 3
    torch.manual_seed(123)
    np.random.seed(123)
    s = ecn.State(**sampling.generate_batch(batch_size))

    s.pool = torch.from_numpy(np.random.choice(10, (batch_size, 3))).long()
    s.pool[1] = torch.LongTensor([3, 7, 2])

    s.utilities = torch.from_numpy(np.random.choice(10, (batch_size, 2, 3))).long()
    s.utilities[1] = torch.LongTensor([[5,4,3], [3,4,5]])

    s.last_proposal = torch.from_numpy(np.random.choice(10, (batch_size, 3))).long()
    s.last_proposal[1] = torch.LongTensor([3, 0, 0])

    term = torch.ByteTensor([1, 1, 1])
    # zeroth should have reward too, and shouldnt affect the reward we are calcing

    s.pool[0] = s.last_proposal[0]
    s.pool[2] = s.last_proposal[2]

    # make rewards for 0 and 2 1.0
    s.utilities[0][1] = torch.max(s.utilities[0], 0)[0].view(1, 3)
    s.utilities[2][1] = torch.max(s.utilities[2], 0)[0].view(1, 3)

    total_available = 3 * 5 + 7 * 4 + 2 * 5
    print('total_available', total_available)
    # so, the  proposer is the second agent, ie agent 1
    # so, the proposer, agent 1, will take: 3 0 0
    # accepter, agent 0, will take 0 7 2
    actual = 0 * 5 + 7 * 4 + 2 * 3 + \
            3 * 3
    print('actual', actual)
    ratio = actual / total_available
    print('ratio', ratio)

    agent = 0 if t  % 2 == 0 else 1
    rewards = rewards_lib.calc_rewards(s=s, t=t, term=term)

    assert rewards[0,2] == 1.0
    assert rewards[1,2] == approx(ratio)
    assert rewards[2,2] == 1.0

    assert rewards[1,0] == approx((0 * 5 + 7 * 4 + 2 * 3) / (3 * 5 + 7 * 4 + 2 * 3))
    assert rewards[1,1] == approx((3 * 3 + 0 * 4 + 0 * 5) / (3 * 3 + 7 * 4 + 2 * 5))


def test_single_game_term_t2_batch3_zero_term():
    t = 2
    batch_size = 3
    torch.manual_seed(123)
    np.random.seed(123)
    s = ecn.State(**sampling.generate_batch(batch_size))

    s.pool = torch.from_numpy(np.random.choice(10, (batch_size, 3))).long()
    s.pool[1] = torch.LongTensor([3, 7, 2])

    s.utilities = torch.from_numpy(np.random.choice(10, (batch_size, 2, 3))).long()
    s.utilities[1] = torch.LongTensor([[5,4,3], [3,4,5]])

    s.last_proposal = torch.from_numpy(np.random.choice(10, (batch_size, 3))).long()
    s.last_proposal[1] = torch.LongTensor([3, 0, 0])

    term = torch.ByteTensor([0, 0, 0])

    s.pool[0] = s.last_proposal[0]
    s.pool[2] = s.last_proposal[2]

    total_available = 3 * 5 + 7 * 4 + 2 * 5
    print('total_available', total_available)
    # so, the  proposer is the second agent, ie agent 1
    # so, the proposer, agent 1, will take: 3 0 0
    # accepter, agent 0, will take 0 7 2
    actual = 0 * 5 + 7 * 4 + 2 * 3 + \
            3 * 3
    print('actual', actual)
    ratio = actual / total_available
    print('ratio', ratio)

    agent = 0 if t  % 2 == 0 else 1
    rewards = rewards_lib.calc_rewards(s=s, t=t, term=term)
    assert rewards[0].tolist() == [0.0, 0.0, 0]
    assert rewards[1].tolist() == [0.0, 0.0, 0]
    assert rewards[2].tolist() == [0.0, 0.0, 0]


def test_single_game_term_t2_batch3_oneth_not_term():
    t = 2
    batch_size = 3
    torch.manual_seed(123)
    np.random.seed(123)
    s = ecn.State(**sampling.generate_batch(batch_size))

    s.pool = torch.from_numpy(np.random.choice(10, (batch_size, 3))).long()
    s.pool[1] = torch.LongTensor([3, 7, 2])

    s.utilities = torch.from_numpy(np.random.choice(10, (batch_size, 2, 3))).long()
    s.utilities[1] = torch.LongTensor([[5,4,3], [3,4,5]])

    s.last_proposal = torch.from_numpy(np.random.choice(10, (batch_size, 3))).long()
    s.last_proposal[1] = torch.LongTensor([3, 0, 0])

    term = torch.ByteTensor([1, 0, 1])

    s.pool[0] = s.last_proposal[0]
    s.pool[2] = s.last_proposal[2]

    # make rewards for 0 and 2 1.0
    s.utilities[0][1] = torch.max(s.utilities[0], 0)[0].view(1, 3)
    s.utilities[2][1] = torch.max(s.utilities[2], 0)[0].view(1, 3)

    total_available = 3 * 5 + 7 * 4 + 2 * 5
    print('total_available', total_available)
    # so, the  proposer is the second agent, ie agent 1
    # so, the proposer, agent 1, will take: 3 0 0
    # accepter, agent 0, will take 0 7 2
    actual = 0 * 5 + 7 * 4 + 2 * 3 + \
            3 * 3
    print('actual', actual)
    ratio = actual / total_available
    print('ratio', ratio)

    agent = 0 if t  % 2 == 0 else 1
    rewards = rewards_lib.calc_rewards(s=s, t=t, term=term)
    assert rewards[0].tolist()[2] == 1.0
    assert rewards[1].tolist() == [0.0, 0.0, 0]
    assert rewards[2].tolist()[2] == 1.0
