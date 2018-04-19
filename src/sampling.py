import torch
import numpy as np


def sample_items(batch_size, num_values=6, seq_len=3, random_state=np.random):
    """
    num_values 6 will give possible values: 0,1,2,3,4,5
    """
    pool = torch.from_numpy(random_state.choice(num_values, (batch_size, seq_len), replace=True))
    return pool


def sample_utility(batch_size, num_values=6, seq_len=3, random_state=np.random):
    u = torch.zeros(seq_len).long()
    while u.sum() == 0:
        u = torch.from_numpy(random_state.choice(num_values, (batch_size, seq_len), replace=True))
    return u


def sample_N(batch_size, random_state=np.random):
    N = random_state.poisson(7, batch_size)
    N = np.maximum(4, N)
    N = np.minimum(10, N)
    N = torch.from_numpy(N)
    return N


def generate_batch(batch_size, random_state=np.random):
    pool = sample_items(batch_size=batch_size, num_values=6, seq_len=3, random_state=random_state)
    utilities = []
    utilities.append(sample_utility(batch_size=batch_size, num_values=6, seq_len=3, random_state=random_state))
    utilities.append(sample_utility(batch_size=batch_size, num_values=6, seq_len=3, random_state=random_state))
    N = sample_N(batch_size=batch_size, random_state=random_state)
    return {
        'pool': pool,
        'utilities': utilities,
        'N': N
    }


def generate_test_batches(batch_size, num_batches, random_state):
    """
    so, we need:
    - pools
    - utilities (one set per agent)
    - N
    """
    # r = np.random.RandomState(seed)
    test_batches = []
    for i in range(num_batches):
        batch = generate_batch(batch_size=batch_size, random_state=random_state)
        test_batches.append(batch)
    return test_batches


def hash_long_batch(int_batch, num_values):
    seq_len = int_batch.size()[1]
    multiplier = torch.LongTensor(seq_len)
    v = 1
    for i in range(seq_len):
        multiplier[-i - 1] = v
        v *= num_values
    hashed_batch = (int_batch * multiplier).sum(1)
    return hashed_batch


def hash_batch(pool, utilities, N):
    v = N
    # use num_values=10, so human-readable
    v = v * 1000 + hash_long_batch(pool, num_values=10)
    v = v * 1000 + hash_long_batch(utilities[0], num_values=10)
    v = v * 1000 + hash_long_batch(utilities[1], num_values=10)
    return v


def hash_batches(test_batches):
    """
    we can store each game as a hash like:
    [N - 1]pppuuuuuu
    (where: [N - 1] is {4-10} - 1), ppp is the pool, like 442; and uuuuuu are the six utilities, like 354321
    so, this integer has 10 digits, which I guess we can just store as a normal python integer?
    """
    hashes = set()
    for batch in test_batches:
        hashed = hash_batch(**batch)
        hashes |= set(hashed.tolist())
        # for v in hashed:
        #     hashes.add(v)
    return hashes


def overlaps(test_hashes, batch):
    target_hashes = set(hash_batch(**batch).tolist())
    return bool(test_hashes & target_hashes)


def generate_training_batch(batch_size, test_hashes, random_state):
    batch = None
    while batch is None or overlaps(test_hashes, batch):
        batch = generate_batch(batch_size=batch_size, random_state=random_state)
    return batch
