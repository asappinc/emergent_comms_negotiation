"""
These are not really unit-tests as such, since they need manual inspection. but it's a convenient
place to store these functions that faciliate manual inspection
"""
import sampling
import torch
import numpy as np


def test_generate_test_batches():
    r = np.random.RandomState(123)
    test_batches = sampling.generate_test_batches(batch_size=32, num_batches=3, random_state=r)
    # for batch in test_batches:
    #     print(batch)


def test_hash3longs():
    a = (torch.rand(4, 3) * 5).long()
    print('a', a)
    hashed = sampling.hash_long_batch(a, num_values=10)
    print(hashed)


def test_hash_batch():
    r = np.random.RandomState(123)
    test_batches = sampling.generate_test_batches(batch_size=8, num_batches=1, random_state=r)
    batch = test_batches[0]
    print(batch)
    hashed = sampling.hash_batch(**batch)
    for v in hashed:
        print(v)


def test_hash_batches():
    r = np.random.RandomState(123)
    test_batches = sampling.generate_test_batches(batch_size=8, num_batches=3, random_state=r)
    hashes = sampling.hash_batches(test_batches)
    for v in hashes:
        print(v)


def test_checkoverlap():
    r = np.random.RandomState(123)
    batches = sampling.generate_test_batches(batch_size=32, num_batches=5, random_state=r)
    test_batches = batches[:2]
    train_batches = batches[2:]
    test_hashes = sampling.hash_batches(test_batches)
    assert sampling.overlaps(test_hashes=test_hashes, batch=test_batches[1])
    assert not sampling.overlaps(test_hashes=test_hashes, batch=train_batches[0])
    # grab one example from test batch, copy to train batch, check now overlaps
    src = test_batches[1]
    dest = train_batches[0]
    dest['pool'][5] = src['pool'][12]
    dest['utilities'][0][5] = src['utilities'][0][12]
    dest['utilities'][1][5] = src['utilities'][1][12]
    dest['N'][5] = src['N'][12]
    assert sampling.overlaps(test_hashes=test_hashes, batch=test_batches[0])


def test_generate_nonoverlapping_train_batch():
    """
    cant really test this, but at least check it runs

    (actually: we could test, by resetting the random state)
    """
    r = np.random.RandomState(123)
    test_batches = sampling.generate_test_batches(batch_size=32, num_batches=3, random_state=r)
    test_hashes = sampling.hash_batches(test_batches)
    train_batch = sampling.generate_training_batch(test_hashes=test_hashes, batch_size=32, random_state=r)
    print('train_batch', train_batch)
