import torch
import alive_sieve


def test_alive_sieve_init():
    batch_size = 12
    sieve = alive_sieve.AliveSieve(batch_size, enable_cuda=False)

    assert len(sieve.alive_mask) == batch_size
    assert len(sieve.alive_idxes) == batch_size
    for b in range(batch_size):
        assert sieve.alive_mask[b] == 1
        assert sieve.alive_idxes[b] == b


def test_alive_sieve_mark_dead():
    batch_size = 12
    sieve = alive_sieve.AliveSieve(batch_size, enable_cuda=False)

    dead_mask = torch.ByteTensor([0,1,0,1, 0,0,0,0, 1,1,1,1])
    sieve.mark_dead(dead_mask)

    assert len(sieve.alive_mask) == 12
    assert len(sieve.alive_idxes) == 6
    assert (sieve.alive_mask != torch.ByteTensor([1,0,1,0, 1,1,1,1, 0,0,0,0])).max() == 0
    assert (sieve.alive_idxes != torch.LongTensor([0,2, 4,5,6,7])).max() == 0


def test_alive_sieve_sieve_self():
    batch_size = 12
    sieve = alive_sieve.AliveSieve(batch_size, enable_cuda=False)

    dead_mask = torch.ByteTensor([0,1,0,1, 0,0,0,0, 1,1,1,1])
    sieve.mark_dead(dead_mask)

    assert len(sieve.alive_mask) == 12
    assert len(sieve.alive_idxes) == 6
    assert (sieve.alive_mask != torch.ByteTensor([1,0,1,0, 1,1,1,1, 0,0,0,0])).max() == 0
    assert (sieve.alive_idxes != torch.LongTensor([0,2, 4,5,6,7])).max() == 0

    assert (sieve.out_idxes != torch.LongTensor([0,1,2,3,4,5,6,7,8,9,10,11])).max() == 0
    assert sieve.batch_size == 12

    sieve.self_sieve_()
    assert (sieve.out_idxes != torch.LongTensor([0,2,4,5,6,7])).max() == 0
    assert sieve.batch_size == 6

    # sieve again...
    sieve.mark_dead(torch.ByteTensor([0,1,1,0,0,1]))
    assert (sieve.out_idxes != torch.LongTensor([0,2,4,5,6,7])).max() == 0
    sieve.self_sieve_()
    assert (sieve.out_idxes != torch.LongTensor([0,5,6])).max() == 0


def test_set_dead_global():
    batch_size = 12
    sieve = alive_sieve.AliveSieve(batch_size, enable_cuda=False)

    dead_mask = torch.ByteTensor([0,1,0,1, 0,0,0,0, 1,1,1,1])
    # so alive will be            1   1    1 1 1 1
    sieve.mark_dead(dead_mask)

    sieve.self_sieve_()

    sieve.mark_dead(torch.ByteTensor([0,1,1,0,0,1]))

    target = torch.rand(batch_size, 3)
    target_orig = target.clone()
    new_v = torch.rand(3, 3)
    sieve.set_dead_global(target, new_v)

    assert target[2][0] == new_v[0][0]
    assert target[4][0] == new_v[1][0]
    assert target[7][0] == new_v[2][0]


def test_playback():
    batch_size = 12
    sieve = alive_sieve.AliveSieve(batch_size, enable_cuda=False)
    alive_masks = []

    dead_mask = torch.ByteTensor([0,1,0,1, 0,0,0,0, 1,1,1,1])
    sieve.mark_dead(dead_mask)
    alive_masks.append(sieve.alive_mask)
    sieve.self_sieve_()

    sieve.mark_dead(torch.ByteTensor([0,1,1,0,0,1]))
    alive_masks.append(sieve.alive_mask)
    sieve.self_sieve_()

    sieve.mark_dead(torch.ByteTensor([0,0,1]))
    alive_masks.append(sieve.alive_mask)

    sieve.self_sieve_()
    sieve.mark_dead(torch.ByteTensor([1,1]))
    alive_masks.append(sieve.alive_mask)

    # print('alive_masks', alive_masks)

    sieve = alive_sieve.SievePlayback(alive_masks, enable_cuda=False)
    ts = []
    global_idxes_s = []
    batch_sizes = []
    for t, global_idxes in sieve:
        # print('t', t)
        # print('global_idxes', global_idxes)
        ts.append(t)
        global_idxes_s.append(global_idxes)
        # print('sieve.batch_size', sieve.batch_size)
        batch_sizes.append(sieve.batch_size)
    assert len(ts) == 4
    assert ts[0] == 0
    assert ts[1] == 1
    assert ts[2] == 2
    assert ts[3] == 3
    assert (global_idxes_s[0] - torch.LongTensor([0,1,2,3,4,5,6,7,8,9,10,11])).abs().max() == 0
    assert (global_idxes_s[1] - torch.LongTensor([0,2,4,5,6,7])).abs().max() == 0
    assert (global_idxes_s[2] - torch.LongTensor([0,5,6])).abs().max() == 0
    assert (global_idxes_s[3] - torch.LongTensor([0,5])).abs().max() == 0
    assert batch_sizes[0] == 12
    assert batch_sizes[1] == 6
    assert batch_sizes[2] == 3
    assert batch_sizes[3] == 2
