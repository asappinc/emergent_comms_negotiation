with context only, gets ~ 0.68, see [images/context-only.png], a6f3afc

adding in proposal, get ~0.73, after ~2500 rounds of 128 games, see [images/8b1caa8.png]

after adding utterance, and running for a while, d7d12843e77fd
```
  N=8
  pool=0,4,0
  util[0] 1,2,9
  util[1] 4,6,7
  A t=0.0 u=509771 p=0,0,0
  B t=1.0 u=832879 p=1,1,0
  steps=2 reward=1.00
episode 74270 avg rewards 0.74 0.74 b=0.74 games/sec 559.0 avg steps 2.0
saved model
  N=8
  pool=4,5,5
  util[0] 5,6,7
  util[1] 7,4,8
  A t=0.0 u=412343 p=0,0,0
  B t=1.0 u=933630 p=1,1,0
  steps=2 reward=0.90
episode 74287 avg rewards 0.73 0.73 b=0.72 games/sec 664.9 avg steps 2.0
  N=5
  pool=3,2,2
  util[0] 2,6,8
  util[1] 10,4,9
  A t=0.0 u=306570 p=0,0,0
  B t=1.0 u=933017 p=1,1,0
  steps=2 reward=0.93
episode 74304 avg rewards 0.73 0.73 b=0.74 games/sec 635.1 avg steps 2.0
saved model
  N=10
  pool=4,2,1
  util[0] 1,4,1
  util[1] 7,1,1
  A t=0.0 u=376059 p=0,0,0
  B t=1.0 u=565309 p=1,1,0
  steps=2 reward=0.84
episode 74317 avg rewards 0.73 0.73 b=0.74 games/sec 514.7 avg steps 2.0
saved model
  N=8
  pool=1,0,2
  util[0] 8,1,2
  util[1] 0,7,4
  A t=0.0 u=480752 p=0,0,0
  B t=1.0 u=133323 p=1,1,0
  steps=2 reward=0.50
episode 74332 avg rewards 0.73 0.73 b=0.72 games/sec 552.2 avg steps 2.0
  N=5
  pool=3,4,3
  util[0] 0,6,10
  util[1] 6,2,4
  A t=0.0 u=227836 p=0,0,0
  B t=1.0 u=735622 p=1,1,0
  steps=2 reward=0.53
episode 74345 avg rewards 0.73 0.73 b=0.73 games/sec 497.3 avg steps 2.0
saved model
```
=> seems to be in a local miminum ...

after ~150,000 batches of 128 games, still in local, 55de45b (see also [images/55de45b.png]) (saved as model model_saves/allnet_noentropy.dat):
```
episode 147066 avg rewards 0.74 0.74 b=0.73 games/sec 464.4 avg steps 2.0
saved model
  N=7
  pool=5,3,1
  util[0] 5,4,6
  util[1] 5,10,9
  A t=0.0 u=883613 p=0,0,0
  B t=1.0 u=884829 p=1,1,0
  steps=2 reward=1.00
episode 147080 avg rewards 0.73 0.73 b=0.72 games/sec 540.8 avg steps 2.0
  N=10
  pool=4,1,3
  util[0] 9,4,3
  util[1] 10,6,9
  A t=0.0 u=161003 p=0,0,0
  B t=1.0 u=556962 p=1,1,0
  steps=2 reward=1.00
episode 147093 avg rewards 0.72 0.72 b=0.73 games/sec 469.5 avg steps 2.0
saved model
  N=9
  pool=0,4,2
  util[0] 5,0,4
  util[1] 5,10,4
  A t=0.0 u=381030 p=0,0,0
  B t=1.0 u=752419 p=1,1,0
  steps=2 reward=1.00
episode 147105 avg rewards 0.72 0.72 b=0.73 games/sec 459.1 avg steps 2.0
saved model
  N=10
  pool=5,2,1
  util[0] 2,0,6
  util[1] 4,1,4
  A t=0.0 u=276362 p=0,0,0
  B t=1.0 u=327743 p=1,1,0
  steps=2 reward=0.93
episode 147117 avg rewards 0.72 0.72 b=0.72 games/sec 460.2 avg steps 2.0
```

fixed some bugs in lstm, because [seq] and [batch] dimensions were reversed/squished/mangled.

Still gets into a local maximum though, where `A` proposes 0,0,0; and `B` accepts....
=> probably needs the entropy regularization, to push it to try other options for 'term'

with term entropy regularization added and proposal entropy regularization added, 1f746b3, crashed:
```
episode 6091 avg rewards 0.74 0.74 b=0.73 games/sec 400 avg steps 2.00
  N=5
  pool=5,4,2
  util[0] 7,7,6
  util[1] 7,2,1
  A t=0.0 u=071278 p=0,0,0
  B t=1.0 u=981516 p=5,3,2
  steps=2 reward=0.60
episode 6102 avg rewards 0.73 0.73 b=0.72 games/sec 413 avg steps 2.00
  N=10
  pool=1,0,2
  util[0] 6,0,10
  util[1] 9,5,3
Traceback (most recent call last):
  File "ecn.py", line 574, in <module>
    run(**args.__dict__)
  File "ecn.py", line 472, in run
    render=render)
  File "ecn.py", line 305, in run_episode
    prev_proposal=Variable(last_proposal)
  File "/hugh/conda/lib/python3.6/site-packages/torch/nn/modules/module.py", line 224, in __call__
    result = self.forward(*input, **kwargs)
  File "ecn.py", line 246, in forward
    proposal_node, _entropy = proposal_policy(h_t)
  File "/hugh/conda/lib/python3.6/site-packages/torch/nn/modules/module.py", line 224, in __call__
    result = self.forward(*input, **kwargs)
  File "ecn.py", line 191, in forward
    out_node = torch.multinomial(x)
  File "/hugh/conda/lib/python3.6/site-packages/torch/autograd/variable.py", line 783, in multinomial
    return Multinomial(num_samples, replacement)(self)
  File "/hugh/conda/lib/python3.6/site-packages/torch/autograd/stochastic_function.py", line 23, in _do_forward
    result = super(StochasticFunction, self)._do_forward(*inputs)
  File "/hugh/conda/lib/python3.6/site-packages/torch/autograd/_functions/stochastic.py", line 16, in forward
    samples = probs.multinomial(self.num_samples, self.with_replacement)
RuntimeError: invalid argument 2: invalid multinomial distribution (sum of probabilities <= 0) at /opt/conda/conda-bld/pytorch_1503970438496/work/torch/lib/TH/generic/THTensorRandom.c:230
```
=> seems like the regularization pushed the weights to zero. I wonder how to solve that?

Also, games per second is much lower with current entropy regularization. I imagine because back-propping twice each time. I wonder if there is a better way of implementing the entropy regularization?

Fixed the entr reg speed issue somewhat, 653d3ad, from ~400 to ~570 games/sec, from the latest model save:
```
episode 6139 avg rewards 0.72 0.72 b=0.72 games/sec 565 avg steps 2.00
  N=9
  pool=1,0,2
  util[0] 9,0,9
  util[1] 5,5,7
  A t=0.0 u=164342 p=0,0,0
  B t=1.0 u=816912 p=0,4,2
  steps=2 reward=0.70
episode 6154 avg rewards 0.74 0.74 b=0.73 games/sec 577 avg steps 2.00
  N=4
  pool=2,5,3
  util[0] 0,1,4
  util[1] 6,2,6
```

invalid multinomial distr crashbug remains

fixed multinomial crashbug in 4d628ee65

turn to looking at the local minimum where first agent proposes all zeros, and second agent immediately accepts.

let's remove comms, to speed up training. trying 4599249 wiht `--disable-comms`, which runs at ~1000 games/second, using cpu.

after a bunch of epochs, using new cudarized version, still in local minimimum:
```
episode 156397 avg rewards 0.733 0.733 b=0.734 games/sec 2339 avg steps 0.0000
  N=7
  pool=3,5,5
  util[0] 2,9,4
  util[1] 4,5,0
  A t=0.0 u=000000 p=0,0,0
  B t=1.0 u=000000 p=3,4,1
  steps=2 reward=0.48
episode 156453 avg rewards 0.731 0.731 b=0.714 games/sec 2346 avg steps 0.0000
  N=7
  pool=0,1,5
  util[0] 6,8,7
  util[1] 0,0,9
  A t=0.0 u=000000 p=0,0,0
  B t=1.0 u=000000 p=3,4,1
  steps=2 reward=0.85
episode 156509 avg rewards 0.727 0.727 b=0.725 games/sec 2342 avg steps 0.0000
saved model
  N=9
  pool=2,0,1
  util[0] 8,9,5
  util[1] 10,6,0
  A t=0.0 u=000000 p=0,0,0
  B t=1.0 u=000000 p=3,4,1
  steps=2 reward=0.80
episode 156564 avg rewards 0.732 0.732 b=0.721 games/sec 2295 avg steps 0.0000
```

same for cpu version:
```
  N=5
  pool=4,1,2
  util[0] 3,3,10
  util[1] 4,3,2
  A t=0.0 u=000000 p=0,0,0
  B t=1.0 u=000000 p=2,4,3
  steps=2 reward=0.59
episode 126863 avg rewards 0.729 0.729 b=0.725 games/sec 1250 avg steps 2.0000
  N=6
  pool=2,5,2
  util[0] 4,1,3
  util[1] 7,4,9
  A t=0.0 u=000000 p=0,0,0
  B t=1.0 u=000000 p=2,3,3
  steps=2 reward=1.00
episode 126894 avg rewards 0.727 0.727 b=0.733 games/sec 1249 avg steps 2.0000
  N=10
  pool=5,2,0
  util[0] 2,6,0
  util[1] 1,0,5
  A t=0.0 u=000000 p=0,0,0
  B t=1.0 u=000000 p=2,3,3
  steps=2 reward=0.23
episode 126924 avg rewards 0.734 0.734 b=0.740 games/sec 1234 avg steps 2.0000
```

Not sure how to fix this. Increase entropy regularization term?

after a few more episodes, still stuck, cpu version:
```
  N=7
  pool=4,3,0
  util[0] 8,1,4
  util[1] 5,9,3
  A t=0.0 u=000000 p=0,0,0
  B t=1.0 u=000000 p=4,4,4
  steps=2 reward=0.80
episode 197938 avg rewards 0.721 0.721 b=0.717 games/sec 1271 avg steps 2.0000
```
gpu version:
```
  N=9
  pool=4,5,0
  util[0] 5,3,9
  util[1] 0,10,6
  A t=0.0 u=000000 p=0,0,0
  B t=1.0 u=000000 p=3,4,1
  steps=2 reward=0.71
episode 296182 avg rewards 0.726 0.726 b=0.733 games/sec 2309 avg steps 0.0000
```

=> lets try increasing regularization a bit, eg set both to 1.0 for now... or ... use .sum() instead of .mean() ?

Using .sum() instead of .mean() seems promising. using gpu version, ad2fad1:
```
  N=7
  pool=0,4,0
  util[0] 9,7,10
  util[1] 8,4,10
  A t=0.0 u=000000 p=2,0,5
  B t=0.0 u=000000 p=5,2,5
  A t=0.0 u=000000 p=0,0,0
  B t=1.0 u=000000 p=1,5,1
  steps=4 reward=0.57
episode 4543 avg rewards 0.740 0.740 b=0.749 games/sec 791 avg steps 4.2285
```

training graph so far: [images/cfb35e5-gpu.png] and [images/cfb35e5-cpu.png]

commandlines:
- cpu: `python ecn.py --model-file model_saves/nocomms_sument.dat --disable-comms`
- gpu: `python ecn.py --model-file model_saves/cuda_nocomms_sument.dat --enable-cuda --disable-comms`

speed at this point, after convergence, ie avg steps/game ~4.2:
- c4.8xlarge: ~520 games/sec
- g3.4xlarge: ~880 games/sec
- v100: ~1380 games/sec (using cuda9 branch, https://github.com/ASAPPinc/emergent_comms_negotiation/tree/cuda9 , ~~not very tested...~~ but reward-curve flat-lined, after ~1 hour; going to abandon this branch for now)

gpu, no comms, logs/log_20171104_144936.log
```
python ecn.py --model-file model_saves/cuda_nocomms_sument_c.dat --enable-cuda --disable-comms
```
python merge.py --hostname gpu1 --logfile logs/log_20171104_144936.log --min-y 0.78 --max-y 0.81 --title 'Proposal, social, no comms'

Looking at the reward curves, looks like they're plateau'ing. Possible next approaches:
- add utterances?
- re-check the paper, and compare with their no-comms results?

checking paper, seems like the reward curve is not a million miles away from 2a 'proposal', albeit that is for unshared rewards.

Let's try adding utterances

added utterances and entropy, training against d836149 using logfile logs/log_20171104_192343gpu2.log (gets ~520games/sec, with avg steps per game ~3.9, on g3.4xlarge)
```
python ecn.py --name gpu2 --enable-cuda --model-file model_saves/cuda_withcomms.dat
```
merge command:
```
python merge.py --hostname gpu2 --logfile logs/log_20171104_192343gpu2.log --min-y 0.78 --max-y 0.81 --title 'Proposal, social, comms'
```

refactored a bit, in `alivesieve` branch, https://github.com/ASAPPinc/emergent_comms_negotiation/tree/alivesieve , testing in with-comms, with-proposal, prosocial mode, just in case changes the results compared to current `master` branch code

commandline:
```
python ecn.py --model-file model_saves/sieve.dat --enable-cuda
```
logfile logs/log_20171105_220128.log

merge command:
```
python merge.py --hostname gpu3 --logfile logs/log_20171105_220128.log --min-y 0.78 --max-y 0.81 --title 'Proposal, social, comms; run 2'
```

nov 6 10:34 GMT, bit more factorization, using SievePlayback. Start new run, wtih comms on, using commandline:
```
python ecn.py --model-file model_saves/sieve2.dat --enable-cuda --name gpu4_sieve2
```
merge command:
```
python merge.py --hostname gpu3 --logfile logs/log_20171106_103440gpu4_sieve2.log --min-y 0.78 --max-y 0.81 --title 'Proposal, social, comms; run 3'
```

2017 nov 13:
- received reply to comments on openreview
- changed utility values from `[0,1,..,10]` to `[0,1,..,5]`

Launch a training run:
```
python ecn.py --model-file model_saves/v030.dat --name gpu1v030 --enable-cuda
```
(launched at ~14:25 uk time)

merge command:
```
python merge.py --hostname gpu1 --logfile logs/log_20171113_142411gpu1v030.log,logs/log_20171113_181106gpu1v030.log
```

After running for a while, training curve is still not going above ~0.78, on prosocial setting.  Let's plot also the percentage of actions matching the argmax, as suggested in the comments response.

added logging of fraction of time policy chooses argmax, and launched on gpu2:
```
python ecn.py --enable-cuda --name gpu2argmaxp --model-file model_saves/gpu2argmaxp.dat
```

After running overnight:
- first run (gpu1) is still around ~0.78, [images/v030_comms_social_prop_run1.png]
- second run (gpu2) plateud at ~0.75 for ~60,000 batches, then rose to 0.82 [images/v030_comms_social_prop_run2.png]

For the second one, we can see the proportion of policy choices that match the argmax:
```
episode 119066 avg rewards 0.814 0.814 b=0.827 games/sec 498 avg steps 3.8084 argmaxp term=0.7426 utt=0.1466 prop=0.3427
```
The proportions are:
- termination policy: 0.74
- utterance policy: 0.15
- proposal policy: 0.34

Seems entropy not too small? (but maybe too large?)

added `--testing` option, to disable training, and use the argmax insteda of stochstic draws. try it on gpu1 using:
```
python ecn.py --testing --enable-cuda --model-file model_saves/gpu2argmaxp.dat --name gpu1test
```

well, thats interseting. term policy never terminates, and all games run out of time :P
```
   287878 5:4/1 5:3/5 2:1/4
                                      999999 1:4/1 4:5/5 2:5/4
   287878 5:1/1 5:1/5 2:0/4
                                      999999 1:4/1 4:2/5 2:2/4
   287878 5:1/1 5:1/5 2:0/4
  [out of time]
  r: 0.00

episode 133267 avg rewards 0.000 0.000 b=0.000 games/sec 1100 avg steps 6.8970 argmaxp term=1.0000 utt=1.0000 prop=1.0000
```

also: look at the utterances: invariant with previous utterance, and previous proposal.

Some possible hypotheses:
- entropy is too high, preventing learning
- some bug means terminator and utterance policy not learning
- some bug means terminator and utterance policy are not being exposed correctly to previous utterance or proposal

We can test the first hypothesis by using a lower entropy. Lets lower by 100 (which was causing local minima earlier, so should be lower enough; could also try 0 perhaps. Actually, let's just jump to zero...). Launch on gpu1:

```
python ecn.py --model-file model_saves/v030_gpu1_zero_ent.dat --name gpu1zeroent --enable-cuda --term-entropy-reg 0 --utterance-entropy-reg 0 --proposal-entropy-reg 0
```

and then occasionally sample with:
```
python ecn.py --testing --enable-cuda --model-file model_saves/v030_gpu1_zero_ent.dat --name gpu1test
```

same results: never terminates, avg reward 0.000, eg:
```
   803232 4:0/5 2:0/3 1:0/2
                                      833333 0:5/5 4:0/3 1:1/2
   803232 4:0/5 2:0/3 1:0/2
                                      833333 0:5/5 4:0/3 1:1/2
  [out of time]
  r: 0.00

episode 933 avg rewards 0.000 0.000 b=0.000 games/sec 929 avg steps 6.9050 argmaxp term=1.0000 utt=1.0000 prop=1.0000
```
(similar for earlier episodes). meanwhile training reward is ok:
```
   700707 2:0/0 2:0/1 1:0/4
                                      ACC
  r: 1.00

episode 1456 avg rewards 0.718 0.718 b=0.710 games/sec 1269 avg steps 2.0012 argmaxp term=0.5003 utt=0.4429 prop=0.9939
```

after fixing bug with greedy term evaluation, results exactly match stochastic rewards, for zero entropy reg case:
```

   399999 3:0/1 2:0/5 4:0/2
                                      ACC
  r: 1.00

episode 20054 avg rewards 0.713 0.713 b=0.718 games/sec 1241 avg steps 2.0000 argmaxp term=0.5000 utt=0.9491 prop=1.0000
```

for default entropy reg, results for `--testing` are:
```
$ python ecn.py --testing --enable-cuda --model-file model_saves/gpu2argmaxp.dat --name gpu1test
...

loaded model

   859217 4:3/3 1:3/4 1:3/2
                                      796049 0:5/3 5:5/4 3:5/2
   149217 4:1/3 1:0/4 1:0/2
                                      ACC
  r: 0.79

episode 145364 avg rewards 0.880 0.880 b=0.878 games/sec 1995 avg steps 3.6915 argmaxp term=1.0000 utt=1.0000 prop=1.0000

   859217 1:3/0 3:3/2 3:3/5
                                      796049 3:5/0 5:5/2 3:5/5
   149217 1:0/0 3:1/2 3:1/5
                                      ACC
  r: 0.92

episode 145448 avg rewards 0.880 0.880 b=0.882 games/sec 2104 avg steps 3.6910 argmaxp term=1.0000 utt=1.0000 prop=1.000
```

results better than training, but still some weirdness:
- much lower than the paper (0.88 vs 0.95 or so)
- utterances always identical, ignore the pool, utilities, and previous proposal :P

rewritten nets to use non-pytorch proprietary expressions for reinforce bits, ie take logs etc, rather than calling `.reinforce`. Written some tests
(nets_test.py) that show these actually learn, though they dont learn terribly well...

fixed ecn.py for the new reinforce approach.

Let's relaunch, on gpu2

```
python ecn.py --enable-cuda --name gpu2newreinf --model-file model_saves/gpu2newreinf.dat
```

after some episodes, reward in `--testing` is about ~0.04 better than training:
```
python ecn.py --testing --enable-cuda --name testing --model-file model_saves/gpu2newreinf.dat
loaded model

   111111 5:5/4 3:2/4 2:0/0
                                      157777 2:1/4 0:0/4 4:4/0
   011111 5:0/4 3:1/4 2:0/0
                                      ACC
  r: 0.34

episode 789 avg rewards 0.741 0.741 b=0.755 games/sec 1602 avg steps 3.9976 argmaxp term=1.0000 utt=1.0000 prop=1.0000
```

compared with training:
```
saved model

   694199 2:4/1 0:3/2 0:1/2
                                      187853 4:5/1 4:3/2 5:3/2
   401333 2:0/1 0:0/2 0:0/2
                                      ACC
  r: 1.00

episode 1002 avg rewards 0.715 0.715 b=0.723 games/sec 553 avg steps 4.0791 argmaxp term=0.9993 utt=0.2212 prop=0.3453

   433535 2:1/5 0:3/0 4:4/2
                                      244519 1:0/5 0:2/0 5:5/2
   333948 2:0/5 0:0/0 4:0/2
                                      ACC
  r: 0.75

episode 1025 avg rewards 0.700 0.700 b=0.687 games/sec 559 avg steps 4.0781 argmaxp term=0.9994 utt=0.2183 prop=0.3428
```
looks like entropy on terminator policy might be fairly low? Lets leave to train for a while.


after a bit:
```
episode 7226 avg rewards 0.747 0.747 b=0.751 games/sec 636 avg steps 4.0165 argmaxp term=0.9997 utt=0.1945 prop=0.3344
saved model

   593874 5:3/3 4:1/4 0:3/1
                                      407631 1:5/3 2:3/4 0:4/1
   017314 5:1/3 4:1/4 0:0/1
                                      ACC
  r: 0.55

episode 7252 avg rewards 0.755 0.755 b=0.757 games/sec 622 avg steps 4.0349 argmaxp term=0.9997 utt=0.2022 prop=0.3354

   575323 3:0/2 5:4/5 5:1/5
                                      403201 3:2/2 3:0/5 3:4/5
   355839 3:0/2 5:1/5 5:1/5
                                      ACC
  r: 0.71

episode 7276 avg rewards 0.744 0.744 b=0.740 games/sec 573 avg steps 4.0534 argmaxp term=0.9992 utt=0.1989 prop=0.3330
```

=> lets multiple term entropy by 100, see what happens. relaunch with
```
python ecn.py --enable-cuda --name gpu2newreinf --model-file model_saves/gpu2newreinf_termentreg5.dat --term-entropy-reg 5
```

cancelled this because too junky, and launched with:
```
python ecn.py --enable-cuda --name gpu2newreinf --model-file model_saves/gpu2newreinf_termentreg05.dat --term-entropy-reg 0.5 --utterance-entropy-reg 0.0001
```

```
episode 37964 avg rewards 0.631 0.631 b=0.644 games/sec 527 avg steps 2.9922 argmaxp term=0.7962 utt=0.6525 prop=0.3949
saved model

   586950 1:0/4 3:2/4 1:0/5
                                      ACC
  r: 0.76

episode 37986 avg rewards 0.624 0.624 b=0.619 games/sec 527 avg steps 2.9734 argmaxp term=0.7879 utt=0.6519 prop=0.3971

   869996 3:5/2 5:2/5 2:5/0
                                      105299 4:2/2 0:0/5 0:0/0
   ACC
  r: 1.00

episode 38009 avg rewards 0.641 0.641 b=0.652 games/sec 537 avg steps 3.0268 argmaxp term=0.7936 utt=0.6527 prop=0.4009

   555089 1:0/0 1:0/2 1:1/4
                                      ACC
  r: 1.00
```

looks like proposal ent reg too high, lets lower it to 0.01
```
python ecn.py --enable-cuda --name gpu2newreinf --model-file model_saves/gpu2newreinf_termentreg0_5_uttreg0_0001_propreg0_01.dat --term-entropy-reg 0.5 --utterance-entropy-reg 0.0001 --proposal-entropy-reg 0.01
```
working ok so far:
```
  r: 0.97

episode 40049 avg rewards 0.758 0.758 b=0.741 games/sec 512 avg steps 2.8317 argmaxp term=0.7328 utt=0.8043 prop=0.7673

   888888 5:3/5 1:0/4 0:0/1
                                      ACC
  r: 0.82

episode 40072 avg rewards 0.790 0.790 b=0.797 games/sec 554 avg steps 2.7826 argmaxp term=0.7552 utt=0.7998 prop=0.7596

   888888 0:0/2 2:0/4 2:0/5
                                      ACC
  r: 0.85

episode 40095 avg rewards 0.785 0.785 b=0.781 games/sec 549 avg steps 2.8013 argmaxp term=0.7541 utt=0.7977 prop=0.7629

   888888 5:0/0 5:2/2 4:2/3
                                      ACC
  r: 0.95
...
episode 88918 avg rewards 0.813 0.813 b=0.806 games/sec 547 avg steps 2.8631 argmaxp term=0.7350 utt=0.7994 prop=0.8470

   808808 0:0/2 2:0/2 3:2/2
                                      ACC
  r: 1.00

episode 88941 avg rewards 0.813 0.813 b=0.811 games/sec 543 avg steps 2.9049 argmaxp term=0.7356 utt=0.7934 prop=0.8440

   888066 3:1/1 3:1/2 4:2/2
                                      ACC
  r: 0.95

episode 88964 avg rewards 0.825 0.825 b=0.827 games/sec 553 avg steps 2.8628 argmaxp term=0.7371 utt=0.7911 prop=0.8454
```

sample:
```
python ecn.py --testing --enable-cuda --name testing --model-file model_saves/gpu2newreinf_termentreg0_5_uttreg0_0001_propreg0_01.dat
```

Ok, cool :)

```
(root) ubuntu@gpu2:~/git/emergent_comms_negotiation$ python ecn.py --testing --enable-cuda --name testing --model-file model_saves/gpu2newreinf_termentreg0_5_uttreg0_0001_propreg0_01.dat
loaded model

   808080 2:2/2 1:0/2 4:1/1
                                      ACC
  r: 0.75

episode 87070 avg rewards 0.945 0.945 b=0.938 games/sec 1059 avg steps 3.0474 argmaxp term=1.0000 utt=1.0000 prop=1.0000

   808080 1:0/3 3:1/1 5:0/0
                                      633333 5:3/3 1:0/1 4:0/0
   ACC
  r: 1.00

episode 87114 avg rewards 0.945 0.945 b=0.941 games/sec 1078 avg steps 3.0506 argmaxp term=1.0000 utt=1.0000 prop=1.0000

   808080 5:1/1 0:0/1 5:2/2
                                      633333 5:1/1 4:1/1 2:0/2
   ACC
  r: 1.00

episode 87159 avg rewards 0.945 0.945 b=0.950 games/sec 1113 avg steps 3.0455 argmaxp term=1.0000 utt=1.0000 prop=1.0000
```

I think we should add an automatic test step into the main training loop.

Plot graph:
```
python merge.py --hostname gpu2 --logfile logs/log_20171114_182322gpu2newreinf.log --title 'Comms,Prop,Soc termreg 0.5 uttreg 0.0001 propreg 0.01'
cp /tmp/out-reward.png images/v030_comms_social_prop_termreg0_5_uttreg0_0001_propreg0_01.png
```

implemented running tests every 30 seconds or so, lets spin up a gpu, and start that running, gpu1:

```
python ecn.py --enable-cuda --name gpu1withtests --model-file model_saves/gpu1newreinf_termentreg0_5_uttreg0_0001_propreg0_01.dat --term-entropy-reg 0.5 --utterance-entropy-reg 0.0001 --proposal-entropy-reg 0.01
python merge.py --hostname gpu1 --logfile logs/log_20171115_012645gpu1withtests.log --title 'Comms,Prop,Soc termreg 0.5 uttreg 0.0001 propreg 0.01, run 2'
```

After ~100k batches, this second run looks promising, reached ~0.94 on test, ie with stochasticity disabled:
- [images/v030_comms_social_prop_termreg0_5_uttreg0_0001_propreg0_01_run2.png]

(shutting down the other one, since 1. plateaud, 2. reached ~200k batches, 3. no non-stochastic logging)

some variance between the two runs. lets launch another run, and see what happens. on new gpu2:

```
python ecn.py --enable-cuda --name gpu2withtestsrun3 --model-file model_saves/gpu2newreinf_termentreg0_5_uttreg0_0001_propreg0_01_run3.dat --term-entropy-reg 0.5 --utterance-entropy-reg 0.0001 --proposal-entropy-reg 0.01
python merge.py --hostname gpu1 --logfile logs/log_20171115_084751gpu2withtestsrun3.log --title 'Comms,Prop,Soc termreg 0.5 uttreg 0.0001 propreg 0.01, run 3'
```
(launched)

(but anyway, seems the ~0.94 or so for the gpu1newreinf_termentreg0_5_uttreg0_0001_propreg0_01.dat model is broadly in agreement with the 0.92 reported in table 1)

forgot to note down the greedy-percentage in the earlier run, on gpu2, but for the current run, on gpu1, ie gpu1newreinf_termentreg0_5_uttreg0_0001_propreg0_01.dat model, it looks like:
```
test rewards 0.894
episode 112004 avg rewards 0.818 0.818 b=0.828 games/sec 531 avg steps 2.8790 argmaxp term=0.7411 utt=0.7788 prop=0.8223
```
ie greedy percentage somewhere around 70-80%

Let's also do a run with comms turned off. Using new gpu3 instance:
```
python ecn.py --enable-cuda --name gpu3nocomms --disable-comms --model-file model_saves/gpu3nocomms_reg0_5_uttreg0_0001_propreg0_01_run1.dat --term-entropy-reg 0.5 --utterance-entropy-reg 0.0001 --proposal-entropy-reg 0.01
```

and another with proposal off, using gpu4:
```
python ecn.py --enable-cuda --name gpu4noprop --disable-proposal --model-file model_saves/gpu4noprop_reg0_5_uttreg0_0001_propreg0_01_run1.dat --term-entropy-reg 0.5 --utterance-entropy-reg 0.0001 --proposal-entropy-reg 0.01
```
(launched)

Let's also do no-comms, no-proposal, on gpu5:

```
python ecn.py --enable-cuda --name gpu5nocommsnoprop --disable-proposal --disable-comms --model-file model_saves/gpu5nocommsnoprop_reg0_5_uttreg0_0001_propreg0_01_run1.dat --term-entropy-reg 0.5 --utterance-entropy-reg 0.0001 --proposal-entropy-reg 0.01
```
(launched)

Tweaked rewards, so we can do --disable-prosocial. Lets launch two instances now:

gpu6, comms,prop,soc, regression test basically:
```
python ecn.py --enable-cuda --name gpu6withtestsrun4 --model-file model_saves/gpu6newreinf_termentreg0_5_uttreg0_0001_propreg0_01_run3.dat --term-entropy-reg 0.5 --utterance-entropy-reg 0.0001 --proposal-entropy-reg 0.01
```
(launched)

and gpu7, test without social, with comms and proposal:
```
python ecn.py --enable-cuda --name gpu7nosoc1 --model-file model_saves/gpu7nosoc_termentreg0_5_uttreg0_0001_propreg0_01_run3.dat --term-entropy-reg 0.5 --utterance-entropy-reg 0.0001 --proposal-entropy-reg 0.01 --disable-prosocial
```
(launched)

list of current logfiles:
```
log_20171115_012645gpu1withtests.log
log_20171115_084751gpu2withtestsrun3.log
log_20171115_091110gpu3nocomms.log
log_20171115_091432gpu3nocomms.log
log_20171115_092824gpu4noprop.log
log_20171115_110859gpu5nocommsnoprop.log
log_20171115_124546gpu6withtestsrun4.log
log_20171115_124559gpu7nosoc1.log
log_20171115_124651gpu6withtestsrun4.log
log_20171115_124703gpu7nosoc1.log
log_20171115_124951gpu6withtestsrun4.log
log_20171115_125020gpu7nosoc1.log
```

for plotting, we have the following logs:
- comms,prop,soc, with testing:
  - logs/log_20171115_012645gpu1withtests.log
  - logs/log_20171115_084751gpu2withtestsrun3.log
  - logs/log_20171115_124951gpu6withtestsrun4.log
  - `python merge.py --hostname gpu1 --logfile 'logs/log_20171115_012645gpu1withtests.log,logs/log_20171115_084751gpu2withtestsrun3.log,logs/log_20171115_124951gpu6withtestsrun4.log' --title 'Comms,Prop,Soc termreg 0.5 uttreg 0.0001 propreg 0.01'`
- nocomms, noprop, soc:
  - logs/log_20171115_110859gpu5nocommsnoprop.log
- comms, prop, no soc:
  - logs/log_20171115_125020gpu7nosoc1.log
- comms, noprop, soc:
  - logs/log_20171115_092824gpu4noprop.log
- nocomms, prop, soc:
  - logs/log_20171115_091432gpu3nocomms.log

Looks like the regression test one, gpu6, is showing test results aligned with the others, but training rewards look odd:
```
{"episode": 25090, "avg_reward_0": 0.40169024260147757, "avg_reward_1": 0.4774097608483356, "test_reward": 0.8788682699203492, "avg_steps": 2.9519701086956522, "games_sec": 475.6461894587855, "elapsed": 6935.344210624695, "argmaxp_term": 0.7783812395001496, "argmaxp_utt": 0.5034174855591098, "argmaxp_prop": 0.6818017935086952}
```
0.4 is a little low. some bug...

found the bug :P  relaunching....

new logfiles for gpu6 and 7:
```
log_20171115_144911gpu6withtestsrun4.log
log_20171115_144928gpu7nosoc1.log
```

new merge command for comms,prop,soc:
```
python merge.py --hostname gpu1 --logfile 'logs/log_20171115_012645gpu1withtests.log,logs/log_20171115_084751gpu2withtestsrun3.log,logs/log_20171115_144911gpu6withtestsrun4.log' --title 'Comms,Prop,Soc termreg 0.5 uttreg 0.0001 propreg 0.01'
```

see: [images/comms_prop_soc_tests_threerunsb.png]

other merge commands:
```
python merge.py --hostname gpu1 --logfile 'logs/log_20171115_110859gpu5nocommsnoprop.log' --title 'No comms, no prop, soc termreg 0.5 uttreg 0.0001 propreg 0.01' --min-y 0.68 --max-y 0.9
cp /tmp/out-reward.png images/20171115_noprop_nocomms_soc.png

python merge.py --hostname gpu1 --logfile 'logs/log_20171115_092824gpu4noprop.log' --title 'Comms, no prop, soc termreg 0.5 uttreg 0.0001 propreg 0.01' --min-y 0.68 --max-y 0.9
cp /tmp/out-reward.png images/20171115_noprop_comms_soc.png

python merge.py --hostname gpu1 --logfile 'logs/log_20171115_091432gpu3nocomms.log' --title 'No comms, prop, soc termreg 0.5 uttreg 0.0001 propreg 0.01' --min-y 0.72 --max-y 0.95
cp /tmp/out-reward.png images/20171115_prop_nocomms_soc.png

python merge.py --hostname gpu1 --logfile 'logs/log_20171115_125020gpu7nosoc1.log' --title 'Comms, prop, no soc termreg 0.5 uttreg 0.0001 propreg 0.01'
cp /tmp/out-reward.png images/20171115_prop_comms_nosoc.png
```

greedy proportions, samples by hand:
```
comms prop soc: term=0.7345 utt=0.7635 prop=0.8304
nocomms prop soc: term=0.6965 utt=0.0000 prop=0.8741
comms noprop soc: term=0.6889 utt=0.7849 prop=0.8222
nocomms noprop soc: term=0.7781 utt=0.0000 prop=0.6006
comms prop nosoc: term=0.7467 utt=0.9284 prop=0.8137
```

graph for nosoc looks weird. Bug?

some sampled conversations:

from training:
```

   ACC
  r: 0.00
```
from testing:
```
   534343 3:4/5 1:2/2 0:0/4
                                      686868 0:0/5 2:2/2 3:4/4
   ACC
  r: 0.96


   534343 3:0/0 2:3/5 3:0/0
                                      686868 0:0/0 2:4/5 5:0/0
   534343 3:0/0 2:3/5 3:0/0
                                      686868 0:0/0 2:4/5 5:0/0
   534343 3:0/0 2:3/5 3:0/0
                                      686868 0:0/0 2:4/5 5:0/0
   534343 3:0/0 2:3/5 3:0/0
                                      686868 0:0/0 2:4/5 5:0/0
  [out of time]
  r: 0.00


   534343 3:4/4 1:0/5 4:4/4
                                      ACC
  r: 0.95


   534343 0:0/1 0:0/5 5:2/2
                                      ACC
  r: 0.94


   534343 4:4/4 5:2/3 5:4/5
                                      686868 1:1/4 5:3/3 2:4/5
   534343 4:3/4 5:0/3 5:4/5
                                      686868 1:0/4 5:3/3 2:4/5
   534343 4:4/4 5:3/3 5:4/5
                                      686868 1:0/4 5:3/3 2:4/5
   534343 4:4/4 5:3/3 5:4/5
  [out of time]
  r: 0.00
```
seems like the agents always say the same thing, and basically just guess based on their own utilities.

termination seems flaky

oh... what is happening is, in training the agents negotiate a bit, but termination entropy always terminates the conversations quickly-ish. but in testing, keep running out of time, since very deterministic: the whole conversation, not just termination.

Let's lower termination entropy and try again. term entropy 0.5 => 0.05.  proposal entropy 0.01 => 0.005
```
 python ecn.py --enable-cuda --name gpu7nosoc2 --model-file model_saves/gpu7nosoc_termentreg0_05_uttreg0_0001_propreg0_005_run3.dat --term-entropy-reg 0.05 --utterance-entropy-reg 0.0001 --proposal-entropy-reg 0.005 --disable-prosocial
 ```
(launched)

results better than earlier ones, see [images/nosoc_term0_05_utt0_0001_prop0_005.png]

greedy ratio for term seems a little high:
```
term=0.9884 utt=0.6202 prop=0.6427
```

Hmmmm, this is interesting: new graph for nocomms,noprop,soc, at 700k batches: [images/20171115_noprop_nocomms_soc700k.png]
- plateaud for 350k batches, from 150k to 500k
- then started increasing again :P

[images/20171115_noprop_comms_soc400k.png]:
- very high variance
- somewhat plateaud (amongst the variance...)
- ~0.83 test reward
=> high variance might mean ent reg too high?  it is:
```
term=0.6889 utt=0.7950 prop=0.8066
```
maybe term entreg too high?

[images/20171115_prop_nocomms_soc_800k.png]:
- medium variance (not so crazy as for noprop,comms; or for nosoc; but higher than say prop,comms,soc)
- reasonably high result: plateau'd around ~0.92 test
- => matches paper Table 1

Let's start summarizing the results in a table:

|Prop? | Comm? | Soc? | Rend term? | Term reg | Utt reg | Prop reg | Subjective variance | Reward | Greedy ratios | gpu |
|-----|-------|-------|-------------|--------|--------|------------|---------------------|---------|---------------|--|
| Y   | Y     | Y      | Y          | 0.5    | 0.0001 | 0.01   | Low                     | ~0.96 | term=0.7345 utt=0.7635 prop=0.8304 | gpu1,gpu2,gpu6 |
| Y   | N      | Y      | Y         | 0.5    | 0.0001 | 0.01   | Medium-High             | ~0.91 | term=0.6965 utt=0.0000 prop=0.8741 | gpu3 |
| N   | Y      | Y     | Y          | 0.5     | 0.0001 | 0.01  | High                   | ~0.83  | term=0.6889 utt=0.7849 prop=0.8222 | gpu4 |
| N   | N       | Y     | Y         | 0.5      | 0.0001 | 0.01  | Very low              | >= 0.90 (climbing) | term=0.7781 utt=0.0000 prop=0.6006 | gpu5 |
| Y   | Y       | N     | Y         | 0.5      | 0.0001 | 0.01  | Very High             | ~0.25  | term=0.7467 utt=0.9284 prop=0.8137 | was gpu7 |
| Y   | Y       | N     | Y         | 0.05     | 0.0001 | 0.005 | Very Low              | >= 0.80 (climbing) | term=0.9820 utt=0.7040 prop=0.6523 | gpu7 |

Lets kill gpu1,2,6 for now (since rewards already really high...)
(killed)