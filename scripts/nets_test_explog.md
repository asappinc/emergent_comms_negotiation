term-policy
-----------

effect of entropy reg:
```
(root) /emergent_comms_negotiation (master|…2△1) $ python nets_test.py test-context --term-entropy-reg 1
episode 0 num_right 2 baseline 0.3
episode 100 num_right 2 baseline 0.6818555706423856
episode 200 num_right 2 baseline 0.9938958233414137
episode 300 num_right 2 baseline 0.7746209980485705
episode 400 num_right 2 baseline 0.9444644898336487
episode 500 num_right 1 baseline 0.545179411787141
episode 600 num_right 2 baseline 0.8696337750631387
episode 700 num_right 2 baseline 0.8364079471631221
episode 800 num_right 2 baseline 0.9973736089014604
episode 900 num_right 2 baseline 0.9870381382361351
episode 1000 num_right 2 baseline 0.9552785890972033
episode 1100 num_right 2 baseline 0.946063538655197
episode 1200 num_right 2 baseline 0.9990292805932697
episode 1300 num_right 2 baseline 0.9594112885125148
episode 1400 num_right 1 baseline 0.7311748112691385
episode 1500 num_right 2 baseline 0.9609664419399175
episode 1600 num_right 2 baseline 0.9358127652652215
episode 1700 num_right 2 baseline 0.979113037255114
episode 1800 num_right 2 baseline 0.9386126065203

(root) /emergent_comms_negotiation (master|…2△1) $ python nets_test.py test-context --term-entropy-reg 0.1
episode 0 num_right 2 baseline 0.3
episode 100 num_right 2 baseline 0.9407348088613829
episode 200 num_right 2 baseline 0.9999707292129745
episode 300 num_right 2 baseline 0.9999983430344679
episode 400 num_right 2 baseline 0.9999951701152274
episode 500 num_right 2 baseline 0.998375681416114
episode 600 num_right 2 baseline 0.9999999839489591
episode 700 num_right 2 baseline 0.9985458530895528
episode 800 num_right 2 baseline 0.9999999999999996

(root) /emergent_comms_negotiation (master|…2△1) $ python nets_test.py test-context --term-entropy-reg 0.05
episode 0 num_right 1 baseline 0.15
episode 100 num_right 1 baseline 0.7706678249350699
episode 200 num_right 1 baseline 0.8478791655530046
episode 300 num_right 2 baseline 0.9999541170201343
episode 400 num_right 2 baseline 0.9999999999992613
episode 500 num_right 2 baseline 0.9999999999999996
episode 600 num_right 2 baseline 0.9999900963496178
episode 700 num_right 2 baseline 0.9989826653907263
```

But hey, just two proposals. So, boosted this up to:
- 128 examples
- proposals are 3 items each drawn from {0,1,..,5}
- embedding size 100 (also tried 50)

It sort of learns, but never gets beyond ~83% right:

Also, high variance:
```
(root) /emergent_comms_negotiation (master|…2△3) $ python nets_test.py test-context --term-entropy-reg 0.1
episode 0 num_right 70 baseline 0.1640625 reward_val 0.5625
episode 100 num_right 96 baseline 0.7576236589075325 reward_val 0.7578125
episode 200 num_right 98 baseline 0.7708004353328515 reward_val 0.7734375
episode 300 num_right 99 baseline 0.7769226017140616 reward_val 0.7734375
episode 400 num_right 99 baseline 0.7707928992849675 reward_val 0.7734375
episode 500 num_right 99 baseline 0.7785308190356455 reward_val 0.7734375
episode 600 num_right 99 baseline 0.7755032680081083 reward_val 0.7734375
episode 700 num_right 99 baseline 0.7730324483130008 reward_val 0.7734375

(root) /emergent_comms_negotiation (master|…2△3) $ python nets_test.py test-context --term-entropy-reg 0.1
episode 0 num_right 69 baseline 0.16171875 reward_val 0.484375
episode 100 num_right 103 baseline 0.8181930018919761 reward_val 0.8203125
episode 200 num_right 108 baseline 0.8395301623313429 reward_val 0.8359375
episode 300 num_right 108 baseline 0.8294853960989927 reward_val 0.8359375
episode 400 num_right 107 baseline 0.8369697603905433 reward_val 0.8359375
episode 500 num_right 107 baseline 0.8337040492137351 reward_val 0.8359375

(root) /emergent_comms_negotiation (master|…2△3) $ python nets_test.py test-context --term-entropy-reg 0.1
episode 0 num_right 64 baseline 0.15 reward_val 0.5390625
episode 100 num_right 100 baseline 0.7707670484877688 reward_val 0.7734375
episode 200 num_right 101 baseline 0.7793422309386008 reward_val 0.7734375
episode 300 num_right 101 baseline 0.7889988730741544 reward_val 0.78125
episode 400 num_right 99 baseline 0.7799481699197561 reward_val 0.78125
episode 500 num_right 100 baseline 0.7786163327214455 reward_val 0.78125
episode 600 num_right 99 baseline 0.7716311929262197 reward_val 0.78125

(root) /emergent_comms_negotiation (master|…2△3) $ python nets_test.py test-context --term-entropy-reg 1
episode 0 num_right 68 baseline 0.159375 reward_val 0.4921875
episode 100 num_right 91 baseline 0.7409719498845829 reward_val 0.84375
episode 200 num_right 97 baseline 0.7612036510910635 reward_val 0.8515625
episode 300 num_right 96 baseline 0.7652133641604445 reward_val 0.8515625
episode 400 num_right 105 baseline 0.7870302031982915 reward_val 0.8515625
episode 500 num_right 103 baseline 0.7807035019770987 reward_val 0.8515625
episode 600 num_right 108 baseline 0.8099813406271723 reward_val 0.8515625
episode 700 num_right 98 baseline 0.7788512047029248 reward_val 0.8515625
episode 800 num_right 105 baseline 0.7982152154731718 reward_val 0.8515625
episode 900 num_right 100 baseline 0.806445717884745 reward_val 0.8515625
episode 1000 num_right 101 baseline 0.7783504904000135 reward_val 0.8515625
episode 1100 num_right 100 baseline 0.7945617332561018 reward_val 0.8515625
episode 1200 num_right 100 baseline 0.7945105563895009 reward_val 0.8515625
episode 1300 num_right 101 baseline 0.79523965390516 reward_val 0.8515625

(root) /emergent_comms_negotiation (master|…2△3) $ python nets_test.py test-context --term-entropy-reg 1
episode 0 num_right 71 baseline 0.16640625 reward_val 0.484375
episode 100 num_right 103 baseline 0.8004367973925395 reward_val 0.8984375
episode 200 num_right 105 baseline 0.8128337014200406 reward_val 0.8828125
episode 300 num_right 104 baseline 0.8061079417926523 reward_val 0.8984375
episode 400 num_right 102 baseline 0.8234150558445549 reward_val 0.890625
episode 500 num_right 110 baseline 0.8275847413792814 reward_val 0.890625
episode 600 num_right 111 baseline 0.8168115856730668 reward_val 0.890625
episode 700 num_right 106 baseline 0.8172801537882965 reward_val 0.8828125
episode 800 num_right 110 baseline 0.8508826184559768 reward_val 0.8984375

(root) /emergent_comms_negotiation (master|…2△3) $ python nets_test.py test-context --term-entropy-reg 1
episode 0 num_right 60 baseline 0.140625 reward_val 0.5
episode 100 num_right 103 baseline 0.7796356521437218 reward_val 0.90625
episode 200 num_right 103 baseline 0.8162637534023536 reward_val 0.90625
episode 300 num_right 105 baseline 0.8264842469940193 reward_val 0.90625
episode 400 num_right 103 baseline 0.7992952603289691 reward_val 0.8984375
episode 500 num_right 109 baseline 0.8355215841159587 reward_val 0.90625
episode 600 num_right 108 baseline 0.8178750579159226 reward_val 0.90625
episode 700 num_right 106 baseline 0.8314176415117113 reward_val 0.90625
```

utteracne policy
----------------

using utterance entropy reg 0.001:
```
$ python nets_test.py test-utterance-policy
episode 0 letter acc 0.085 baseline 0.152 reward_greedy 0.105
episode 100 letter acc 0.108 baseline 0.636 reward_greedy 0.130
episode 200 letter acc 0.134 baseline 0.771 reward_greedy 0.135
episode 300 letter acc 0.145 baseline 0.857 reward_greedy 0.139
episode 400 letter acc 0.158 baseline 0.871 reward_greedy 0.145
episode 500 letter acc 0.150 baseline 0.876 reward_greedy 0.155
episode 600 letter acc 0.147 baseline 0.888 reward_greedy 0.160
episode 700 letter acc 0.154 baseline 0.947 reward_greedy 0.152
episode 800 letter acc 0.163 baseline 0.981 reward_greedy 0.167
episode 900 letter acc 0.165 baseline 0.996 reward_greedy 0.172
episode 1000 letter acc 0.174 baseline 1.026 reward_greedy 0.172
episode 1100 letter acc 0.172 baseline 1.055 reward_greedy 0.182
episode 1200 letter acc 0.184 baseline 1.111 reward_greedy 0.193
episode 1300 letter acc 0.190 baseline 1.123 reward_greedy 0.193
episode 1400 letter acc 0.194 baseline 1.166 reward_greedy 0.201
episode 1500 letter acc 0.201 baseline 1.205 reward_greedy 0.211
episode 1600 letter acc 0.210 baseline 1.237 reward_greedy 0.208
episode 1700 letter acc 0.206 baseline 1.214 reward_greedy 0.204
episode 1800 letter acc 0.212 baseline 1.245 reward_greedy 0.208
episode 1900 letter acc 0.212 baseline 1.261 reward_greedy 0.214
episode 2000 letter acc 0.214 baseline 1.288 reward_greedy 0.220
```

start it running on gpu1:
```
python nets_test.py test-utterance-policy
```

(need to add logging and model save really...)

(and cuda...)

added cuda, relaunchgin on gpu1:
```
python nets_test.py test-utterance-policy --enable-cuda
```

after some time, looks like:
```
episode 20700 letter acc 0.242 baseline 1.451 reward_greedy 0.242
episode 20800 letter acc 0.245 baseline 1.473 reward_greedy 0.249
episode 20900 letter acc 0.243 baseline 1.458 reward_greedy 0.247
episode 21000 letter acc 0.241 baseline 1.442 reward_greedy 0.241
episode 21100 letter acc 0.243 baseline 1.461 reward_greedy 0.245
episode 21200 letter acc 0.249 baseline 1.487 reward_greedy 0.247
episode 21300 letter acc 0.238 baseline 1.436 reward_greedy 0.240
episode 21400 letter acc 0.238 baseline 1.444 reward_greedy 0.243
episode 21500 letter acc 0.243 baseline 1.453 reward_greedy 0.245
episode 21600 letter acc 0.243 baseline 1.465 reward_greedy 0.246
episode 21700 letter acc 0.251 baseline 1.487 reward_greedy 0.247
episode 21800 letter acc 0.245 baseline 1.482 reward_greedy 0.249
episode 21900 letter acc 0.249 baseline 1.491 reward_greedy 0.249

episode 43200 letter acc 0.240 baseline 1.437 reward_greedy 0.240
episode 43300 letter acc 0.240 baseline 1.437 reward_greedy 0.240
episode 43400 letter acc 0.240 baseline 1.437 reward_greedy 0.240
episode 43500 letter acc 0.240 baseline 1.437 reward_greedy 0.240
episode 43600 letter acc 0.240 baseline 1.437 reward_greedy 0.240
episode 43700 letter acc 0.240 baseline 1.437 reward_greedy 0.240
episode 43800 letter acc 0.240 baseline 1.437 reward_greedy 0.240
episode 43900 letter acc 0.240 baseline 1.437 reward_greedy 0.240
episode 44000 letter acc 0.240 baseline 1.437 reward_greedy 0.240
episode 44100 letter acc 0.240 baseline 1.437 reward_greedy 0.240
episode 44200 letter acc 0.240 baseline 1.437 reward_greedy 0.240
episode 44300 letter acc 0.240 baseline 1.437 reward_greedy 0.240
episode 44400 letter acc 0.240 baseline 1.437 reward_greedy 0.240
episode 44500 letter acc 0.240 baseline 1.437 reward_greedy 0.240
episode 44600 letter acc 0.240 baseline 1.437 reward_greedy 0.240
episode 44700 letter acc 0.240 baseline 1.437 reward_greedy 0.240
```

=> only 25% of letters correct

killing it, since not improving

proposal policy
---------------

test added, try running:
```
$ python nets_test.py test-proposal-policy
episode 0 propitemacc 0.182 baseline 0.164 reward_greedy 0.185
episode 100 propitemacc 0.271 baseline 0.767 reward_greedy 0.299
episode 200 propitemacc 0.349 baseline 1.068 reward_greedy 0.432
episode 300 propitemacc 0.461 baseline 1.342 reward_greedy 0.497
episode 400 propitemacc 0.521 baseline 1.503 reward_greedy 0.552
episode 500 propitemacc 0.549 baseline 1.662 reward_greedy 0.604
episode 600 propitemacc 0.599 baseline 1.765 reward_greedy 0.625
episode 700 propitemacc 0.617 baseline 1.828 reward_greedy 0.643
episode 800 propitemacc 0.630 baseline 1.913 reward_greedy 0.667
episode 900 propitemacc 0.648 baseline 1.928 reward_greedy 0.661
episode 1000 propitemacc 0.656 baseline 1.962 reward_greedy 0.669
episode 1100 propitemacc 0.656 baseline 1.979 reward_greedy 0.677
episode 1200 propitemacc 0.661 baseline 1.987 reward_greedy 0.677
episode 1300 propitemacc 0.669 baseline 2.009 reward_greedy 0.680
episode 1400 propitemacc 0.667 baseline 1.998 reward_greedy 0.677
```

=> learning.

lets send it to gpu1, and run in parallel with the utterance policy
```
python nets_test.py test-proposal-policy --enable-cuda
```

after some time looks like:
```
episode 46800 propitemacc 0.409 baseline 1.227 reward_greedy 0.409
episode 46900 propitemacc 0.409 baseline 1.227 reward_greedy 0.409
episode 47000 propitemacc 0.409 baseline 1.227 reward_greedy 0.409
episode 47100 propitemacc 0.409 baseline 1.227 reward_greedy 0.409
episode 47200 propitemacc 0.409 baseline 1.227 reward_greedy 0.409
episode 47300 propitemacc 0.409 baseline 1.227 reward_greedy 0.409
episode 47400 propitemacc 0.409 baseline 1.227 reward_greedy 0.409
episode 47500 propitemacc 0.409 baseline 1.227 reward_greedy 0.409
episode 47600 propitemacc 0.409 baseline 1.227 reward_greedy 0.409
episode 47700 propitemacc 0.409 baseline 1.227 reward_greedy 0.409
episode 47800 propitemacc 0.409 baseline 1.227 reward_greedy 0.409
episode 47900 propitemacc 0.409 baseline 1.227 reward_greedy 0.409
episode 48000 propitemacc 0.409 baseline 1.227 reward_greedy 0.409
episode 48100 propitemacc 0.409 baseline 1.227 reward_greedy 0.409
episode 48200 propitemacc 0.409 baseline 1.227 reward_greedy 0.409
episode 48300 propitemacc 0.409 baseline 1.227 reward_greedy 0.409
episode 48400 propitemacc 0.409 baseline 1.227 reward_greedy 0.409
episode 48500 propitemacc 0.409 baseline 1.227 reward_greedy 0.409
episode 48600 propitemacc 0.409 baseline 1.227 reward_greedy 0.409

episode 79000 propitemacc 0.396 baseline 1.187 reward_greedy 0.396
episode 79100 propitemacc 0.396 baseline 1.187 reward_greedy 0.396
episode 79200 propitemacc 0.396 baseline 1.187 reward_greedy 0.396
episode 79300 propitemacc 0.396 baseline 1.187 reward_greedy 0.396
episode 79400 propitemacc 0.396 baseline 1.187 reward_greedy 0.396
episode 79500 propitemacc 0.396 baseline 1.187 reward_greedy 0.396
episode 79600 propitemacc 0.396 baseline 1.187 reward_greedy 0.396
episode 79700 propitemacc 0.396 baseline 1.187 reward_greedy 0.396
episode 79800 propitemacc 0.396 baseline 1.187 reward_greedy 0.396
episode 79900 propitemacc 0.396 baseline 1.187 reward_greedy 0.396
```

=> only 40% of proposal quantities correct

killing it, since not improving
