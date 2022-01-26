python3 __main__.py --patient procenko --mode regression --model SimpleNetBase_WithLSTM__CNANNELS_6_12__LAG_1000_0__40MELS --runs_count 1 &&\
python3 __main__.py --patient procenko --mode classification --model SimpleNetBase_WithLSTM__CNANNELS_6_12__LAG_1000_0__40MELS &&\
python3 __main__.py --patient ivanova --mode regression --model SimpleNetBase_WithLSTM__CNANNELS_8_16__LAG_1000_0__40MELS --runs_count 1 &&\
python3 __main__.py --patient ivanova --mode classification --model SimpleNetBase_WithLSTM__CNANNELS_8_16__LAG_1000_0__40MELS

