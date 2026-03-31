isqwerty = True
embed_size = 300
encoder_num_banks = 16
num_highwaynet_blocks = 4
maxlen = 50
minlen = 10
norm_type = "bn"
dropout_rate = 0.5
lr = 0.0001
logdir = "log/qwerty" if isqwerty else "log/nine"
batch_size = 200
num_epochs = 30

