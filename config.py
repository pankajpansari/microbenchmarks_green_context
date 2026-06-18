S_dec= 2048 # sequence len
D = 8192 # model embedding dim
N = 32 # number of heads (same for K,Q,V)
H = D // N # head dimension
F = 4 * D # FFN hidden dimension
B_dec_sweep =[32, 64, 128, 256, 512]
S_prefill_sweep = [512, 1024, 2048]
B_prefill = 8

