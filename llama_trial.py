from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers.cache_utils import DynamicCache
import time

model_id = "meta-llama/Llama-3.1-8B"
dtype = torch.bfloat16
device = "cuda"
attn = "sdpa"

B = 16 
S = 512

num_warmups = 10
num_iters = 100

model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, device_map=device, attn_implementation=attn).eval()

D = model.config.hidden_size

# Do prefills to simulate already seen tokens
prefill_embeds = torch.randn(B, S, D, device=model.device, dtype=dtype)
cache = DynamicCache()
with torch.no_grad():
    out = model.model(inputs_embeds = prefill_embeds, past_key_values = cache, use_cache=True)
cache = out.past_key_values
assert(cache.get_seq_length() == S)

# Do batched decodes
decode_embeds = torch.randn(B, 1, D, device=model.device, dtype=dtype)

def step(pos):
    with torch.no_grad():
        model.model(inputs_embeds = decode_embeds, past_key_values = cache, use_cache=True, cache_position=pos)

pos = torch.tensor([cache.get_seq_length()], device=model.device)

for _ in range(num_warmups):
    step(pos)
torch.cuda.synchronize()

start_t = time.perf_counter()
for _ in range(num_iters):
    step(pos)
torch.cuda.synchronize()
elapsed_time = time.perf_counter() - start_t

print("Decode latency: {:.2f} ms".format(elapsed_time * 1000 / num_iters))
