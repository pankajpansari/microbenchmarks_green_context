import torch
import torch.nn.functional as Func
import flashinfer

class Model:
  def __init__(self, D, N):
    H, F = D // N, 4 *D
    self.D, self.N, self.H, self.F = D, N, H, F
    self.W_k = torch.randn((D, N, H), dtype = torch.float16, device = "cuda")
    self.W_q = torch.randn((D, N, H), dtype = torch.float16, device = "cuda")
    self.W_v = torch.randn((D, N, H), dtype = torch.float16, device = "cuda")
    self.W_o = torch.randn((N, H, D), dtype = torch.float16, device = "cuda")

    self.W_in = torch.randn((D, F), dtype = torch.float16, device = "cuda")
    self.W_out = torch.randn((F, D), dtype = torch.float16, device = "cuda")

  def do_serial_prefill(self, x):
    
    # Iterate over samples in batch
    # self-attention block
    output = torch.zeros_like(x)

    for i in range(x.shape[0]):
      q_i = torch.einsum('sd, dnh -> snh', x[i], self.W_q)   

      k_i = torch.einsum('sd, dnh -> snh', x[i], self.W_k)   
      v_i = torch.einsum('sd, dnh -> snh', x[i], self.W_v)   
      
      v_attention_i = flashinfer.single_prefill_with_kv_cache(q_i, k_i, v_i, causal = True)

      o_i = torch.einsum('snh, nhd -> sd', v_attention_i, self.W_o)

      # ffn block
      o_1_i = torch.einsum('sd, df -> sf', o_i, self.W_in)
      o_1_i = Func.relu(o_1_i)
      output[i] = torch.einsum('sf, fd -> sd', o_1_i, self.W_out)
    return output

  def do_batched_prefill(self, x, prefill_wrapper):
    
    #x is a batch of prefill token embedding 
    b, s = x.shape[0], x.shape[1]
    q = torch.einsum('bsd, dnh -> bsnh', x, self.W_q).reshape(b * s, self.N, self.H)

    #we don't append these to kv cache
    k = torch.einsum('bsd, dnh -> bsnh', x, self.W_k).reshape(b * s, self.N, self.H)
    v = torch.einsum('bsd, dnh -> bsnh', x, self.W_v).reshape(b * s, self.N, self.H)

    v_attention = prefill_wrapper.run(q, k, v).reshape(b, s, self.N, self.H)

    o = torch.einsum('bsnh, nhd -> bsd', v_attention, self.W_o) 

    #ffn block
    o_1 = torch.einsum('bsd, df -> bsf', o, self.W_in) 
    o_1 = Func.relu(o_1)
    output = torch.einsum('bsf, fd -> bsd', o_1, self.W_out) 
    return output


  def do_batched_decode(self, x, paged_kv, decode_wrapper):
    
    # context len always at S_dec; cache does not grow

    #x is a batch of last generated token embedding 
    q = torch.einsum('bsd, dnh -> bsnh', x, self.W_q).squeeze(1)   # s = 1 

    #we don't append these to kv cache
    k = torch.einsum('bsd, dnh -> bsnh', x, self.W_k).squeeze(1)   # s = 1 
    v = torch.einsum('bsd, dnh -> bsnh', x, self.W_v).squeeze(1)   # s = 1 

    v_attention = decode_wrapper.run(q, paged_kv).unsqueeze(1)

    o = torch.einsum('bsnh, nhd -> bsd', v_attention, self.W_o) # s = 1 

    #ffn block
    o_1 = torch.einsum('bsd, df -> bsf', o, self.W_in) # s = 1 
    o_1 = Func.relu(o_1)
    output = torch.einsum('bsf, fd -> bsd', o_1, self.W_out) # s = 1 
    return output
