import subprocess, json, sys, datetime
import torch
import os

def generate_csv_header_sidecar(df, params, out_dir = "data"):
  
  commit_hash = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
  dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, 
                         text=True).stdout.strip())
  gpu = torch.cuda.get_device_name(0)
  now = datetime.datetime.now()
  timestamp = now.isoformat(timespec='seconds')
  command = " ".join(sys.argv)

  stamp = now.strftime("%Y%m%d-%H%M")
  gpu_slug = torch.cuda.get_device_name(0).replace(" ", "-")
  csv_filename = f"{out_dir}/{params['exp']}__d{params['D']}__{gpu_slug}__{stamp}.csv"

  meta = {"commit_hash": commit_hash, "dirty": dirty, "gpu": gpu, "timestamp": timestamp, 
          "command": command, "params": params}
  
  os.makedirs(out_dir, exist_ok = True)
  with open(f"{csv_filename}.meta.json", "w") as f:
    json.dump(meta, f, indent = 2)  

  header = "# " + " ".join(f"{k}={meta[k]}" for k in ["commit_hash", "dirty", "gpu", "timestamp"]) 

  with open(csv_filename, "w", newline="") as f:
    f.write(header + "\n")
    df.to_csv(f, index=False)