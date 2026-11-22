"""Minimal scratch pretraining loop. Dataset/tokenizer pipeline plugs in via encoded tokens."""
import os,time,torch
from torch.utils.data import Dataset,DataLoader
from .model import SERPENTISTransformer

class TokenWindows(Dataset):
    def __init__(self, tokens, context):
        self.tokens=torch.as_tensor(tokens,dtype=torch.long);self.context=context
    def __len__(self): return max(0,len(self.tokens)-self.context-1)
    def __getitem__(self,i):
        chunk=self.tokens[i:i+self.context+1]
        return chunk[:-1],chunk[1:]

def train(model, tokens, *, steps=1000, batch_size=16, context=256, lr=3e-4, device=None, log_every=50, save_path=None):
    device=device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device); ds=TokenWindows(tokens,context)
    if len(ds)==0: raise ValueError("Not enough tokenized data for requested context")
    loader=DataLoader(ds,batch_size=batch_size,shuffle=True,drop_last=True)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=0.1)
    scaler=torch.amp.GradScaler("cuda",enabled=device.startswith("cuda"))
    model.train();step=0;start=time.time()
    while step<steps:
        for x,y in loader:
            x,y=x.to(device),y.to(device);opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda",dtype=torch.float16,enabled=device.startswith("cuda")):
                _,loss=model(x,y)
            scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);scaler.step(opt);scaler.update();step+=1
            if step%log_every==0: print(f"step {step}/{steps} loss {loss.item():.4f} elapsed {time.time()-start:.1f}s")
            if save_path and step%log_every==0:
                os.makedirs(os.path.dirname(save_path) or ".",exist_ok=True)
                torch.save({"step":step,"model":model.state_dict(),"optimizer":opt.state_dict(),"loss":loss.item()},save_path)
            if step>=steps: break
    return model
