"""SERPENTIS PFA1: scratch-built causal transformer. No pretrained weights."""
import math
import torch
from torch import nn
from torch.nn import functional as F

class SERPENTISTransformer(nn.Module):
    def __init__(self, vocab_size=4096, context_length=256, width=256, layers=6, heads=8, dropout=0.1):
        super().__init__()
        if width % heads: raise ValueError("width must be divisible by heads")
        self.context_length=context_length
        self.token_embedding=nn.Embedding(vocab_size,width)
        self.position_embedding=nn.Embedding(context_length,width)
        self.blocks=nn.ModuleList([nn.TransformerEncoderLayer(d_model=width,nhead=heads,dim_feedforward=4*width,dropout=dropout,batch_first=True,norm_first=True,activation="gelu") for _ in range(layers)])
        self.final_norm=nn.LayerNorm(width)
        self.output=nn.Linear(width,vocab_size,bias=False)
        self.output.weight=self.token_embedding.weight
        self.apply(self._init)
    @staticmethod
    def _init(m):
        if isinstance(m,nn.Linear): nn.init.normal_(m.weight,std=0.02); 
        if isinstance(m,nn.Embedding): nn.init.normal_(m.weight,std=0.02)
        if isinstance(m,nn.Linear) and m.bias is not None: nn.init.zeros_(m.bias)
    def forward(self, tokens, targets=None):
        b,t=tokens.shape
        if t>self.context_length: raise ValueError(f"Sequence length {t} exceeds context {self.context_length}")
        x=self.token_embedding(tokens)+self.position_embedding(torch.arange(t,device=tokens.device))
        mask=torch.triu(torch.ones(t,t,device=tokens.device,dtype=torch.bool),diagonal=1)
        for layer in self.blocks: x=layer(x,src_mask=mask)
        logits=self.output(self.final_norm(x))
        loss=F.cross_entropy(logits.reshape(-1,logits.size(-1)),targets.reshape(-1)) if targets is not None else None
        return logits,loss
    @torch.no_grad()
    def generate(self, tokens, max_new_tokens=80, temperature=0.8, top_k=40):
        self.eval()
        for _ in range(max_new_tokens):
            x=tokens[:,-self.context_length:]
            logits,_=self(x); logits=logits[:,-1,:]/max(temperature,1e-4)
            if top_k: logits[logits<torch.topk(logits,top_k).values[:,-1,None]]=float("-inf")
            nxt=torch.multinomial(torch.softmax(logits,dim=-1),1)
            tokens=torch.cat([tokens,nxt],dim=1)
        return tokens
