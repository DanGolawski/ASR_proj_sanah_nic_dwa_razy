from pathlib import Path
from tqdm import tqdm

import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from pyctcdecode import build_ctcdecoder
from wavio import read
import jiwer
import arpa
import random

processor=Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-10k-voxpopuli-ft-pl')
model=Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-base-10k-voxpopuli-ft-pl') #.to('cuda')

files={}
for f in Path('sanah').glob('*.wav'):
    print(f)
    data=read(str(f))
    files[f.stem]=data.data.squeeze().astype('float32')


Fs=data.rate
for name,d in files.items():
    print(f'{name}: {d.size/Fs:0.2f}s')

trans={}
for name,data in tqdm(files.items()):
    feats=processor(data,sampling_rate=Fs,return_tensors='pt',padding=True) #.to('cuda')
    print(name, feats.input_values)
    out=model(input_values=feats.input_values)
    predicted_ids=torch.argmax(out.logits,dim=-1)
    sent=processor.batch_decode(predicted_ids)[0]
    trans[name]=sent

ref={}
with open('sanah/text') as f:
    for l in f:
        tok=l.strip().split()
        ref[tok[0]]=' '.join(tok[1:])


h=[]
r=[]

for name in trans.keys():
    print(f'>>{name}')
    print(trans[name])
    print(ref[name])
    print('')

    h.append(trans[name])
    r.append(ref[name])

print(jiwer.compute_measures(r,h))

digits=['zero','jeden','dwa','trzy','cztery','pięć','sześć','siedem','osiem','dziewięć']
with open('digits.txt','w') as f:
  for l in range(100):
    f.write(' '.join([digits[x] for x in random.sample(range(0, 10), 10)])+'\n')


tokens=[x[0] for x in sorted(processor.tokenizer.get_vocab().items(),key=lambda x:x[1])]
print(tokens)
tokens[4]=' '
print(tokens)

