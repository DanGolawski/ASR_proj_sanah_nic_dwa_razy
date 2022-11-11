from pathlib import Path
from tqdm import tqdm

import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from pyctcdecode import build_ctcdecoder
from wavio import read
import jiwer

processor=Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-10k-voxpopuli-ft-pl')
model=Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-base-10k-voxpopuli-ft-pl')

files={}
for f in Path('wyznanie').glob('*.wav'):
    print(f)
    data=read(str(f))
    files[f.stem]=data.data.squeeze().astype('float32')


Fs=data.rate
for name,d in files.items():
    print(f'{name}: {d.size/Fs:0.2f}s')

trans={}
for name,data in tqdm(files.items()):
    feats=processor(data,sampling_rate=Fs,return_tensors='pt',padding=True)
    out=model(input_values=feats.input_values)
    predicted_ids=torch.argmax(out.logits,dim=-1)
    sent=processor.batch_decode(predicted_ids)[0]
    trans[name]=sent

ref={}


h=[]
r=[]

print('trans',trans)
for idx in range(1, len(trans.keys())+1):
    name = f'ania{idx}'
    print('-------------------')
    print(f'>>{name}')
    print(trans[name])
    print('')

    h.append(trans[name])


