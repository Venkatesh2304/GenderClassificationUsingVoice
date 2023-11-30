import math, random
import pickle
import librosa
from matplotlib import pyplot as plt
import torch
import torchaudio
from torchaudio import transforms
from IPython.display import Audio
from torch.utils.data import random_split
import os 
import pandas as pd
from torch.utils.data import DataLoader, Dataset, random_split
import torchaudio
from tqdm import tqdm
import torchaudio.transforms as T


data_path = "dataset/dataset/"
gender_file = "dataset/voice.tsv"
   
# ----------------------------
# Sound Dataset
# ----------------------------

gender_map = {"male" : 0 , "female" : 1}

class SoundDS(Dataset):
  def __init__(self, df, data_path):
    self.df = df
    self.data_path = str(data_path)
    self.duration = 4000
    self.sr = 48000 
    self.channel = 1
    self.shift_pct = 0.4
    self.dataset = []
    self.load() 
   
  def load(self) :         
    for idx in tqdm( range( len(self) ) ) :
        audio_file = self.data_path + self.df.loc[idx, 'path']
        class_id = gender_map[ self.df.loc[idx, 'gender'] ]
        
        aud = AudioUtil.open(audio_file)
        rechan = AudioUtil.resample(aud, self.sr)
        dur_aud = AudioUtil.pad_trunc(rechan, self.duration)
        shift_aud = AudioUtil.time_shift(dur_aud, self.shift_pct)
        sgram = AudioUtil.spectro_gram(shift_aud, n_mels=64, n_fft=1024, hop_len=None)
        aug_sgram = AudioUtil.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2) 
        self.dataset.append((aug_sgram, class_id))
     
  def __len__(self):
     return len(self.df)    
 
  def __getitem__(self, idx): 
      return self.dataset[idx]
     
class AudioUtil():
  # ----------------------------
  # Load an audio file. Return the signal as a tensor and the sample rate
  # ----------------------------
  @staticmethod
  def open(audio_file): 
    sig, sr = torchaudio.load(audio_file)
    return (sig, sr)
  
  @staticmethod
  def rechannel(aud, new_channel):
    sig, sr = aud

    if (sig.shape[0] == new_channel):
      # Nothing to do
      return aud

    if (new_channel == 1):
      # Convert from stereo to mono by selecting only the first channel
      resig = sig[:1, :]
    else:
      # Convert from mono to stereo by duplicating the first channel
      resig = torch.cat([sig, sig])

    return ((resig, sr))

  @staticmethod
  def resample(aud, newsr):
      sig, sr = aud
  
      if (sr == newsr):
        # Nothing to do
        return aud
  
      num_channels = sig.shape[0]
      # Resample first channel
      resig = torchaudio.transforms.Resample(sr, newsr)(sig[:1,:])
      if (num_channels > 1):
        # Resample the second channel and merge both channels
        retwo = torchaudio.transforms.Resample(sr, newsr)(sig[1:,:])
        resig = torch.cat([resig, retwo])
  
      return ((resig, newsr))
  
  @staticmethod
  def pad_trunc(aud, max_ms):
      sig, sr = aud
      num_rows, sig_len = sig.shape
      max_len = sr//1000 * max_ms
  
      if (sig_len > max_len):
        # Truncate the signal to the given length
        sig = sig[:,:max_len]
  
      elif (sig_len < max_len):
        # Length of padding to add at the beginning and end of the signal
        pad_begin_len = random.randint(0, max_len - sig_len)
        pad_end_len = max_len - sig_len - pad_begin_len
  
        # Pad with 0s
        pad_begin = torch.zeros((num_rows, pad_begin_len))
        pad_end = torch.zeros((num_rows, pad_end_len))
  
        sig = torch.cat((pad_begin, sig, pad_end), 1)
      return (sig, sr)

  @staticmethod
  def time_shift(aud, shift_limit):
      sig,sr = aud
      _, sig_len = sig.shape
      shift_amt = int(random.random() * shift_limit * sig_len)
      return (sig.roll(shift_amt), sr)      
  
  @staticmethod
  def spectro_gram(aud, n_mels=64, n_fft=1024, hop_len=None):
      sig,sr = aud
      top_db = 80
  
      # spec has shape [channel, n_mels, time], where channel is mono, stereo etc
      spec = transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)

      # Convert to decibels
      spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
      return (spec)
  
  @staticmethod
  def spectro_augment(spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
      _, n_mels, n_steps = spec.shape
      mask_value = spec.mean()
      aug_spec = spec
  
      freq_mask_param = max_mask_pct * n_mels
      for _ in range(n_freq_masks):
        aug_spec = transforms.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)
  
      time_mask_param = max_mask_pct * n_steps
      for _ in range(n_time_masks):
        aug_spec = transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)
      return aug_spec


files = list(os.listdir(data_path))
df = pd.read_csv(gender_file,sep="\t")
df = df[ df.path.isin(files) & df.gender.isin( gender_map.keys() ) ].reset_index(drop=True)
myds = SoundDS( df , data_path)
pickle.dump(myds,open("dataset.bin","wb+"))


  
  
  