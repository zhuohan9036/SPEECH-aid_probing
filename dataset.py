import json
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from json_file_operations import get_json_content, set_json_file


class AIDDataset(Dataset):
    def __init__(self, data_path, target_sample_rate=16000):
        self.items = get_json_content(data_path)
        self.target_sample_rate = target_sample_rate

    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        item = self.items[idx]
        wave_path = item["wav_path"]
        waveform, cur_sample_rate = torchaudio.load(wave_path)

        # print(f"Loaded waveform shape: {waveform.shape}, sample rate: {cur_sample_rate}")

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if cur_sample_rate != self.target_sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, 
                orig_freq=cur_sample_rate,
                new_freq=self.target_sample_rate
            )
        # print(f"Resampled waveform shape: {waveform.shape}, sample rate: {self.target_sample_rate}")
        
        waveform = waveform.squeeze(0)

        sample = {
            "input_values": waveform,
            "label": torch.tensor(item["label_id"], dtype=torch.long),
            "utt_id": item["utt_id"],
            "speaker_id": item["speaker_id"],
            "accent_label": item["accent_label"],
        }
        return sample
    

def aid_collate_fn(batch):
    input_values = [x["input_values"] for x in batch]
    labels = torch.stack([x["label"] for x in batch])

    input_values = pad_sequence(
        input_values,
        batch_first=True,
        padding_value=0.0,
    )

    attention_mask = torch.zeros_like(input_values, dtype=torch.long)

    for i, x in enumerate(batch):
        length = x["input_values"].shape[0]
        attention_mask[i, :length] = 1

    return {
        "input_values": input_values,
        "attention_mask": attention_mask,
        "labels": labels,
        "utt_ids": [x["utt_id"] for x in batch],
        "speaker_ids": [x["speaker_id"] for x in batch],
        "accent_labels": [x["accent_label"] for x in batch],
    }