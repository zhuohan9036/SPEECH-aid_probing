import json
import torch
import torchaudio
import random
import math

from pathlib import Path
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from json_file_operations import get_json_content, set_json_file

try:
    import parselmouth
    from parselmouth.praat import call
    PARSELMOUTH_AVAILABLE = True
except ImportError:
    PARSELMOUTH_AVAILABLE = False


class VoicePerturbation:
    def __init__(self, 
                 sample_rate=16000,
                 apply_prob=0.75,
                 beta_min=1.0,
                 beta_max=1.4,
            ):
        self.sample_rate = sample_rate
        self.apply_prob = apply_prob
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.pitch_cache = {}

    def sample_beta(self):
        beta = random.uniform(self.beta_min, self.beta_max)
        if random.random() < 0.5:
            beta = 1.0 / beta
        return beta

    def apply_formant_scaling(self, waveform):
        if not PARSELMOUTH_AVAILABLE:
            return waveform
        beta = self.sample_beta()
        original_length = waveform.shape[0]
        original_device = waveform.device
        original_dtype = waveform.dtype

        try:
            wav_np = waveform.detach().cpu().float().numpy()
            sound = parselmouth.Sound(wav_np, self.sample_rate)
            shifted = call(
                sound,
                "Change gender",
                75,      # pitch floor
                600,     # pitch ceiling
                beta,    # formant shift ratio
                0,       # new pitch median; 0 means preserve original pitch
                1,       # pitch range factor; 1 means preserve pitch range
                1,       # duration factor; 1 means preserve duration
            )

            shifted_np = shifted.values.squeeze()
            shifted_tensor = torch.tensor(
                shifted_np, 
                dtype=original_dtype,
                device=original_device,    
            )
            shifted_tensor = self.fix_length(shifted_tensor, original_length)

            max_val = shifted_tensor.abs().max()
            if max_val > 1.0:
                shifted_tensor = shifted_tensor / max_val

            return shifted_tensor
        except Exception as e:
            return waveform
        
    
    def apply_pitch_shift(self, waveform):
        # beta = self.sample_beta()

        # cents = 1200.0 * math.log2(beta)
        # wav = waveform.unsqueeze(0)  # (1, num_samples)

        # effects = [
        #     ["pitch", f"{cents}"],
        #     ["rate", f"{self.sample_rate}"],
        # ]

        # wav, _ = torchaudio.sox_effects.apply_effects_tensor(
        #     wav, self.sample_rate, effects
        # )
        # return wav.squeeze(0)
        beta = self.sample_beta()
        n_steps = 12.0 * math.log2(beta)
        n_steps = round(n_steps * 2) / 2.0
        if n_steps not in self.pitch_cache:
            self.pitch_cache[n_steps] = torchaudio.transforms.PitchShift(
                sample_rate=self.sample_rate,
                n_steps=n_steps,
            )
        wav = waveform.unsqueeze(0)  # (1, num_samples)
        # device = wav.device
        # wav = wav.cpu()  
        # pitch_shift = torchaudio.transforms.PitchShift(
        #     sample_rate=self.sample_rate,
        #     n_steps=n_steps,
        # )
        with torch.no_grad():
            # wav = pitch_shift(wav)
            wav = self.pitch_cache[n_steps](wav)

        # wav = wav.to(device)

        return wav.squeeze(0)
    
    def apply_random_equalizer(self, waveform):
        # wav = waveform.unsqueeze(0)  # (1, num_samples)
        # effects = []
        # num_filters = random.randint(1, 3)
        # for _ in range(num_filters):
        #     cent_freq = random.uniform(100, 7000)
        #     q = random.uniform(0.5, 2.0)
        #     gain = random.uniform(-6.0, 6.0)
        #     effects.append(["equalizer", f"{cent_freq}", f"{q}", f"{gain}"])
        # effects.append(["rate", f"{self.sample_rate}"])
        # wav, _ = torchaudio.sox_effects.apply_effects_tensor(
        #     wav, self.sample_rate, effects
        # )
        # return wav.squeeze(0)
        wav = waveform.unsqueeze(0)  # (1, num_samples)
        num_filters = random.randint(1, 3)

        for _ in range(num_filters):
            center_freq = random.uniform(100.0, 7000.0)
            q = random.uniform(0.5, 2.0)
            gain = random.uniform(-6.0, 6.0)

            wav = torchaudio.functional.equalizer_biquad(
                wav,
                sample_rate=self.sample_rate,
                center_freq=center_freq,
                Q=q,
                gain=gain,
            )
        return wav.squeeze(0)
    
    def fix_length(self, waveform, target_length):
        if waveform.shape[0] > target_length:
            return waveform[:target_length]
        elif waveform.shape[0] < target_length:
            padding = target_length - waveform.shape[0]
            return torch.nn.functional.pad(waveform, (0, padding))
        return waveform
    
    def __call__(self, waveform):
        if random.random() > self.apply_prob:
            return waveform
        original_length = waveform.shape[0]
        waveform = self.apply_formant_scaling(waveform)
        waveform = self.apply_pitch_shift(waveform)
        waveform = self.apply_random_equalizer(waveform)
        waveform = self.fix_length(waveform, original_length)
        return waveform
    


class AIDDataset(Dataset):
    def __init__(self, data_path, apply_perturbation=False, target_sample_rate=16000, perturbation_prob=0.75):
        self.items = get_json_content(data_path)
        self.apply_perturbation = apply_perturbation
        self.target_sample_rate = target_sample_rate
        self.perturbation_prob = perturbation_prob

        self.perturb = VoicePerturbation(sample_rate=target_sample_rate, apply_prob=self.perturbation_prob)

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
        
        waveform = waveform.squeeze(0).float()

        if self.apply_perturbation:
            waveform = self.perturb(waveform)

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