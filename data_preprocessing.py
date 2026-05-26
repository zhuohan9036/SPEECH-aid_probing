from parameters import parse
from pathlib import Path
from json_file_operations import get_json_content, set_json_file
from collections import Counter, defaultdict

import re
import json

DATASET_REGISTRY = {
    "l2arctic": {
        "path": Path("~/speech/l2arctic").expanduser(),
        "accent_labels": {
            "ABA": "Arabic",
            "SKA": "Arabic",
            "YBAA": "Arabic",
            "ZHAA": "Arabic",
            "BWC": "Mandarin",
            "LXC": "Mandarin",
            "NCC": "Mandarin",
            "TXHC": "Mandarin",
            "ASI": "Hindi",
            "RRBI": "Hindi",
            "SVBI": "Hindi",
            "TNI": "Hindi",
            "HJK": "Korean",
            "HKK": "Korean",
            "YDCK": "Korean",
            "YKWK": "Korean",
            "EBVS": "Spanish",
            "ERMS": "Spanish",
            "MBMPS": "Spanish",
            "NJS": "Spanish",
            "HQTV": "Vietnamese",
            "PNV": "Vietnamese",
            "THV": "Vietnamese",
            "TLV": "Vietnamese",
        },
        "speaker_split": {
            "ABA": "aid_train",
            "SKA": "aid_test",
            "YBAA": "aid_train",
            "ZHAA": "aid_train",
            "BWC": "aid_test",
            "LXC": "aid_train",
            "NCC": "aid_train",
            "TXHC": "aid_train",
            "ASI": "aid_train",
            "RRBI": "aid_train",
            "SVBI": "aid_test",
            "TNI": "aid_train",
            "HJK": "aid_train",
            "HKK": "aid_test",
            "YDCK": "aid_train",
            "YKWK": "aid_train",
            "EBVS": "aid_train",
            "ERMS": "aid_train",
            "MBMPS": "aid_train",
            "NJS": "aid_test",
            "HQTV": "aid_test",
            "PNV": "aid_train",
            "THV": "aid_train",
            "TLV": "aid_train",
        },
        "transcript_path": "/transcript/",
        "wav_path": "/wav/"
    },
    "cmu_arctic": {
        "path": Path("~/speech/cmu_arctic").expanduser(),
        "accent_labels": {
            "cmu_us_clb_arctic": "American",
            "cmu_us_rms_arctic": "American",
            "cmu_us_bdl_arctic": "American",
            "cmu_us_slt_arctic": "American",
        },
        "speaker_split": {
            "cmu_us_clb_arctic": "aid_train",
            "cmu_us_rms_arctic": "aid_train",
            "cmu_us_bdl_arctic": "aid_train",
            "cmu_us_slt_arctic": "aid_test",
        },
        "transcript_path": "/etc/txt.done.data",
        "wav_path": "/wav/"
    },
}


def do_l2arctic(args):
    dataset_name = "l2arctic"
    dataset_path = DATASET_REGISTRY[dataset_name]["path"]
    all_data = []
    for speaker in DATASET_REGISTRY[dataset_name]["accent_labels"].keys():
        # print(Path.joinpath(dataset_path, speaker, "wav").exists()) True
        wave_path = Path.joinpath(dataset_path, speaker, "wav")
        for utterance in wave_path.glob("*.wav"):
            utterance_id = str(utterance).split(".")[0].split("/")[-1]
            speaker_id = speaker
            accent_label = DATASET_REGISTRY[dataset_name]["accent_labels"][speaker]
            dataset_name = dataset_name
            wav_path = str(utterance)
            transcript_path = Path.joinpath(dataset_path, speaker, "transcript", f"{utterance_id}.txt")
            with open(transcript_path, "r", encoding="utf-8") as f:
                transcript = f.read()
            f.close()
            # if transcript == "":
            #     print(transcript)
            split = DATASET_REGISTRY[dataset_name]["speaker_split"][speaker]
            all_data.append({
                "utt_id": utterance_id,
                "speaker_id": speaker_id,
                "accent_label": accent_label,
                "dataset_name": dataset_name,
                "wav_path": wav_path,
                "transcript": transcript,
                "split": split
            })

    return all_data


def do_cmu_arctic(args):
    dataset_name = "cmu_arctic"
    dataset_path = DATASET_REGISTRY[dataset_name]["path"]
    all_data = []
    for speaker in DATASET_REGISTRY[dataset_name]["accent_labels"].keys():
        wave_path = Path.joinpath(dataset_path, speaker, "wav")
        transcript_path = Path.joinpath(dataset_path, speaker, "etc", "txt.done.data")
        transcript_items = []
        with transcript_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                match = re.match(r'\(\s*(\S+)\s+"(.*)"\s*\)', line)
                if match:
                    utt_id = match.group(1)
                    sentence = match.group(2)
                    transcript_items.append((utt_id, sentence))
        transcript_lookup = dict(transcript_items)
        f.close()
        for utterance in wave_path.glob("*.wav"):
            utterance_id = str(utterance).split(".")[0].split("/")[-1]
            speaker_id = speaker.split("_")[2].upper()
            accent_label = DATASET_REGISTRY[dataset_name]["accent_labels"][speaker]
            dataset_name = dataset_name
            wav_path = str(utterance)
            transcript = transcript_lookup.get(utterance_id, "")
            split = DATASET_REGISTRY[dataset_name]["speaker_split"][speaker]
            # if transcript == "":
            #     print(speaker_id, utterance_id)
            all_data.append({
                "utt_id": utterance_id,
                "speaker_id": speaker_id,
                "accent_label": accent_label,
                "dataset_name": dataset_name,
                "wav_path": wav_path,
                "transcript": transcript,
                "split": split
            })

    return all_data

# schema: utt_id, speaker_id, accent_label, dataset_name, wav_path, transcript, split
def organize_source_data(args):
    # dataset_name = args.dataset_name
    # if dataset_name == "l2arctic":
    #     all_data = do_l2arctic(args)
    # elif dataset_name == "cmu_arctic":
    #     all_data = do_cmu_arctic(args)
    # else:
    #     raise ValueError(f"Unsupported dataset: {dataset_name}")
    l2arctic_data = do_l2arctic(args)
    cmu_arctic_data = do_cmu_arctic(args)
    all_data = l2arctic_data + cmu_arctic_data
    output_path = Path(f"data/metadata/combined_metadata_all.json")
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=4)
    f.close()

    l2arctic_output_path = Path(f"data/metadata/l2arctic_metadata_all.json")
    with l2arctic_output_path.open("w", encoding="utf-8") as f:
        json.dump(l2arctic_data, f, ensure_ascii=False, indent=4)
    f.close()
    cmu_arctic_output_path = Path(f"data/metadata/cmu_arctic_metadata_all.json")
    with cmu_arctic_output_path.open("w", encoding="utf-8") as f:
        json.dump(cmu_arctic_data, f, ensure_ascii=False, indent=4)
    f.close()


def sanity_check(args):
    AID_TEST_SPEAKERS = {
        "SKA", "BWC", "SVBI", "HKK", "NJS", "HQTV", "SLT"
    }

    SPEAKER_TO_ACCENT = {
        "ABA": "Arabic", "SKA": "Arabic", "YBAA": "Arabic", "ZHAA": "Arabic",
        "BWC": "Mandarin", "LXC": "Mandarin", "NCC": "Mandarin", "TXHC": "Mandarin",
        "ASI": "Hindi", "RRBI": "Hindi", "SVBI": "Hindi", "TNI": "Hindi",
        "HJK": "Korean", "HKK": "Korean", "YDCK": "Korean", "YKWK": "Korean",
        "EBVS": "Spanish", "ERMS": "Spanish", "MBMPS": "Spanish", "NJS": "Spanish",
        "HQTV": "Vietnamese", "PNV": "Vietnamese", "THV": "Vietnamese", "TLV": "Vietnamese",
        "CLB": "American", "RMS": "American", "BDL": "American", "SLT": "American",
    }

    LABELS = [
        "Arabic",
        "Mandarin",
        "Hindi",
        "Korean",
        "Spanish",
        "Vietnamese",
        "American",
    ]
    
    out_dir = Path("data/metadata")
    out_dir.mkdir(parents=True, exist_ok=True)

    label2id = get_json_content("data/metadata/label2id.json")
    metadata = get_json_content(f"data/metadata/combined_metadata_all.json")
    
    checked = []
    missing_wavs = []
    empty_transcripts = []
    unknown_speakers = []

    for item in metadata:
        speaker = item["speaker_id"]

        if speaker not in SPEAKER_TO_ACCENT:
            unknown_speakers.append(speaker)
            continue

        accent = SPEAKER_TO_ACCENT[speaker]
        split = "aid_test" if speaker in AID_TEST_SPEAKERS else "aid_train"

        item["accent_label"] = accent
        item["label_id"] = label2id[accent]
        item["split"] = split

        if not Path(item["wav_path"]).exists():
            missing_wavs.append(item["wav_path"])

        if not item.get("transcript", "").strip():
            empty_transcripts.append(item)

        checked.append(item)

    aid_train = [x for x in checked if x["split"] == "aid_train"]
    aid_test = [x for x in checked if x["split"] == "aid_test"]

    set_json_file(Path.joinpath(out_dir, "metadata_checked.json"), checked)

    set_json_file(Path.joinpath(out_dir, "aid_train.json"), aid_train)
    set_json_file(Path.joinpath(out_dir, "aid_test.json"), aid_test)
    print("Total utterances:", len(checked))
    print("Train utterances:", len(aid_train))
    print("Test utterances:", len(aid_test))
    print()

    print("Accent distribution:")
    print(Counter(x["accent_label"] for x in checked))
    print()

    print("Split distribution:")
    print(Counter(x["split"] for x in checked))
    print()

    print("Speaker distribution:")
    speaker_counter = Counter(x["speaker_id"] for x in checked)
    for speaker, count in sorted(speaker_counter.items()):
        print(speaker, count)

    print()
    print("Missing wav files:", len(missing_wavs))
    if missing_wavs[:10]:
        print("First missing wavs:")
        for p in missing_wavs[:10]:
            print(p)

    print()
    print("Empty transcripts:", len(empty_transcripts))

    print()
    print("Unknown speakers:", sorted(set(unknown_speakers)))


def voice_perturbation_check(args):
    from aid_dataset import VoicePerturbation
    import torchaudio
    import torch

    perturb = VoicePerturbation(sample_rate=args.target_sample_rate, apply_prob=args.perturbation_prob)
    data = get_json_content(args.test_data_path)[:10]
    for idx, item in enumerate(data):
        wave_path = item["wav_path"]
        waveform, cur_sample_rate = torchaudio.load(wave_path)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if cur_sample_rate != args.target_sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, 
                orig_freq=cur_sample_rate,
                new_freq=args.target_sample_rate
            )
        waveform = waveform.squeeze(0)
        perturbed_waveform = perturb(waveform)
        # print(f"Original: {waveform[:10]}, Perturbed: {perturbed_waveform[:10]}")
        torchaudio.save(f"data/tmp/perturbation_check_{idx}_original.wav", waveform.unsqueeze(0).cpu(), args.target_sample_rate)
        torchaudio.save(f"data/tmp/perturbation_check_{idx}_perturbed.wav", perturbed_waveform.unsqueeze(0).cpu(), args.target_sample_rate)


if __name__ == "__main__":
    args = parse()
    # organize_source_data(args)
    # sanity_check(args)  
    voice_perturbation_check(args)