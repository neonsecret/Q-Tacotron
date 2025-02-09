from datetime import datetime
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from adabelief_pytorch import AdaBelief
from dllogger import StdOutBackend, JSONStreamBackend, Verbosity, Logger
from torch.utils.data import DataLoader

from synthesizer.models.tacotron import audio
from synthesizer.models.qtacotron.synthesizer_dataset import SynthesizerDataset, \
    collate_synthesizer
from synthesizer.models.qtacotron.model import QTacotron
from synthesizer.utils import ValueWindow
from synthesizer.utils.plot import plot_spectrogram
from synthesizer.utils.symbols import symbols
from synthesizer.utils.text import sequence_to_text
from vocoder.display import *
import warnings
# ah yes, the speed up
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)
torch.backends.cudnn.benchmark = False
warnings.filterwarnings("ignore", category=UserWarning)

dl_logger = Logger(backends=[JSONStreamBackend(Verbosity.DEFAULT, 'log.txt'),
                             StdOutBackend(Verbosity.VERBOSE)])


def np_now(x: torch.Tensor): return x.detach().cpu().numpy()


def time_string():
    return datetime.now().strftime("%Y-%m-%d %H:%M")


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def train(run_id: str, syn_dir: Path, models_dir: Path, save_every: int, backup_every: int, force_restart: bool,
          hparams, use_amp, multi_gpu, print_every, batch_size, gradinit_bsize,
          n_epoch, perf_limit, debug=False, **args):
    if debug:
        start_time = time.time()
        use_time = time.time()
    models_dir.mkdir(exist_ok=True)
    torch.cuda.empty_cache()
    model_dir = models_dir.joinpath(run_id)
    plot_dir = model_dir.joinpath("plots")
    wav_dir = model_dir.joinpath("wavs")
    mel_output_dir = model_dir.joinpath("mel-spectrograms")
    meta_folder = model_dir.joinpath("metas")
    model_dir.mkdir(exist_ok=True)
    plot_dir.mkdir(exist_ok=True)
    wav_dir.mkdir(exist_ok=True)
    mel_output_dir.mkdir(exist_ok=True)
    meta_folder.mkdir(exist_ok=True)

    weights_fpath = model_dir / f"synthesizer.pt"
    metadata_fpath = syn_dir.joinpath("train.txt")

    print("Checkpoint path: {}".format(weights_fpath))
    print("Loading training data from: {}".format(metadata_fpath))
    print("Using model: QTacotron")
    if debug:
        use_time = time.time()
        print("Init time: {}, elapsed: {}, point 1".format(use_time, time.time() - start_time))

    time_window = ValueWindow(100)
    loss_window = ValueWindow(100)

    if torch.cuda.is_available() or True:
        dev_index = 0  # useful for multigpu
        device = torch.device(dev_index)
        from torch.cuda.amp import autocast
        if torch.cuda.device_count() > 1 and multi_gpu:
            print("Using devices:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
        else:
            print("Using device:", torch.cuda.get_device_name(dev_index))
    else:
        device = torch.device("cpu")
        from torch.cpu.amp import autocast
        print("Using device:", device)

    print("\nInitialising QTacotron Model...\n")
    model = QTacotron().to(device)

    # Initialize the dataset
    metadata_fpath = syn_dir.joinpath("train.txt")
    mel_dir = syn_dir.joinpath("mels")
    embed_dir = syn_dir.joinpath("embeds")
    dataset = SynthesizerDataset(metadata_fpath, mel_dir, embed_dir, hparams)

    gradinit_bsize = int(batch_size / 2) if gradinit_bsize < 0 else int(gradinit_bsize / 2)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    optimizer = AdaBelief(model.parameters(), lr=5e-4, eps=1e-16, betas=(0.9, 0.999), weight_decouple=True,
                          rectify=True)
    if force_restart or not weights_fpath.exists():
        print("\nStarting the training of QTacotron from scratch\n")
        model.step = torch.tensor(0)
        model.save(weights_fpath, optimizer)

        # Embeddings metadata
        char_embedding_fpath = meta_folder.joinpath("CharacterEmbeddings.tsv")
        with open(char_embedding_fpath, "w", encoding="utf-8") as f:
            for symbol in symbols:
                if symbol == " ":
                    symbol = "\\s"  # For visual purposes, swap space with \s

                f.write("{}\n".format(symbol))

    else:
        print("\nLoading weights at %s" % weights_fpath)
        model.load(weights_fpath, optimizer)
        print("Tacotron weights loaded from step %d" % model.step)
    r, lr, max_step, batch_size = hparams.tts_schedule_dict[hparams.get_max(model.step)]
    if force_restart or not weights_fpath.exists():
        model.save(weights_fpath, optimizer)
    else:
        model.load(weights_fpath, optimizer)

    current_step = model.get_step()

    training_steps = max_step - current_step
    collate_fn = partial(collate_synthesizer, r=r, hparams=hparams)
    # Begin the training
    simple_table([(f"Steps with r={r}", str(training_steps // 1000) + "k Steps"),
                  ("Batch Size", batch_size),
                  ("Learning Rate", lr)])

    data_loader = DataLoader(dataset, batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn,
                             pin_memory=True, timeout=300)
    torch.save(data_loader, "data_loader")
    total_iters = len(dataset)
    steps_per_epoch = np.ceil(total_iters / batch_size).astype(np.int32)
    epochs = np.ceil(training_steps / steps_per_epoch).astype(np.int32)
    dt_len = len(data_loader)
    print("Printing every", print_every, "steps")
    while True:
        r, lr, max_step, batch_size = hparams.tts_schedule_dict[hparams.get_max(model.step)]
        model.r = r
        try:
            for epoch in range(1, epochs + 1):
                for i, (texts, mels, embeds, bert_embeds, idx) in enumerate(data_loader, 1):
                    if perf_limit and i >= 500:  # for testing the dataset, 500 will be enough
                        break

                    start_time = time.time()
                    # Generate stop tokens for training
                    stop = torch.ones(mels.shape[0], mels.shape[2], device=device)
                    for j, k in enumerate(idx):
                        stop[j, :int(len(mels[j])) - 1] = 0

                    # switch to gpu
                    texts = texts.to(device)
                    mels = mels.to(device)
                    embeds = embeds.to(device)
                    bert_embeds = bert_embeds.to(device)

                    use_amp = bool(use_amp)

                    with autocast(enabled=use_amp):
                        # forward pass
                        m1_hat, m2_hat, attention, stop_pred = model(texts, mels, embeds, bert_embeds)

                    m1_loss = F.mse_loss(m1_hat, mels) + F.l1_loss(m1_hat, mels)
                    m2_loss = F.mse_loss(m2_hat, mels)
                    stop_loss = F.binary_cross_entropy(stop_pred, stop)

                    loss = m1_loss + m2_loss + stop_loss

                    optimizer.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)

                    if hparams.tts_clip_grad_norm is not None:
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), hparams.tts_clip_grad_norm)
                        if np.isnan(grad_norm.cpu()):
                            print("grad_norm was NaN!")

                    scaler.step(optimizer)
                    scaler.update()

                    time_window.append(time.time() - start_time)
                    loss_window.append(loss.item())

                    step = model.get_step()
                    k = step // 1000

                    msg = {
                        "Loss": f"{loss_window.average:#.4}",
                        "steps/s": f"{1. / time_window.average:#.2}",
                        "Step": step,
                        "One step time": str(round(time.time() - start_time, 2)) + "s"
                    }
                    if i % print_every == 0:
                        dl_logger.log(step=(epoch, str(i) + "/" + str(dt_len)), data=msg)
                    # Backup or save model as appropriate
                    if backup_every != 0 and step % backup_every == 0:
                        backup_fpath = weights_fpath.parent / f"synthesizer_{k:06d}.pt"
                        model.save(backup_fpath, optimizer)

                    if save_every != 0 and step % save_every == 0:
                        model.save(weights_fpath, optimizer)
                        dl_logger.log("INFO", data={
                            "status": "saved.."
                        })
                        dl_logger.flush()

                    # Evaluate model to generate samples
                    epoch_eval = hparams.tts_eval_interval == -1 and i == steps_per_epoch  # If epoch is done
                    step_eval = hparams.tts_eval_interval > 0 and step % hparams.tts_eval_interval == 0  # Every N steps
                    if epoch_eval or step_eval:
                        for sample_idx in range(hparams.tts_eval_num_samples):
                            # At most, generate samples equal to number in the batch
                            if sample_idx + 1 <= len(texts):
                                # Remove padding from mels using frame length in metadata
                                mel_length = int(dataset.metadata[idx[sample_idx]][4])
                                mel_prediction = np_now(m2_hat[sample_idx]).T[:mel_length]
                                target_spectrogram = np_now(mels[sample_idx]).T[:mel_length]
                                attention_len = mel_length // model.r

                                eval_model(attention=np_now(attention[sample_idx][:, :attention_len]),
                                           mel_prediction=mel_prediction,
                                           target_spectrogram=target_spectrogram,
                                           input_seq=np_now(texts[sample_idx]),
                                           step=step,
                                           plot_dir=plot_dir,
                                           mel_output_dir=mel_output_dir,
                                           wav_dir=wav_dir,
                                           sample_num=sample_idx + 1,
                                           loss=loss,
                                           hparams=hparams)

                    # Break out of loop to update training schedule
                    if step >= max_step:
                        break
        except:
            pass


def test(model, device, data_loader, dataset):
    losses = []
    for i, (texts, mels, embeds, idx) in enumerate(data_loader, 1):
        if i == 5:
            break
        stop = torch.ones(mels.shape[0], mels.shape[2], device=device)
        mels = mels.to(device)
        for j, k in enumerate(idx):
            stop[j, :int(dataset.metadata[k][4]) - 1] = 0
        with torch.no_grad():
            m1_hat, m2_hat, attention, stop_pred = model(texts.to(device), mels, embeds.to(device))
        m1_loss = F.mse_loss(m1_hat, mels) + F.l1_loss(m1_hat, mels)
        m2_loss = F.mse_loss(m2_hat, mels)
        stop_loss = F.binary_cross_entropy(stop_pred, stop)  # ?

        losses.append(float(m1_loss + m2_loss + stop_loss))
    return round(1 - avg(losses), 2)


def avg(x):
    return sum(x) / len(x)


def eval_model(attention, mel_prediction, target_spectrogram, input_seq, step,
               plot_dir, mel_output_dir, wav_dir, sample_num, loss, hparams):
    # Save some results for evaluation
    attention_path = str(plot_dir.joinpath("attention_step_{}_sample_{}".format(step, sample_num)))
    save_attention(attention, attention_path)

    # save predicted mel spectrogram to disk (debug)
    mel_output_fpath = mel_output_dir.joinpath("mel-prediction-step-{}_sample_{}.npy".format(step, sample_num))
    np.save(str(mel_output_fpath), mel_prediction, allow_pickle=False)

    # save griffin lim inverted wav for debug (mel -> wav)
    wav = audio.inv_mel_spectrogram(mel_prediction.T, hparams)
    wav_fpath = wav_dir.joinpath("step-{}-wave-from-mel_sample_{}.wav".format(step, sample_num))
    audio.save_wav(wav, str(wav_fpath), sr=hparams.sample_rate)

    # save real and predicted mel-spectrogram plot to disk (control purposes)
    spec_fpath = plot_dir.joinpath("step-{}-mel-spectrogram_sample_{}.png".format(step, sample_num))
    title_str = "{}, {}, step={}, loss={:.5f}".format("Tacotron", time_string(), step, loss)
    plot_spectrogram(mel_prediction, str(spec_fpath), title=title_str,
                     target_spectrogram=target_spectrogram,
                     max_len=target_spectrogram.size // hparams.num_mels)
    print("\nInput at step {}: {}".format(step, sequence_to_text(input_seq)))
