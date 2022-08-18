class ConfigArgs:
    """
    Setting Configuration Arguments
    See the comments in this code

    """
    data_path = '../../data/kss'
    mel_dir, mag_dir = 'mels', 'mags'
    meta = 'transcript.v.1.3.txt'
    testset = 'ko_sents.txt'
    logdir = 'logs'  # log directory
    sampledir = 'samples'  # directory where samples are located
    mem_mode = False  # load all of the mel spectrograms into memory
    log_mode = True  # whether it logs
    log_term = 300  # log every n-th step
    eval_term = 1000  # log every n-th step
    synth_wav = False  # whether it synthesizes waveform
    save_term = 2000  # save every n-th step
    n_workers = 4  # number of subprocesses to use for data loading
    global_step = 0  # global step

    tp_start = 100000
    sr = 22050  # sampling rate
    n_fft = 1024  # n point Fourier transform
    n_mags = n_fft // 2 + 1  # magnitude spectrogram frequency
    n_mels = 80  # mel spectrogram dimension
    hop_length = 256  # hop length as a number of frames
    win_length = 1024  # window length as a number of frames
    r = 5  # reduction factor.

    batch_size = 32  # for training
    test_batch = 16  # for test
    max_step = 400000  # maximum training step
    lr = 0.001  # learning rate
    warm_up_steps = 4000.0  # warm up learning rate
    # lr_decay_step = 50000 # actually not decayed per this step
    # lr_step = [100000, 300000] # multiply 1/10
    Ce = 256  # dimension for character embedding
    Cx = 128  # dimension for context encoding
    Ca = 256  # attention dimension
    drop_rate = 0.05  # dropout rate
    n_tokens = 10  # number of tokens for style token layer
    n_heads = 8  # for multihead attention

    max_Tx = 188  # maximum length of text
    max_Ty = 250  # maximum length of audio

    _pad = "_"
    _eos = "<eos>"
    _characters = 'abcdefghijklmnopqrstuvwxyz!\'\"(),-.:;? '

    # Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
    _arpabet = [
        'aa', 'aa0', 'aa1', 'aa2', 'ae', 'ae0', 'ae1', 'ae2', 'ah', 'ah0', 'ah1', 'ah2', 'ao', 'ao0', 'ao1', 'ao2',
        'aw',
        'aw0', 'aw1', 'aw2', 'ay', 'ay0', 'ay1', 'ay2', 'b', 'ch', 'd', 'dh', 'eh', 'eh0', 'eh1', 'eh2', 'er', 'er0',
        'er1',
        'er2', 'ey', 'ey0', 'ey1', 'ey2', 'f', 'g', 'hh', 'ih', 'ih0', 'ih1', 'ih2', 'iy', 'iy0', 'iy1', 'iy2', 'jh',
        'k',
        'l',
        'm', 'n', 'ng', 'ow', 'ow0', 'ow1', 'ow2', 'oy', 'oy0', 'oy1', 'oy2', 'p', 'r', 's', 'sh', 't', 'th', 'uh',
        'uh0', 'uh1', 'uh2', 'uw', 'uw0', 'uw1', 'uw2', 'v', 'w', 'y', 'z', 'zh', 'vj', 'a0', 'a1', 'bj', 'c', 'dj',
        'e0',
        'e1', 'fj', 'gj', 'h', 'hj', 'i0', 'i1', 'j', 'kj', 'lj', 'mj', 'nj', 'o0', 'o1', 'pj', 'rj', 'sch', 'sj', 'tj',
        'u0',
        'u1', 'y0', 'y1', 'zj', "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH", "EH", "ER", "EY", "F", "G",
        "HH",
        "IH", "IY", "JH", "K", "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH", "T", "TH", "UH", "UW", "V", "W",
        "Y",
        "Z", "ZH"]

    # Export all symbols:
    symbols = [_pad, _eos] + list(_characters) + _arpabet
