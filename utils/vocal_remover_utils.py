import math
import os
import numpy as np
import torch
import librosa
import audioread
import soundfile

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# include "utlimatevocalremovergui" as python path
import sys

from tqdm import tqdm
sys.path.append('/home/ultimatevocalremovergui/')

from lib_v5.vr_network import nets, nets_new
from lib_v5 import spec_utils
from lib_v5.vr_network.model_param_init import ModelParameters
from gui_data.constants import *

SETTING_PATH = '/home/ultimatevocalremovergui/lib_v5/vr_network/modelparams/'
VOCAL_STEM = 'Vocals'
INST_STEM = 'Instrumental'
OTHER_STEM = 'Other'
BASS_STEM = 'Bass'
DRUM_STEM = 'Drums'
GUITAR_STEM = 'Guitar'
PIANO_STEM = 'Piano'
SYNTH_STEM = 'Synthesizer'
STRINGS_STEM = 'Strings'
WOODWINDS_STEM = 'Woodwinds'
BRASS_STEM = 'Brass'
WIND_INST_STEM = 'Wind Inst'
NON_ACCOM_STEMS = (
            VOCAL_STEM,
            OTHER_STEM,
            BASS_STEM,
            DRUM_STEM,
            GUITAR_STEM,
            PIANO_STEM,
            SYNTH_STEM,
            STRINGS_STEM,
            WOODWINDS_STEM,
            BRASS_STEM,
            WIND_INST_STEM)

default_param = {}
default_param['bins'] = 768
default_param['unstable_bins'] = 9 # training only
default_param['reduction_bins'] = 762 # training only
default_param['sr'] = 44100
default_param['pre_filter_start'] = 757
default_param['pre_filter_stop'] = 768
default_param['mid_side'] = False
default_param['mid_side_b2'] = False
default_param['reverse'] = False
default_param['band'] = {}


default_param['band'][1] = {
    'sr': 11025,
    'hl': 128,
    'n_fft': 960,
    'crop_start': 0,
    'crop_stop': 245,
    'lpf_start': 61, # inference only
    'lpf_stop': 139,
    'res_type': 'polyphase'
}

default_param['band'][2] = {
    'sr': 44100,
    'hl': 512,
    'n_fft': 1536,
    'crop_start': 24,
    'crop_stop': 547,
    'hpf_start': 81, # inference only
    'hpf_stop': 15,
    'res_type': 'sinc_best'
}

def rerun_mp3(audio_file, sample_rate=44100):

    with audioread.audio_open(audio_file) as f:
        track_length = int(f.duration)

    return librosa.load(audio_file, duration=track_length, mono=False, sr=sample_rate)[0]


class SeperateVR():        

    def seperate(
        self, 
        audio_file, 
        export_path, 
        window_size=1024,
        batch_size=4,
        vocal_or_instrument='vocal', 
        vr_model_setting='4band_44100', # 4band_v3
        verbose=False,
    ):
        ### added manually
        self.verbose = verbose
        self.model_capacity = 32, 128
        # self.param = default_param
        self.param_path = os.path.join(SETTING_PATH, "{}.json".format(vr_model_setting))
        self.param = ModelParameters(self.param_path).param
        self.audio_file = audio_file
        self.export_path = export_path
        self.high_end_process = 'None' # 'None' , 'mirroring'
        self.model_path = '/home/ultimatevocalremovergui/models/VR_Models/model_data/1_HP-UVR.pth'
        if vocal_or_instrument == 'both':
            self.primary_source = True
            self.secondary_source = True
        else:
            self.primary_source = True if 'vocal' in vocal_or_instrument.lower()  else False
            self.secondary_source = True if 'inst' in vocal_or_instrument.lower() else False
        self.primary_stem = VOCAL_STEM
        self.secondary_stem = INST_STEM
        self.audio_file_base = os.path.basename(self.audio_file).split('.')[0]
        self.model_samplerate = 44100
        self.is_vr_51_model = False
        self.aggressiveness = {
            'value': 10, 
            'split_bin': self.param['band'][1]['crop_stop'], 
            'aggr_correction': None
        }
        self.window_size = window_size
        self.is_tta = False
        self.batch_size = batch_size
        self.is_post_process = False
        self.is_normalization = False
        # self.primary_stem = NO_WIND_INST_STEM
        # self.secondary_stem = STEM_PAIR_MAPPER[self.primary_stem]
        self.post_process_threshold = 0.2
        self.input_high_end_h = None
        ###
        
        # determin the device to use
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        nn_arch_sizes = [
            31191, # default
            33966, 56817, 123821, 123812, 129605, 218409, 537238, 537227]
        vr_5_1_models = [56817, 218409]
        model_size = math.ceil(os.stat(self.model_path).st_size / 1024)
        nn_arch_size = min(nn_arch_sizes, key=lambda x:abs(x-model_size))

        if nn_arch_size in vr_5_1_models or self.is_vr_51_model:
            self.model_run = nets_new.CascadedNet(self.param['bins'] * 2, nn_arch_size, nout=self.model_capacity[0], nout_lstm=self.model_capacity[1])
        else:
            self.model_run = nets.determine_model_capacity(self.param['bins'] * 2, nn_arch_size)
                        
        self.model_run.load_state_dict(torch.load(self.model_path, map_location='cpu')) 
        self.model_run.to(device) 
        
        # seperate the audio file
        y_spec, v_spec = self.inference_vr(self.loading_mix(), device, self.aggressiveness)
        

        self.original_audio, _ = librosa.load(self.audio_file, self.model_samplerate, False, dtype=np.float32)
        # instrument
        self.secondary_source_audio = spec_utils.normalize(self.spec_to_wav(y_spec), self.is_normalization).T
        # vocal
        self.primary_source_audio = self.original_audio.T - self.secondary_source_audio

        if self.primary_source:
            primary_stem_path = os.path.join(self.export_path, f'{self.audio_file_base}_({self.primary_stem}).wav')
            
            if not self.model_samplerate == 44100:
                self.primary_source_audio = librosa.resample(self.primary_source_audio.T, orig_sr=self.model_samplerate, target_sr=44100).T
            
            self.write_audio(primary_stem_path, self.primary_source_audio, 44100)
            if self.verbose:
                logging.info(f"{self.primary_stem} is written to {primary_stem_path}")

        if self.secondary_source:
            secondary_stem_path = os.path.join(self.export_path, f'{self.audio_file_base}_({self.secondary_stem}).wav')
            
            if not self.model_samplerate == 44100:
                self.secondary_source_audio = librosa.resample(self.secondary_source_audio.T, orig_sr=self.model_samplerate, target_sr=44100).T
            
            self.write_audio(secondary_stem_path, self.secondary_source_audio, 44100)
            if self.verbose:
                logging.info(f"{self.secondary_stem} is written to {secondary_stem_path}")
            
        torch.cuda.empty_cache()
        
    def write_audio(self, stem_path, stem_source, samplerate, secondary_model_source=None, model_scale=None):
                
        if not os.path.exists(os.path.dirname(stem_path)):
            os.makedirs(os.path.dirname(stem_path))
            
        soundfile.write(stem_path, stem_source, samplerate)
            
    def loading_mix(self):

        X_wave, X_spec_s = {}, {}
        
        bands_n = len(self.param['band'])
        
        for d in range(bands_n, 0, -1):        
            bp = self.param['band'][d]
        
            wav_resolution = bp['res_type']
        
            if d == bands_n: # high-end band
                X_wave[d], _ = librosa.load(self.audio_file, bp['sr'], False, dtype=np.float32, res_type=wav_resolution)
                    
                if not np.any(X_wave[d]) and self.audio_file.endswith('.mp3'):
                    X_wave[d] = rerun_mp3(self.audio_file, bp['sr'])

                if X_wave[d].ndim == 1:
                    X_wave[d] = np.asarray([X_wave[d], X_wave[d]])
            else: # lower bands
                X_wave[d] = librosa.resample(X_wave[d+1], self.param['band'][d+1]['sr'], bp['sr'], res_type=wav_resolution)
                
            X_spec_s[d] = spec_utils.wave_to_spectrogram_mt(X_wave[d], bp['hl'], bp['n_fft'], self.param['mid_side'], 
                                                            self.param['mid_side_b2'], self.param['reverse'])
            
            if d == bands_n and self.high_end_process != 'none':
                self.input_high_end_h = (bp['n_fft']//2 - bp['crop_stop']) + (self.param['pre_filter_stop'] - self.param['pre_filter_start'])
                self.input_high_end = X_spec_s[d][:, bp['n_fft']//2-self.input_high_end_h:bp['n_fft']//2, :]

        X_spec = spec_utils.combine_spectrograms(X_spec_s, self.param)
        
        del X_wave, X_spec_s

        return X_spec

    def inference_vr(self, X_spec, device, aggressiveness):
        def _execute(X_mag_pad, roi_size):
            X_dataset = []
            patches = (X_mag_pad.shape[2] - 2 * self.model_run.offset) // roi_size
            total_iterations = patches//self.batch_size if not self.is_tta else (patches//self.batch_size)*2
            for i in range(patches):
                start = i * roi_size
                X_mag_window = X_mag_pad[:, :, start:start + self.window_size]
                X_dataset.append(X_mag_window)

            X_dataset = np.asarray(X_dataset)
            self.model_run.eval()
            with torch.no_grad():
                mask = []
                for i in tqdm(range(0, patches, self.batch_size), total=total_iterations, desc='Processing audio via model', leave=False):
                    X_batch = X_dataset[i: i + self.batch_size]
                    X_batch = torch.from_numpy(X_batch).to(device)
                    pred = self.model_run.predict_mask(X_batch)
                    if not pred.size()[3] > 0:
                        raise Exception("There is a problem with the model.")
                    pred = pred.detach().cpu().numpy()
                    pred = np.concatenate(pred, axis=2)
                    mask.append(pred)
                if len(mask) == 0:
                    raise Exception("There is a problem with the model.")
                
                mask = np.concatenate(mask, axis=2)
            return mask

        def postprocess(mask, X_mag, X_phase):
            
            is_non_accom_stem = False
            for stem in NON_ACCOM_STEMS:
                if stem == self.primary_stem:
                    is_non_accom_stem = True
            
            is_non_accom_stem = False ## added manually
                    
            mask = spec_utils.adjust_aggr(mask, is_non_accom_stem, aggressiveness)

            if self.is_post_process:
                mask = spec_utils.merge_artifacts(mask, thres=self.post_process_threshold)

            y_spec = mask * X_mag * np.exp(1.j * X_phase)
            v_spec = (1 - mask) * X_mag * np.exp(1.j * X_phase)
        
            return y_spec, v_spec
        X_mag, X_phase = spec_utils.preprocess(X_spec)
        n_frame = X_mag.shape[2]
        pad_l, pad_r, roi_size = spec_utils.make_padding(n_frame, self.window_size, self.model_run.offset)
        X_mag_pad = np.pad(X_mag, ((0, 0), (0, 0), (pad_l, pad_r)), mode='constant')
        X_mag_pad /= X_mag_pad.max()
        mask = _execute(X_mag_pad, roi_size)
        
        if self.is_tta:
            pad_l += roi_size // 2
            pad_r += roi_size // 2
            X_mag_pad = np.pad(X_mag, ((0, 0), (0, 0), (pad_l, pad_r)), mode='constant')
            X_mag_pad /= X_mag_pad.max()
            mask_tta = _execute(X_mag_pad, roi_size)
            mask_tta = mask_tta[:, :, roi_size // 2:]
            mask = (mask[:, :, :n_frame] + mask_tta[:, :, :n_frame]) * 0.5
        else:
            mask = mask[:, :, :n_frame]

        y_spec, v_spec = postprocess(mask, X_mag, X_phase)
        
        return y_spec, v_spec

    def spec_to_wav(self, spec):
        
        if self.high_end_process.startswith('mirroring'):        
            input_high_end_ = spec_utils.mirroring(self.high_end_process, spec, self.input_high_end, self.param)
            wav = spec_utils.cmb_spectrogram_to_wave(spec, self.param, self.input_high_end_h, input_high_end_)       
        else:
            wav = spec_utils.cmb_spectrogram_to_wave(spec, self.param)
            
        return wav