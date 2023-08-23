# singing_conversion
A personal project trying to satisfy myself. In this repository, you can enter your beloved singers and let the machine convert any other songs into the style of your beloved singers.

## steps to run the code
### 1. Code Access
I made the code accessible by providing a Docker-Compose. You can build the docker image by running the following command in the terminal:
```shell
docker-compose up --build -d
```
Note: to run this command, you need to install docker and docker-compose on your machine. You can find the installation guide [here](https://docs.docker.com/get-docker/). You also need to install nvidia-docker2 to run the code on GPU. You can find the installation guide [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker).
### 2. Connect to the Container
After the image is built, you can run the following command to enter the container:
```shell
docker exec -it singing_conversion bash
```
or you can connect via your vscode docker extension.
### 3. Run the Code
There are steps needed before you can run the code. You can find the steps in the following sections.


#### 3.1. Download the Pretrained Models
Please follow the instructions in the [README.md](https://github.com/AmirSh15/singing_conversion/tree/main/so-vits-svc#readme) of the so-vits-svc repo to download the pretrained model. More specifically, 
- [__ContentVec__ ](https://ibm.box.com/s/z1wgl1stco8ffooyatzdwsqn2psd9lrr) and placed it under the `so-vits-svc/pretrain` directory.
- [__NSF-HIFIGAN__ ](https://github.com/openvpi/vocoders/releases/download/nsf-hifigan-v1/nsf_hifigan_20221211.zip) and placed it under the `so-vits-svc/pretrain/nsf_hifigan` directory.
 - [__G_0__](https://huggingface.co/therealvul/so-vits-svc-4.0-init/resolve/main/G_0.pth) and [__D_0__](https://huggingface.co/therealvul/so-vits-svc-4.0-init/resolve/main/D_0.pth) models and placed them under the `so-vits-svc/logs/44k` directory.
 - [__model_0__](https://huggingface.co/datasets/ms903/DDSP-SVC-4.0/resolve/main/pre-trained-model/model_0.pt) a diffusion model from [this repo](https://github.com/yxlllc/DDSP-SVC) and placed it under the `so-vits-svc/logs/44k/diffusion` directory.
 

#### 3.2. Download the Dataset
The very first step is to download the dataset. You can run the following command to download the dataset automatically:
```shell
python main.py --keywords "adele,micheal jakson" 
```
Required arguments:
- `--keywords`: a string of keywords separated by comma. The keywords are the names of the singers you want to download their songs. The default value is `adele,micheal jakson`.
- `--num_pages`: the number of pages to download from youtube. The default value is `5`.
- `--output_path`: the path to save the downloaded songs. The default value is `./so-vits-svc/raw`.
- `--advanced`: if you want to use the advanced slicing method, you can set this argument to `True`. The default value is `False`.

Note: All that is required is that the data either be put or the script automatically take care of under the `dataset_raw` folder in the structure format provided below.
```shell
dataset_raw
├───speaker0
│   ├───xxx1-xxx1.wav
│   ├───...
│   └───Lxx-0xx8.wav
└───speaker1
    ├───xx2-0xxx2.wav
    ├───...
    └───xxx7-xxx007.wav
```
Caveat: You need to make sure the downloaded clips from youtube are actually the songs of the singer. There are also some limits on the clips. For more info have a look into the `utils/utils.py` file and `download_audio_results` function.

****

<mark> All the steps below are explained for information. However the main.py script is already taking care of them. So you can just run the main.py script and it will take care of all the steps below. </mark>

#### 3.3. Slice the Audio Files
To avoid video memory overflow during training or pre-processing, it is recommended to limit the length of audio clips. Cutting the audio to a length of "5s - 15s" is more recommended. Slightly longer times are acceptable, however, excessively long clips may cause problems such as `torch.cuda.OutOfMemoryError`.

In general, only the `Minimum Interval` needs to be adjusted. For spoken audio, the default value usually suffices, while for singing audio, it can be adjusted to around `100` or even `50`, depending on the specific requirements.

After slicing, it is recommended to remove any audio clips that are excessively long or too short.

In this repo, you can run the following command to slice the audio files:
```shell
python utils/audio_slicer.py --raw_audio_path ./so-vits-svc/raw --processed_audio_path ./so-vits-svc/dataset_raw --min_length 2 --max_length 20
```

required arguments:
- `--raw_audio_path`: the path to the raw audio files. The default value is `./so-vits-svc/raw`.
- `--processed_audio_path`: the path to save the sliced audio files. The default value is `./so-vits-svc/sliced`.
- `--min_length`: the minimum length of the sliced audio files in seconds. The audio files shorter that this value, will be discarded. The default value is `5`.
- `--max_length`: the maximum length of the sliced audio files in seconds. The files longer that this values, will be sliced. The default value is `15`.
Optional arguments:
- `--advanced`: if you want to use the advanced slicing method, you can set this argument to `True`. The default value is `False`.
- `--raw_vocal_audio_path`: the path to the raw vocal audio files. The default value is `./so-vits-svc/raw_vocal`. If you set the `advanced` argument to `True`, you need to provide this argument.

The basic slicer is developed based on the code available [audio-slicer-CLI](https://github.com/openvpi/audio-slicer), but I personally found it not very accurate to use. The additional advanced slicer is developed based on vocal remover model available at [Ultimate Vocal Remover GUI v5.5.1](https://github.com/Anjok07/ultimatevocalremovergui). You can either install the OS related software and manually run the GUI to remove the vocal from the audio files, or you can run the following command to remove the vocal from the audio files automatically:
```shell
python utils/vocal_remover.py --raw_vocal_audio_path ./so-vits-svc/raw --output_path ./so-vits-svc/raw_vocal 
```

Note: make sure to run the audio slicer with the advanced argument set to `True` after running the vocal remover.

The final data structure in the `so-vits-svc/raw_vocal` directory should be as follows:
```shell
raw_vocal
├───speaker0
│   ├───xxx1-xxx1_(Vocals).wav
│   ├───...
│   └───Lxx-0xx8_(Vocals).wav
└───speaker1
    ├───xx2-0xxx2_(Vocals).wav
    ├───...
    └───xxx7-xxx007_(Vocals).wav
```

#### 3.4. Resample the Audio Files to 44.1kHz
The audio files need to be resampled to 44.1kHz. You can run the following command to resample the audio files:
```shell
python so-vits-svc/resample.py
```
For more info check the [so-vits-svc README.md](https://github.com/svc-develop-team/so-vits-svc/tree/4.1-Stable#readme).

#### 3.5. Automatically split the dataset into training and validation sets, and generate configuration files.

The dataset needs to be split into training and validation sets. You can run the following command to split the dataset and generate the configuration files:
```shell
python so-vits-svc/preprocess_first_config.py --speech_encoder vec768l12
```
For more info check the [so-vits-svc README.md](https://github.com/svc-develop-team/so-vits-svc/tree/4.1-Stable#readme).

#### 3.6. Preprocess and generating hubert and F0 features
You can run the following command to preprocess the dataset and generate the hubert and F0 features:
```shell
python so-vits-svc/preprocess_hubert_f0.py --f0_predictor dio --num_processes 8
```
For more info check the [so-vits-svc README.md](https://github.com/svc-develop-team/so-vits-svc/tree/4.1-Stable#readme).

#### 3.7. Train the model
You can run the following command to train the model:
```shell
python so-vits-svc/train.py -c so-vits-svc/configs/config.json -m 44k
```

## Inference

Use [inference_main.py](https://github.com/svc-develop-team/so-vits-svc/blob/4.0/inference_main.py)

```shell
# Example
python so-vits-svc/inference_main.py -m "so-vits-svc/logs/44k/G_30400.pth" -c "so-vits-svc/configs/config.json" -n "0XatWsZzx.wav" -t 0 -s "adele"
```

Required parameters:
- `-m` | `--model_path`: path to the model.
- `-c` | `--config_path`: path to the configuration file.
- `-n` | `--clean_names`: a list of wav file names located in the `raw` folder.
- `-t` | `--trans`: pitch shift, supports positive and negative (semitone) values.
- `-s` | `--spk_list`: Select the speaker ID to use for conversion.
- `-cl` | `--clip`: Forced audio clipping, set to 0 to disable(default), setting it to a non-zero value (duration in seconds) to enable.

Optional parameters: see the next section
- `-lg` | `--linear_gradient`: The cross fade length of two audio slices in seconds. If there is a discontinuous voice after forced slicing, you can adjust this value. Otherwise, it is recommended to use the default value of 0.
- `-f0p` | `--f0_predictor`: Select a F0 predictor, options are `crepe`, `pm`, `dio`, `harvest`, `rmvpe`,`fcpe`, default value is `pm`(note: f0 mean pooling will be enable when using `crepe`)
- `-a` | `--auto_predict_f0`: automatic pitch prediction, do not enable this when converting singing voices as it can cause serious pitch issues.
- `-cm` | `--cluster_model_path`: Cluster model or feature retrieval index path, if left blank, it will be automatically set as the default path of these models. If there is no training cluster or feature retrieval, fill in at will.
- `-cr` | `--cluster_infer_ratio`: The proportion of clustering scheme or feature retrieval ranges from 0 to 1. If there is no training clustering model or feature retrieval, the default is 0.
- `-eh` | `--enhance`: Whether to use NSF_HIFIGAN enhancer, this option has certain effect on sound quality enhancement for some models with few training sets, but has negative effect on well-trained models, so it is disabled by default.
- `-shd` | `--shallow_diffusion`: Whether to use shallow diffusion, which can solve some electrical sound problems after use. This option is disabled by default. When this option is enabled, NSF_HIFIGAN enhancer will be disabled
- `-usm` | `--use_spk_mix`: whether to use dynamic voice fusion
- `-lea` | `--loudness_envelope_adjustment`：The adjustment of the input source's loudness envelope in relation to the fusion ratio of the output loudness envelope. The closer to 1, the more the output loudness envelope is used
- `-fr` | `--feature_retrieval`：Whether to use feature retrieval If clustering model is used, it will be disabled, and `cm` and `cr` parameters will become the index path and mixing ratio of feature retrieval
  
Shallow diffusion settings:
- `-dm` | `--diffusion_model_path`: Diffusion model path
- `-dc` | `--diffusion_config_path`: Diffusion config file path
- `-ks` | `--k_step`: The larger the number of k_steps, the closer it is to the result of the diffusion model. The default is 100
- `-od` | `--only_diffusion`: Whether to use Only diffusion mode, which does not load the sovits model to only use diffusion model inference
- `-se` | `--second_encoding`：which involves applying an additional encoding to the original audio before shallow diffusion. This option can yield varying results - sometimes positive and sometimes negative.