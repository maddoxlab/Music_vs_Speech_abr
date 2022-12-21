# Deriving Music and Speech Response from Deconvolution
This repository is the code for the article "Music and Speech Elicit Similar Subcortical Responses" by Shan et al (2022) (https://doi.org/10.1101/2022.10.14.512309).

The EEG-BIDS format data for the paper are available on openneuro: https://openneuro.org/datasets/ds004356/versions/1.0.0

## Stimuli preprocessing
- `stim_preproc.py`: music and speech stimulus preprocessing, including converting, trimming the silence, flatten the music envelope.
- `spectral_matching_bandpower.py`: script for the spectral matching procesing.
## Regressor generateion
- `rectified_regressor_gen.py`: script for generating half-wave rectified stimulus waveform as the regressor.
- `IHC_ANM_regressor_gen.py`: script for generating IHC and ANM regressor. These files must be downloaded before the code can work: 
    - Code: 2018 Model: Cochlea+OAE+AN+ABR+EFR (Matlab/Python) from https://www.waves.intec.ugent.be/members/sarah-verhulst (this is for the `ic_cn2018` module) 
    - UR_EAR_2020b from https://www.urmc.rochester.edu/labs/carney.aspx (this is for importing the `cohclea` module)
## Response deriving
- `derive_click_ABR.py`: deriving click-evoked ABR using cross-correlation method.
- `derive_music_speech_ABR`: deriving music- and speech-eovked ABR using deconvolution method from three regressors.
- `derive_music_speech_cortical`: deriving music- and speech-eovked cortical resonses using deconvolution method from three regressors.
## Analysis
- `SNR.py`: calculating the SNR of music- and speech-evoked ABR from the three regressors.
- `SNR_recordingtime.py`: calculating the averaged ABR derived from different length of recording.
- `EEG_predict.py`: script for creating predicted EEG data from the kernels from the three regressors using convolution.
- `EEG_predict_correlation`: calculating the correlation coefficient between predicted EEG and real EEG data.
- `spectral_coherence`: calculating the spectral coherence between predicted EEG and real EEG data.
- `odd_even_morphology.py`: calculating the averaged ABR of evenly splitted two set and their waveform correlation (null). This also calculate the correaltion between music- and speech-evoked ABR from ANM regressor, and do the wilcoxon test this with the null distribution.