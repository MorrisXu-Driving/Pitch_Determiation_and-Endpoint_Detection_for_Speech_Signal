# Pitch_Determiation_for_Speech_Signal
It is a Pitch Determination Algorithm based on Short-time Autocorrelation and Shortest-distance Search
# 1. Installation
1. `git clone https://github.com/MorrisXu-Driving/Pitch_Determiation_for_Speech_Signal.git`
2. Create a new project in Python IDE and choose file `mainvoid.py` as the script path in `configuration`.
3. Make sure the test input wav file `tone4_w.wav` is under the same directory as the `mainvoid.py`.

# 2. Parameter Setting
 In this algorithm we have: 
- **Parameters for input preprocessing**
  - `wlen = int(0.03 * fs)` # 0.03 stands for wlen in time domain, here the wlen is 30ms.
  - `inc = int(0.01 * fs)`  # 0.03 stands for inc in time domain, here the inc is 10ms.
  ![Image](https://github.com/MorrisXu-Driving/Pitch_Determiation_for_Speech_Signal/blob/master/venv/readme_img/Frequency%20Response.JPG)
  - `lf = 60  # Hz`         # lf stands for the lower pass frequency of the bandpass denoising filter 
  - `hf = 500  # Hz`        # hf stands for the high pass frequency of the bandpass denoising filter
- **Parameters for pitch determination**
 ![Image](https://github.com/MorrisXu-Driving/Pitch_Determiation_for_Speech_Signal/blob/master/venv/readme_img/Endpoint_Detection.JPG)
  
  - `IS = 0.8`  # Observe the waveform of the input audio at above diagram and set non-speech time at the start of the input in second
  - `r1= 0.03`  # Threshold Coefficient for energy threshold T1 (shown in the above diagram) judging speech segment, namely `T1 = np.mean(H[:NIS]) * r1` where `H[:NIS]` is the energy of speech between 0-IS.
  - `r2 = 0.26`  # Threshold Coefficient for judging mainbodys in a speech segment, each speech segment has a different T2 (shown in the above diagram)
  - `ThrC = [10, 15]`  # Max difference in F0 between adjacent frames when conducting the shortest-distance search in order to avoid unnatural change in final result
  - `miniL = 10`  # Minimum length for a speech segment
  - `mnlong = 3`  # Minimum length for a major body in speech segments
  
 # 3. Result Demo
 ![Image]( ![Image](https://github.com/MorrisXu-Driving/Pitch_Determiation_for_Speech_Signal/blob/master/venv/readme_img/Endpoint_Detection.JPG))
 The above diagram consists of the spectrogram of the input audio and the pitch extracted from the input file.
  
 # 4. Conclusion
 The algorithm is not adaptive to differnt types of audio signals. 
 For those inputs with low SNR(i.e. the background energy between 0-IS is very high already needs to set a low r1)
 For those inputs with low energy at each speech segments, r2 should be lower in order to better recognize the extended parts besides each mainbodys.
 
 # 5.Networking
 Please do email me if you have any questions at philosoengineer@outlook.com. I am open to positions related to speech feature extraction and end-to-end speech recognition model. I really appreciate if you have relavent R%D positions that I could join in. 

