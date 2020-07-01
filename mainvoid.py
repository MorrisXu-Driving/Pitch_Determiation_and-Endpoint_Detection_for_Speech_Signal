import numpy as np
import matplotlib.pyplot as plt
import Pitch_Extraction_Function as func
import scipy.signal as signal
import soundfile as sf

if __name__ == '__main__':
    # Initialization of main parameters
    wave, fs = sf.read('tone4_m.wav')
    wave_data = wave - np.mean(wave)
    N = len(wave_data)
    IS = 0.8  # Observe the waveform of the input audio and set non-speech time at the start of the input in second
    wlen = int(0.03 * fs)
    inc = int(0.01 * fs)
    NIS = round((IS * fs - wlen) / inc + 1)  # The number of frames for the non-speech signal at the start
    x, fn = func.enframe(wave_data, wlen, inc)  # One frame for one column vector

    # Bandpass Filtering and Normalization for Formant Elimination and Noise Cancellation
    lf = 60  # Hz
    hf = 500  # Hz
    b, a = func.bandpass(fs, hf, lf)
    wave_data = signal.lfilter(b, a, wave_data)
    wave_data = wave_data - np.mean(wave_data)  # DC Cancellation
    wave_data = wave_data / np.abs(wave_data.max())  # Normalization
    y, fn = func.enframe(wave_data, wlen, inc)
    fmax = np.int(np.floor(fs / lf))
    fmin = np.int(np.floor(fs / hf))

    # 1. Endpoint Detection

    # The corresponding time referring to each frame
    scale = np.linspace(0, N - 1, N, dtype=np.int64)
    time = scale / fs  # Time scale for each sample point
    frameTime = func.frame2time(fn, wlen, inc, fs)  # Time scale for each frame

    # Initialization for parameters
    r1= 0.03  # Threshold Coefficient for judging speech segment
    r2 = 0.26  # Threshold Coefficient for judging mainbodys in a speech segment
    ThrC = [10, 15]  # Max difference in F0 between adjacent frames
    miniL = 10  # Minimum length for a speech segment
    mnlong = 3  # Minimum length for a vowel major body

    # 2. Start detection for the endpoints of speech segments and vowels
    voiceseg, vosl, vseg, vsl, T2, T1, Bth, SF, Ef = func.endpoint(y, fn, r1 ,r2, NIS, miniL, mnlong)

    # 3. Start the F0 estimation for main body of vowel
    period = func.f_mainbody(y, fn, vseg, vsl, fmax, fmin, ThrC[0])

    # 4. Calculate extended area besides the main body of a vowel
    Extseg, Dlong = func.extendframes(voiceseg, vseg, vsl, Bth)

    # 5. Start the F0 estimation for extend area
    T0 = np.zeros(fn)
    F0 = np.zeros(fn)

    for k in range(0, vsl):
        ix1 = vseg.begin[k]
        ix2 = vseg.end[k]
        in1 = Extseg.begin[k]
        in2 = Extseg.end[k]
        ixl1 = Dlong[k, 0]
        ixl2 = Dlong[k, 1]

        if ixl1 > 0:
            Bt, voiceseg = func.backextend(y, fn, voiceseg, Bth, ix1, ixl1, period, k, fmax, fmin, ThrC)
        else:
            Bt = []
        if ixl2 > 0:
            Ft, voiceseg = func.foreextend(y, fn, voiceseg, Bth, ix2, ixl2, vsl, period, k, fmax, fmin, ThrC)
        else:
            Ft = []
        '''Concatenate the F0(pitch) of backward extended part,Bt, and front extended part,Ft, 
        with the F0(pitch) of mainbody in a speech segments'''
        FT = np.array([Bt, period[ix1:ix2], Ft])

        temp = []
        for sublist in FT:
            for item in sublist:
                temp.append(item)
        temp = np.array(temp)

        diff = len(temp) - (in2 - in1)
        T0[in1:in2 + diff] = temp

    tindex = [tindex for (tindex, val) in enumerate(T0) if val > fmax]
    T0[tindex] = fmax
    tindex = [tindex for (tindex, val) in enumerate(T0) if val < fmin and val != 0]
    T0[tindex] = fmin
    tindex = [tindex for (tindex, val) in enumerate(T0) if val != 0]
    F0[tindex] = fs / T0[tindex]

    # Smoothing of final result
    F0 = func.medianfiltering(F0, Extseg, vsl)

    # Spectrogram of the input audio
    data_input, time_scale, freq_scale, fn = func.GetFrequencyFeature(wave_data, fs)

    plt.figure(1, facecolor='w')
    plt.subplot(411)
    plt.plot(time, wave_data)
    plt.ylabel('Amplitude')
    plt.xlabel('time/s')
    plt.title('Original Signal Waveform')
    plt.xlim((0, np.max(time_scale)))
    plt.ylim(-1, 1)
    for k in range(0, vosl):
        plt.axvline(frameTime[voiceseg.begin[k]], -1, 1, color='r')
        plt.axvline(frameTime[voiceseg.end[k]], -1, 1, color='r', linestyle='-.')

    plt.subplot(412, facecolor='w')
    plt.plot(frameTime, Ef)
    plt.xlim(0, np.max(time))
    plt.ylim(0, np.nanmax(Ef) / 2.5)  # /2.5 for Magnify the details
    plt.ylabel('Magnitude')
    plt.xlabel('time/s')
    plt.title('Energy/Spectral Entropy ratio')
    plt.axhline(T1, 0, np.max(frameTime), color='r', linestyle='--', label='T1')
    plt.plot(frameTime, T2, color='r', linewidth=1, label='T2')

    for k in range(0, vsl):
        plt.axvline(frameTime[vseg.begin[k]], 0, np.nanmax(Ef), color='k', linestyle='-')
        plt.axvline(frameTime[vseg.end[k]], 0, np.nanmax(Ef), color='k', linestyle='-')

    plt.subplot(413)
    plt.plot(frameTime, T0, 'k')
    plt.title('Location of Period Pitch')
    plt.xlim(0, np.max(time))
    plt.ylim(0, fmax)
    plt.xlabel('time/s')
    plt.ylabel('location/sample')

    plt.subplot(414)
    plt.plot(frameTime, F0, 'k')
    plt.title('Fundamental Frequency')
    plt.xlim(0, np.max(time))
    plt.ylim(0, 400)
    plt.xlabel('time/s')
    plt.ylabel('F0/Hz')
    plt.subplots_adjust(wspace=1.5, hspace=1.5)

    plt.figure(2)  # Spectrogram Analysis
    plt.plot(time_scale, F0, color='snow', linewidth=0.5)
    spectrum = plt.contourf(time_scale, freq_scale, data_input)
    plt.ylim(0, 1000)
    plt.xlim(0, np.max(time))
    plt.xlabel('Time/s')
    plt.ylabel('Frequency/Hz')

    plt.figure(3)
    plt.imshow(data_input)
    plt.title('Spectrogram')
    plt.xlabel('Time/s')
    plt.ylabel('Frequency/Hz')
    plt.show()
