import wave
import sys
import numpy as np
import scipy.signal as signal
from scipy.fftpack import fft
import matplotlib.pyplot as plt
from numpy import NaN, Inf, arange, isscalar, asarray, array


def read_wav_data(filename):
    wav = wave.open(filename, "rb")
    num_frame = wav.getnframes()
    num_channel = wav.getnchannels()
    framerate = wav.getframerate()
    num_sample_width = wav.getsampwidth()
    str_data = wav.readframes(num_frame)
    wav.close()
    wave_data = np.fromstring(str_data, dtype=np.short)
    wave_data.shape = -1, num_channel
    wave_data = wave_data.T

    return wave_data, framerate


# Divide each piece of audio signal into a matrix with shape(fn,wlen)
def enframe(wavsignal, wlen, inc):
    # xw = np.linspace(0, wlen - 1, wlen, dtype=np.int64)
    # wf = 0.54 - 0.46 * np.cos(2 * np.pi * xw / (wlen - 1))  # Add a hamming window
    fn = int((len(wavsignal) - wlen)) // inc + 1
    framed_signal = np.zeros(shape=(wlen, fn), dtype=np.float)
    wav_arr = np.array(wavsignal)

    for i in range(0, fn):
        p_start = i * inc
        p_end = p_start + wlen
        one_frame = wav_arr[p_start:p_end]
        framed_signal[:, i] = one_frame  # one column vector for one frame
    return framed_signal, int(fn)


# Conducting the Short-time Fourier Transform(STFT) on audio signal
def GetFrequencyFeature(wavsignal, fs):
    step = int(10 / 1000 * fs)
    window_length = int(30 / 1000 * fs)

    wav_arr = np.array(wavsignal)
    wav_length = wav_arr.shape[0]

    dataframe, fn = enframe(wav_arr, window_length, step)  # fn(total number of frames)

    data_input = np.zeros((window_length // 2, fn), dtype=np.float)

    for i in range(0, fn):
        data_line = np.abs(fft(dataframe[:, i], window_length)) / wav_length
        data_input[:, i] = data_line[0: window_length // 2]

    spectrogram = np.log(data_input + 1)
    time_scale = frame2time(fn, window_length, step, fs)
    x = np.linspace(0, window_length - 1, window_length, dtype=np.int64)
    freq_scale = (x * fs / window_length)
    freq_scale = freq_scale[0:window_length // 2]

    return spectrogram, time_scale, freq_scale, fn  # data_input: one column for one frame


# Time scale for each frame
def frame2time(fn, wlen, step, fs):
    x = np.linspace(0, fn - 1, fn, dtype=np.int64)
    frametime = (x * step + wlen / 2) / fs
    return frametime


'''It is a class that contain information(i.e.begin frame index, end frame index, duration in frame number) of every 
speech segments in a audio signal and it calculates the number of speech segments in an audio signal'''


class Segments():
    def __init__(self):
        self.begin = []
        self.end = []
        self.duration = []

    def find_segment(self, voiced_index):  # get segments information from a sequence of voice index
        if voiced_index[0] == 0:
            pass
        k = 0
        self.begin.insert(k, voiced_index[0])

        for i in range(0, len(voiced_index) - 1):
            if voiced_index[i + 1] - voiced_index[i] > 1:
                self.end.insert(k, voiced_index[i])
                self.begin.insert(k + 1, voiced_index[i + 1])  # i is the end of k , i+1 is the start of k+1
                k = k + 1

        self.end.insert(k, voiced_index[-1])
        for i in range(0, k + 1):
            self.duration.insert(i, self.end[i] - self.begin[i] + 1)

        return k + 1  # k: total number of segments


'''Each speech segment consists of two parts 1. Main body 2. Start and End(Extended Part), this function
calculate the endpoint of each speech segments in an audio signal and the main body of each speech segment'''


def endpoint(x, fn, r1, r2, NIS, miniL, mnlong):
    [row, col] = np.shape(x)
    if col != fn:
        x.np.transpose()
    wlen = row  # get the frame length

    Esum = np.zeros(fn)
    H = np.zeros(fn)
    SF = np.zeros(fn)

    for i in range(0, fn):
        Sp = np.abs(fft(x[:, i]))
        Sp = Sp[:Sp.size // 2 + 1]
        Esum[i] = np.dot(Sp, Sp)  # Energy for one frame
        prob = Sp / (np.sum(Sp))
        H[i] = -np.dot(prob, np.log(prob))  # Spectral Entropy of each frame

    Ef = np.sqrt(1 + np.abs(Esum / H))  # Energy/Spectral Entropy ratio
    Ef = Ef / np.nanmax(Ef)
    T1 = np.mean(H[:NIS]) * r1 # Setting threshold for speech segment (Adjustable)
    zindex = [idx for (idx, val) in enumerate(Ef) if val >= T1]  # Find the index of all speech frames (1.)
    zseg1 = Segments()
    if len(zindex) == 0:
        print("Please change T1!!!")
        exit(-1)
    else:
        zsl = zseg1.find_segment(zindex)  # zseg: Information of speech segment zsl: total number of speech segment

    voiceseg = Segments()  # Find Speech Segments
    count = 0
    for k in range(0, zsl):
        if zseg1.duration[k] >= miniL:  # delete the segments if its length<miniL
            in1 = zseg1.begin[k]
            in2 = zseg1.end[k]
            voiceseg.begin.insert(count, in1)
            voiceseg.end.insert(count, in2)
            voiceseg.duration.insert(count, zseg1.duration[k])
            count = count + 1
            SF[in1:in2] = 1  # Set the SF[i] to 1 if it is a speech frame

    vosl = len(voiceseg.duration)
    zseg2 = Segments()
    vseg = Segments()  # Speech Segments found, now find major body of vowel(s) in each segment (2.)
    T2 = np.zeros(fn)
    count = 0
    Bth = []
    for k in range(0, vosl):  # in a speech segment
        inx1 = voiceseg.begin[k]
        inx2 = voiceseg.end[k]
        Eff = Ef[inx1:inx2]  # T2 Setting for each segment
        Th2 = r2 * np.nanmax(Eff)  # Set the threshold for major body of vowel(s)
        if Th2 <= T1:
            Th2 = 1.5 * T1  # Adjustable
        T2[inx1:inx2] = Th2

        zindex = [i for i in range(0, len(Eff)) if Eff[i] >= Th2]  # index of all major bodies of vowels in the segment

        if len(zindex) != 0:
            zsl = zseg2.find_segment(zindex)

            for m in range(0, zsl):
                if zseg2.duration[m] >= mnlong:
                    vseg.begin.insert(count, zseg2.begin[m] + inx1)
                    vseg.end.insert(count, zseg2.end[m] + inx1)
                    vseg.duration.insert(count, zseg2.duration[m])
                    Bth.append(k)  # The index of corresponding segments for each vowel
                    count = count + 1
    vsl = len(vseg.duration)

    return voiceseg, vosl, vseg, vsl, T2, T1, Bth, SF, Ef
    # voiceseg: voiced segments（.start .end .duration）
    # vosl: number of voiced segments
    # vseg: major body of all vowels（.start .end .duration）
    # vsl: number of vowels
    # T2: dynamic threshold for detection of major body of vowel(s) in one voiced segment
    # T1：dynamic threshold for detection of voiced segments (T1 = 0.04 * np.mean(H[0:NIS]))
    # Bth: The index of corresponding segments for each vowel
    # SF：voiced frames are set to 1
    # Ef: Entropy spectral of the input signal


'''This function calculates the frame information of extend parts of each speech segments in an audio signal based on
four different conditions as listed below'''


def extendframes(voiceseg, vseg, vsl, Bth):
    extseg = Segments()
    Dlong = np.zeros((vsl, 2), dtype=np.int)
    for j in range(0, vsl):
        # 1. The intermediate vowel in a multi-vowel speech segment, extend forward
        if j != 0 and j != (vsl - 1) and Bth[j] == Bth[j - 1] and Bth[j] == Bth[j + 1]:
            extseg.begin.insert(j, vseg.begin[j])
            extseg.end.insert(j, vseg.begin[j + 1])
            Dlong[j, 0] = 0
            Dlong[j, 1] = extseg.end[j] - vseg.end[j]
            extseg.duration.insert(j, extseg.end[j] - extseg.begin[j] + 1)

        # 2. The last vowel in a multi-vowel speech segment
        elif j != 0 and Bth[j] == Bth[j - 1]:
            extseg.begin.insert(j, vseg.begin[j])
            extseg.end.insert(j, voiceseg.end[Bth[j]])
            Dlong[j, 0] = 0
            Dlong[j, 1] = extseg.end[j] - vseg.end[j]
            extseg.duration.insert(j, extseg.end[j] - extseg.begin[j] + 1)

        # 3. The first vowel in a multi-vowel speech segment
        elif j != (vsl - 1) and Bth[j] == Bth[j + 1]:
            extseg.begin.insert(j, voiceseg.begin[Bth[j]])
            extseg.end.insert(j, vseg.begin[j + 1])
            Dlong[j, 0] = vseg.begin[j] - extseg.begin[j]
            Dlong[j, 1] = extseg.end[j] - vseg.end[j]
            extseg.duration.insert(j, extseg.end[j] - extseg.begin[j] + 1)

        # 4. Only one vowel in a speech segment
        else:
            extseg.begin.insert(j, voiceseg.begin[Bth[j]])
            extseg.end.insert(j, voiceseg.end[Bth[j]])
            Dlong[j, 0] = vseg.begin[j] - extseg.begin[j]
            Dlong[j, 1] = extseg.end[j] - vseg.end[j]
            extseg.duration.insert(j, extseg.end[j] - extseg.begin[j] + 1)
    return extseg, Dlong


# A bandpass filter that denoise input signal
def bandpass(fs, hf, lf):
    fs2 = fs / 2
    wp = np.array([lf, hf]) / fs2
    ws = np.array([20, 2000]) / fs2
    rp = 1
    rs = 40
    n, wn = signal.ellipord(wp, ws, rp, rs)
    b, a = signal.ellip(n, rp, rs, wn, 'bandpass')
    # Frequency Response
    w, h = signal.freqz(b, a)
    power = 20 * np.log10(np.clip(np.abs(h), 1e-8, 1e100))
    plt.plot(w / np.pi * fs / 2, power)
    plt.title("Frequency Response")
    plt.ylim(-100, 20)
    plt.xlim(0, 4000)
    plt.figure(1)
    plt.show()
    return b, a


# Auto correlation for positive delay
def autocorr(x):
    xx = np.correlate(x, x, "full")
    normx = np.linalg.norm(x, 2)
    result = xx / np.power(normx, 2)
    return result[int(result.size / 2):]


def peakdet(v, delta, x=None):
    '''Find local maximum points of the input. A point is considered a maximum peak
    if it has the maximal value, and was preceded (to the left) by a value lower by DELTA'''
    maxtab = []
    mintab = []

    if x is None:
        x = arange(len(v))

    v = asarray(v)

    if len(v) != len(x):
        sys.exit('Input vectors v and x must have same length')

    if not isscalar(delta):
        sys.exit('Input argument delta must be a scalar')

    if delta <= 0:
        sys.exit('Input argument delta must be positive')

    mn, mx = Inf, -Inf
    mnpos, mxpos = NaN, NaN

    lookformax = True

    for i in arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]

        if this < mn:
            mn = this
            mnpos = x[i]

        if lookformax:
            if this < mx - delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn + delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True

    return array(maxtab)  # maxtab[:,0] is locs / maxlab[:,1] is pks


def findmaxesm3(one_frame, fmax, fmin):
    '''Extract 3 pitch candidates from each frame's ACF where fmax and fmin is
    the lower and upper pass frequency of the bandpass filter'''
    if np.any(np.isnan(one_frame)):  # Return if fail to extract one frame
        return
    maxind = peakdet(one_frame[fmin:fmax], 0.1)  # maxtab[:,0] is locs / maxlab[:,1] is pks
    locs = maxind[:, 0]
    locs = locs + fmin  # To get the real postions of the pitches

    pks = maxind[:, 1]
    index = np.argsort(-pks)  # Sort the index of peaks in ascending order by amplitude
    locs = locs[index]  # Resort the locs according to index
    return locs[:min(len(locs), 3)]  # Only the first three pitches are stored


def ztcont11(Ptk, bdb, Ptb, Ptbp, cl):
    kl = bdb - 1
    T0 = Ptb
    T1 = Ptbp
    pdb = np.zeros(kl)
    for k in range(kl - 1, -1, -1):
        distance = np.abs(Ptk[:, k] - T0)
        distance = distance.flatten()
        ml = np.where(distance == np.min(distance))
        pdb[k] = Ptk[ml, k].item(0, 0)
        TT = Ptk[ml, k]
        TT = TT[0, 0]
        if np.abs(T0 - TT) > cl:  # Searching the best estimation (T[k]-T[k-1]<=cl) for Pam[k]==0
            TT = 2 * T0 - T1
            pdb[k] = TT
        T1 = T0
        T0 = TT
    return pdb


def ztcont21(Ptk, bdb, bde, Ptb, Pte, c1):  # ztcont 11/21/31 is for shortest-distance search
    kl = bde - bdb - 1
    T0 = Ptb
    pdm = np.zeros(kl)
    jmp = 0
    emp = 0
    for k in range(0, kl):
        j = k + bdb
        distance = np.abs(Ptk[:, j] - T0)
        distance = distance.flatten()
        ml = np.where(distance == np.min(distance))
        TT = Ptk[ml, j]
        TT = TT[0, 0]
        if np.abs(T0 - TT) > c1:  # Searching the best estimation (T[k]-T[k-1]<=cl) for Pam[k]==0
            emp = 1
            jmp = k
            break

        pdm[k] = Ptk[ml, j]
        T0 = Ptk[ml, j]
    if emp == 1:
        zxl = kl - jmp + 1
        deltazx = (Pte - T0) / (zxl + 1)
        for k2 in range(0, zxl):
            pdm[k2] = T0 + k2 * deltazx

    return pdm


def ztcont31(Ptk, bde, Pte, Pten, c1):
    fn = Ptk.shape[1]
    kl = fn - bde
    T0 = Pte
    T1 = Pten
    pde = np.zeros(kl)

    for k in range(0, kl):
        j = k + bde
        distance = np.abs(Ptk[:, j] - T0)
        distance = distance.flatten()
        ml = np.where(distance == np.min(distance))
        pde[k] = (Ptk[ml, j]).item(0, 0)
        TT = Ptk[ml, j]
        TT = TT[0, 0]
        if np.abs(T0 - TT) > c1:  # Searching the best estimation (T[k]-T[k-1]<=cl) for Pam[k]==0
            TT = 2 * T0 - T1
            pde[k] = TT
        T1 = T0
        T0 = TT

    return pde


def f_mainbody(y, fn, vseg, vsl, fmax, fmin, Thrc):
    '''To calculate the pitch of mainbodies in each speech segment in a audio signal
    Thrc: Maximum difference between two adjacent frames during estimation'''
    [row, col] = np.shape(y)
    if col != fn:
        y.np.transpose()  # one column vector for one frame

    wlen = row  # get the frame length
    period = np.zeros(fn)
    c1 = Thrc  # Maximum difference between two adjacent frames during estimation
    p = np.poly1d([0.1 / np.power(fmax, 4), 0, 0.1 / np.power(fmax, 2), 0.4 / fmax, 2])
    calibration = np.polyval(p, np.arange(wlen))

    for i in range(0, vsl):
        ixb = vseg.begin[i]
        ixe = vseg.end[i]
        ixd = ixe - ixb + 1
        Ptk = np.zeros((3, ixd), dtype=np.float)  # Matrix storing 3 candidates for each frame in a speech segments

        for k in range(0, ixd):
            u = y[:, k + ixb]
            ru = autocorr(u)
            ru = np.multiply(calibration, ru)
            ru = ru.flatten()
            locs = findmaxesm3(ru, fmax, fmin)
            Ptk[0:len(locs), k] = locs

        Kal = Ptk[0, :]
        meanx = np.mean(Kal)
        thegma = np.std(Kal)
        mt1 = meanx + thegma  # Upper bound for first screening
        mt2 = meanx - thegma  # Lower bound for first screening

        if thegma > 5:
            Ptemp = np.zeros((3, ixd))  # Ptemp: Index of valid candidates
            for i in range(0, 3):
                for j in range(0, ixd):
                    if mt2 < Ptk[i, j] < mt1:
                        Ptemp[i, j] = 1

            Pam = np.zeros(ixd)
            for k in range(0, ixd):
                if Ptemp[0, k] == 1:
                    Pam[k] = np.max(
                        np.multiply(Ptk[:, k], Ptemp[:, k]))  # Max valid candidates in each frame stored in Pam

            meanx = np.mean(Pam[Pam != 0])
            thegma = np.std(Pam[Pam != 0])

            if thegma < 0.5:
                thegma = 0.5
            mt1 = meanx + thegma
            mt2 = meanx - thegma
            pindex = [pindex for (pindex, val) in enumerate(Pam) if mt2 < val < mt1]
            Pamtmp = np.zeros(ixd)
            Pamtmp[pindex] = Pam[pindex]  # valid candidates after second screening

            if len(pindex) != ixd and len(pindex) != 0:
                bpseg = Segments()
                bpl = bpseg.find_segment(pindex)
                bdb = bpseg.begin[0]
                if bdb != 0:
                    Ptb = Pamtmp[bdb]
                    Ptbp = Pamtmp[bdb + 1]
                    pdb = ztcont11(Ptk, bdb, Ptb, Ptbp, c1)
                    Pam[0: bdb - 1] = pdb

                if bpl >= 2:
                    for k in range(0, bpl - 1):
                        pdb = int(bpseg.end[k])
                        pde = int(bpseg.begin[k + 1])
                        Ptb = Pamtmp[pdb]
                        Pte = Pamtmp[pde]
                        pdm = ztcont21(Ptk, pdb, pde, Ptb, Pte, c1)
                        Pam[pdb:pde - 1] = pdm

                bde = bpseg.end[bpl - 1]
                Pte = Pamtmp[bde - 1]
                Pten = Pamtmp[bde - 2]
                if bde != ixd:
                    pde = ztcont31(Ptk, bde, Pte, Pten, c1)
                    Pam[bde:ixd] = pde
            period[ixb:ixe + 1] = Pam
        else:
            period[ixb:ixe + 1] = Kal

    return period


def extendxcorr(y, sign, TT1, XL, ixb, fmax, fmin, ThrC):
    wlen = y.shape[0]
    c1 = ThrC[0]
    c2 = ThrC[1]
    Ptk = np.zeros((3, XL))

    for k in range(XL):
        j = ixb + sign * k
        u = y[:, j]
        ru = autocorr(u)
        locs = findmaxesm3(ru, fmax, fmin)
        Ptk[0:len(locs), k] = locs

    # F0 estimation by shortest distance
    Pkint = np.zeros(XL)
    ts = TT1
    emp = 0
    Ptk = np.array(Ptk)
    for k in range(0, XL):
        tp = Ptk[:, k]
        tz = [val - TT1 for (idx, val) in enumerate(tp)]
        tl = [idx for (idx, val) in enumerate(tz) if val == np.min(tz)]
        tv = tz[tl[0]]

        if k == 0:
            if tv <= c1:
                Pkint[k] = tp[tl[0]]
                ts = tp[tl]
            else:
                Pkint[k] = 0
                emp = 1
        else:
            if Pkint[k - 1] == 0:
                if tv < c2:
                    Pkint[k] = tp[tl[0]]
                    ts = tp[tl]
                else:
                    Pkint[k] = 0
                    emp = 1
            else:
                if tv <= c1:
                    Pkint[k] = tp[tl[0]]
                    ts = tp[tl]
                else:
                    Pkint[k] = 0
                    emp = 1

    # Insert
    pzseg = Segments()
    Pm = Pkint
    vsegch = 0
    vsegchlong = 0
    if emp == 1:
        pindexz = np.where(Pkint == 0)
        pindexz = np.array(pindexz)
        pindexz = pindexz.flatten()
        pzl = pzseg.find_segment(pindexz)
        for k1 in range(0, pzl):
            zx1 = pzseg.begin[k1]
            zx2 = pzseg.end[k1]
            zxl = pzseg.duration[k1]
            if zx1 != 0 and zx2 != XL:
                deltazx = (Pm[zx2] - Pm[zx1 - 1]) / (zxl + 1)
                for k2 in range(0, zxl):
                    Pm[zx1 + k2 - 1] = Pm[zx1 - 1] + k2 * deltazx
            elif zx1 == 0 and zx2 != XL:
                deltazx = (Pm[zx2] - TT1) / (zxl + 1)
                for k2 in range(0, zxl):
                    Pm[zx1 + k2 - 1] = TT1 + k2 * deltazx
            else:
                vsegch = 1
                vsegchlong = zxl

    return Pm, vsegch, vsegchlong


def backextend(y, fn, voiceseg, Bth, ix1, ixl1, period, m, fmax, fmin, ThrC):
    if y.shape[1] != fn:
        y = y.transpose()

    wlen = y.shape[0]
    TT1 = period[ix1 + 1]
    XL = ixl1
    sign = -1
    ixb = ix1

    Pm, vsegch, vsegchlong = extendxcorr(y, sign, TT1, XL, ixb, fmax, fmin, ThrC)

    if vsegch == 1:
        j = Bth[m]
        if m != 0:
            j1 = Bth[m - 1]
            if j != j1:
                voiceseg.begin[j] = voiceseg.begin[j] + vsegchlong
                voiceseg.duration[j] = voiceseg.duration[j] - vsegchlong
            else:
                voiceseg.begin[j] = voiceseg.begin[j] + vsegchlong
                voiceseg.duration[j] = voiceseg.duration[j] - vsegchlong

    Pm = np.array(Pm)
    if len(Pm) != 1:
        Pump = Pm[::-1]
    else:
        Pump = Pm
    Ext_T = Pump

    return Ext_T, voiceseg


def foreextend(y, fn, voiceseg, Bth, ix2,
               ixl2, vsl, period, m, lmax, lmin, ThrC):
    if y.shape[1] != fn:
        y = y.transpose()

    wlen = y.shape[0]
    XL = ixl2
    sign = 1
    TT1 = round(period[ix2 + 1])
    ixb = ix2
    Ext_T, vsegch, vsegchlong = extendxcorr(y, sign, TT1, XL, ixb, lmax, lmin, ThrC)

    if vsegch == 1:
        j = Bth[m]
        if m != (vsl - 1):
            j1 = Bth[m + 1]
            if j != j1:
                voiceseg.end[j] = voiceseg.end[j] - vsegchlong
                voiceseg.duration[j] = voiceseg.duration[j] - vsegchlong
        else:
            voiceseg.end[j] = voiceseg.end[j] - vsegchlong
            voiceseg.duration[j] = voiceseg.duration[j] - vsegchlong

    return Ext_T, voiceseg

# Output Smoothing
def linsmoothm(x, n):
    le = len(x)
    x = x.reshape((le))
    w = signal.hanning(n)
    w = w / np.sum(w)
    y = np.zeros((le))
    if np.mod(n, 2) == 0:
        l = int(n / 2)
        x = [1 * x[0], x, np.ones(l) * x[le - 1]]
        temp = []
        for sublist in x:
            for item in sublist:
                temp.append(item)
        x = np.array(temp)

    else:
        l = int((n - 1) / 2)
        x = [np.ones(1) * x[0], x, np.ones(l + 1) * x[le - 1]]
        temp = []
        for sublist in x:
            for item in sublist:
                temp.append(item)
        x = np.array(temp)

    for k in range(0, le):
        y[k] = np.dot(w, x[k:k + n])
    y = y.flatten()

    return y


def medianfiltering(x, vseg, vsl):
    y = np.zeros(len(x))
    for i in range(0, vsl):
        ixb = vseg.begin[i]
        ixe = vseg.end[i]
        u0 = x[ixb:ixe]
        y0 = signal.medfilt(u0, 3)
        v0 = linsmoothm(y0, 3)
        y[ixb:ixe] = v0
    return y
