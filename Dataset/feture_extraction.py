from scipy.signal import stft
import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
load_dotenv()
np.random.seed(42)
seq_len=int(os.environ['WINDOW_SIZE'])
scaler = StandardScaler()
current_dir = os.path.dirname(os.path.abspath(__file__))
pth =os.path.join(current_dir,"gaitpdb","data")
UPDRS_path=os.path.join(current_dir,"gaitpdb","demographics.xls")


files= os.listdir(pth)
#reading demographic sheet for severity levels
df = pd.read_excel(UPDRS_path, usecols=['ID', 'UPDRS'])
UPDRS_level = dict(zip(df['ID'], df['UPDRS']))


m=3
tau=1
epsilon="auto"
eps_quantile=0.10
membership="rational"
scale="zscore"
mode="fuzzy"

def compute_instFreq(signal, fs, nperseg=None, noverlap=None):
    #Compute single IF using STFT weighted average.
    if nperseg is None:
        nperseg = len(signal)   # use full window
    if noverlap is None:
        noverlap = nperseg // 2

    f, t, Zxx = stft(signal, fs=fs, nperseg=nperseg, noverlap=noverlap)
    P = np.abs(Zxx) ** 2
    f_mean_t = np.sum(P * f[:, None], axis=0) / np.sum(P, axis=0)
    IF = np.mean(f_mean_t)   # average over time bins → 1 value per window
    return IF

#max possible SE of frequency-domain
def compute_spectral_entropy(signal):
    #Compute single SE using FFT."""
    fft_vals = np.fft.fft(signal)
    psd = np.abs(fft_vals[:len(signal)//2])**2
    psd /= np.sum(psd) + 1e-12
    SE = -np.sum(psd * np.log2(psd + 1e-12)) / np.log2(len(psd))
    return SE

def time_delay_embedding(x, m=3, tau=1, scale='zscore'):

    x = np.asarray(x, dtype=float).copy()

    if scale == 'zscore':
        std = x.std()
        if std > 0:
            x = (x - x.mean()) / std
        else:
            x = x - x.mean()
    elif scale == 'none':
        pass
    else:
        raise ValueError("scale must be 'zscore' or 'none'")

    T = x.shape[0]
    N = T - (m - 1) * tau
    if N <= 0:
        raise ValueError("Time series too short for given m and tau.")

    # Build Hankel-like embedded matrix
    idx = np.arange(m)[:, None] * tau + np.arange(N)[None, :]
    X = x[idx.T]  # shape (N, m)
    return X


def pairwise_distances(X):
    # Broadcasting method: O(N^2 m)
    diff = X[:, None, :] - X[None, :, :]       # (N, N, m)
    D = np.sqrt(np.sum(diff * diff, axis=2))   # (N, N)
    return D


def fuzzy_membership(D, epsilon, kind='rational',mode="fuzzy"):

    if epsilon is None or epsilon <= 0:
        raise ValueError("epsilon must be a positive float.")

    if mode == 'fuzzy':
        if kind == 'rational':
            M = 1.0 / (1.0 + (D / float(epsilon)))
        elif kind == 'gaussian':
            M = np.exp(-(D * D) / (2.0 * float(epsilon) ** 2))
        else:
            raise ValueError("kind must be 'rational' or 'gaussian'")
    elif mode == 'binary':
        # Binary recurrence: 1 if distance <= epsilon else 0
        M = (D <= float(epsilon)).astype(float)
    else:
        raise ValueError("mode must be 'fuzzy' or 'binary'")

    np.fill_diagonal(M, 1.0)  # always 1 on diagonal
    return M

def choose_epsilon(D, quantile=0.10):

    d = D[np.triu_indices_from(D, k=1)]  # upper triangle (exclude diagonal)
    d = d[d > 0]
    if d.size == 0:
        # All points identical; fall back to tiny epsilon
        return 1e-6
    q = np.clip(quantile, 1e-6, 1 - 1e-6)
    eps = np.quantile(d, q)
    if eps <= 0:
        eps = np.maximum(np.median(d), 1e-6)
    return float(eps)


def fuzzy_recurrence_plot(x, m=3, tau=1,epsilon='auto', eps_quantile=0.10,                  membership='rational', scale='zscore',mode="fuzzy"):

    # Embed
    X = time_delay_embedding(x, m=m, tau=tau, scale=scale)

    # Pairwise distances
    D = pairwise_distances(X)

    # Choose epsilon
    if epsilon == 'auto':
        eps_used = choose_epsilon(D, quantile=eps_quantile)
    else:
        eps_used = float(epsilon)

    # Convert to fuzzy similarities
    FRP = fuzzy_membership(D, eps_used, kind=membership,mode=mode)

    return FRP, X, eps_used

'''More bins → entropy becomes sensitive to noise and sample size.

Fewer bins → entropy overestimates complexity (because details are lost).'''

def fuzzy_recurrence_image_entropy(FRP, bins=64, normalize=True, exclude_diagonal=True):

    M = np.asarray(FRP, dtype=float)

    if exclude_diagonal: #If you include them, the histogram will be biased toward high similarity (1.0).
        mask = ~np.eye(M.shape[0], dtype=bool)
        vals = M[mask]
    else:
        vals = M.ravel()

    # Constrain to [0,1] for safety
    vals = np.clip(vals, 0.0, 1.0)

    # Histogram → probability distribution
    counts, bin_edges = np.histogram(vals, bins=bins, range=(0.0, 1.0))
    total = counts.sum() #total pixcel
    if total == 0:
        # Degenerate case (e.g., empty after masking)
        P = np.zeros_like(counts, dtype=float)
        H = 0.0
        return (H, P, bin_edges)

    P = counts.astype(float) / float(total)

    # Remove zero bins to avoid log(0) (they contribute 0 to the sum)
    nz = P > 0
    FRIE = -np.sum(P[nz] * np.log2(P[nz]))

    if normalize:
        FRIE = FRIE / np.log2(bins)

    return FRIE, P, bin_edges


def fuzzy_recurrence_entropy(FRP, exclude_diagonal=True):
    M = np.asarray(FRP, dtype=float)
    if exclude_diagonal:
        mask = ~np.eye(M.shape[0], dtype=bool)
        vals = M[mask]
    else:
        vals = M.ravel()

    # Clip both terms to avoid log(0)
    vals = np.clip(vals, 1e-12, 1 - 1e-12)
    one_minus_vals = np.clip(1 - vals, 1e-12, 1 - 1e-12)

    term1 = -np.sum(vals * np.log2(vals))
    term2 = -np.sum(one_minus_vals * np.log2(one_minus_vals))

    return (term1 + term2) / (M.shape[0] * M.shape[0])



def CVGRF_segmented(files,UPDRS_level, path, window_size=3000) -> pd.DataFrame:
    dataset = []
    label = []
    UPDRS =[]
    id=[]

    for i in files:
        file_path = os.path.join(path, i)
        data = pd.read_csv(file_path, sep='\t', header=None)

        # Determine the label based on filename
        if "Pt" in i:
            file_label = 1  # Parkinson's disease
            updrs=UPDRS_level[i.split('_')[0]]
            if updrs =="nan" or updrs=="NAN":
              updrs=4
            sl=1
            if updrs < 5:
                sl = 1
            elif updrs < 15:
                sl = 2
            elif updrs < 25:
                sl = 3
            elif updrs <35:
                sl = 4
            else:
                sl = 5

        elif "Co" in i:
            file_label = 0  # Control
            sl=1

        else:
            # Handle files that don't match "Pt" or "Co" if necessary
            continue

        cm_sum = data.iloc[:, -1] + data.iloc[:, -2]
        cm_sum = cm_sum.values

        # number of full windows
        num_segments = len(cm_sum) // window_size

        for seg in range(num_segments):
            start = seg * window_size
            end = start + window_size
            segment = cm_sum[start:end]
            
            segment = scaler.fit_transform(segment.reshape(-1, 1)).flatten() # type: ignore
            #segment = (segment - np.mean(segment)) / np.std(segment)

            dataset.append(segment)
            label.append(file_label)  # Assign the numerical label
            UPDRS.append(sl)
            id.append(i)

    return pd.DataFrame({
        "id":id,
        "signal": dataset,
        "label": label,
        "UPDRS":UPDRS,
    })



def create_feature_dataset(dataset,n_samples=1034, sequence_length=30, fs=100):

    print("extracting features ...")

    # Create CVGRF signals feature
    features_list = []
    labels_list = []
    severity_labels = []


    t = np.linspace(0, sequence_length, sequence_length * fs)
    for i in range(n_samples):
      cvgrf=dataset['signal'][i]

      label = dataset['label'][i]

      updrs = dataset['UPDRS'][i]



      IF=compute_instFreq(cvgrf,fs)
      SE=compute_spectral_entropy(cvgrf)
      FRP, X, eps_used = fuzzy_recurrence_plot(cvgrf, m=m, tau=tau, epsilon="auto", eps_quantile=eps_quantile,membership=membership, scale=scale,mode=mode)
      FRIE,P,edge_bins=fuzzy_recurrence_image_entropy(FRP)
      FRE=fuzzy_recurrence_entropy(FRP)

      features=np.array([IF,SE,FRIE,FRE])

      features_list.append(features)
      labels_list.append(label)
      severity_labels.append(updrs)

      if (i + 1) % 100 == 0:
          print(f"Processed {i + 1}/{n_samples} samples")

    return np.array(features_list), np.array(labels_list),np.array(severity_labels)


dataset = CVGRF_segmented(files, UPDRS_level, pth, seq_len)

X,y,grading=create_feature_dataset(dataset)

# Create DataFrame for features (name them properly)
df = pd.DataFrame(X, columns=['f1', 'f2', 'f3', 'f4'])
db = pd.concat([dataset, df], axis=1)

num_cols = ['f1', 'f2', 'f3','f4']

db[num_cols] = scaler.fit_transform(db[num_cols])

db.to_pickle('preprocessed_data.pkl')

print(f"DataFrame 'db' successfully saved")

if __name__=='__main__':
    print(db.head(10))