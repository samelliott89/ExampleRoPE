# Rotaion
import torch

def main():
    freqs = torch.arange(0, 4, 2) # [0, 2] — every other index
    freqs = freqs.float() / 4 # [0/4, 2/4] = [0.0, 0.5]
    freqs = 10000.0 ** freqs # [10000^0, 10000^0.5] = [1, 100]
    freqs = 1.0 / freqs # [1/1, 1/100] = [1.0, 0.01]
    print("freqs.shape: ", freqs.shape)
    print("freqs: ", freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    print("freqs_cis.shape: ", freqs_cis.shape)
    print("freqs_cis: ", freqs_cis)
    t = torch.arange(3) # [0, 1, 2]
    print("t.shape: ", t.shape)
    print("t: ", t)
    freqs = torch.outer(t, freqs)
    print("freqs.shape: ", freqs.shape)
    print("freqs: ", freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    print("freqs_cis.shape: ", freqs_cis.shape)
    print("freqs_cis: ", freqs_cis)

    print("-"*100)
    # Example
    q = torch.tensor([0.5, 0.3, 0.8, 0.2])  # 4 dims
                    # ↑────↑      ↑────↑  -> 4 dimensions
                    # pair 1   pair 2
    q = q.reshape(-1, 2)           # [[0.5, 0.3], [0.8, 0.2]] -> 2 pairs of real numbers -> 4 dimensions
    q = torch.view_as_complex(q) # [0.5+0.3i, 0.8+0.2i] -> 2 complex numbers
    print(q.shape)
    print(q)
    q = q * freqs_cis[1] # [0.54+0.84i, 1+0.01i] -> 2 complex numbers
    q = torch.view_as_real(q).flatten(-2) # [[0.54, 0.84], [1.0, 0.01]] -> 2 pairs of real numbers
    print(q.shape)
    print(q)
    print(freqs_cis[1].shape)
    print(freqs_cis[1])

if __name__ == "__main__":
    main()