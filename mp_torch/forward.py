import torch
def forward(matrix,inc_k_perp,jacket_a):
    def cor(S):
        N, _ = S.shape
        L = 2*N - 1 
        F = torch.fft.fft2(S, s=(L, L))
        C = torch.fft.ifft2(F.conj() * F)  
        C = torch.fft.fftshift(C, dim=(0, 1))
        idx = torch.arange(L)
        diag = C[idx, idx]
        d = torch.arange(-(N-1), N)
        w = torch.sinc(d / 2)
        return (w * diag).real.sum()
    
    terms=int(len(matrix)/2)
    S_0_EE=matrix[:terms,:terms]
    S_0_EK=matrix[:terms,terms:]
    S_0_KE=matrix[terms:,:terms]
    S_0_KK=matrix[terms:,terms:]
    
    forward_sigma=(cor(S_0_EE)+cor(S_0_EK)+cor(S_0_KE)+cor(S_0_KK)+2*(S_0_EE+S_0_KK).trace().real)
    return forward_sigma/(inc_k_perp*2*jacket_a)