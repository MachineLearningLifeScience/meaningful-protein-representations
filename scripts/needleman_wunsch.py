import numpy as np
import numba

@numba.njit
def needleman_wunsch_forward(theta, A):
    N, M = theta.shape
    F = np.zeros((N + 1, M + 1))     # N x M
    D = np.zeros((N + 1, M + 1), dtype=np.int64)     # N x M
    for i in range(1, N+1):
        F[i, 0] = F[i-1,0] + A[1, i-1, 0]
        D[i, 0] = 0
    for j in range(1, M+1):
        F[0, j] = F[0,j-1] + A[0, 0, j-1]
        D[0, j] = 2
    dir = np.empty(3)
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            dir[0] = A[0, i - 1, j - 1]  + F[i - 1, j]      # x
            dir[1] = theta[i - 1, j - 1] + F[i - 1, j - 1]  # match
            dir[2] = A[1, i - j, 1 - 1]  + F[i, j - 1]      # y

            D[i,j] = np.argmax(dir)
            
            F[i, j] = dir[D[i,j]]
    return F,D

@numba.njit
def needleman_wunsch_traceback(theta, A, forward, direction):
    m, x, y = 1, 0, 2
    N, M = forward.shape
    i, j = N - 1, M - 1
    states = []
    while (i>0 or j>0):
        idx = [(i-1,j), (i-1, j-1), (i, j-1)]
        d = direction[i,j]
        states.append((i-1, j-1, d))
        i,j = idx[d]
        
    return states[::-1]

def needleman_wunsch_align(potentials, gaps):

    potentials_np = potentials.detach().cpu().numpy()
    gaps_np = gaps.detach().cpu().numpy()
    
    forward, direction = needleman_wunsch_forward(potentials_np, gaps_np)
    alignment = needleman_wunsch_traceback(potentials_np, gaps_np, forward, direction)
    return alignment
