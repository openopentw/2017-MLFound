from math import exp

def calc_confidence(N):
    return 4 * ((2*N)**10) * exp( -1/8 * (0.05**2) * N )

ch = [420000, 440000, 460000, 480000, 500000]

for c in ch:
    print(c, ':', calc_confidence(c))
