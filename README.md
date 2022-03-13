# ResNet_different_activations
Trying to understand the performance of various activation functions

# Sigmoid and its improvements

Sigmoid = nn.Sigmoid()

Hard Sigmoid Function = nn.Hardsigmoid()

Sigmoid-Weighted Linear Units (SiLU) = nn.SiLU()

Hyperboilic Tangent = nn.Tanh()

Hard Hyperbolic Function = nn.Hardtanh(-2, 2)

#  Softmax Function

Softmax = nn.Softmax(dim=1), nn.Softmax2d(), nn.LogSoftmax()

# Softsign

Softsign = nn.Softsign()

# ReLU and its improvements

ReLU = nn.ReLU(inplace=TRUE)

LeakyReLu (LReLU) = nn.LeakyReLU(0.1)

Parametric ReLU (PReLU) = nn.PReLU()

Randomized ReLU (RReLU) = nn.RReLU(0.1, 0.3)

#  Softplus Function

Softplus = nn.Softplus()

# ELU and its improvements

ELU = nn.ELU()

Scaled Exponential Linear Units (SELU) = nn.SELU() 

# Swish Function

Hardswish = nn.Hardswish()

Mish = nn.Mish()

# Others

nn.GELU()

nn.CELU()
