import math

#pyTorchAlpha = 1.6732632423543772848170429916717
tensorFlowAlpha = 1.7580993408473768599402175208123
#scale = 1.0507009873554804934193349852946
scale = 1.05070098
pyTorchAlpha = 1.67326324


def activationJacobianSELU(vector, usePyTorch):
    alpha = 0
    if (usePyTorch):
        alpha = pyTorchAlpha
    else:
        alpha = tensorFlowAlpha

    gradients = []
    for value in range(len(vector)):
        gradient = scale * alpha * math.exp(vector[value])
        if (vector[value] == 0):
            print(
                '** Holy moly! Tell Kathlen there is a 0 value!! ** @Index: ' + str(value))
            gradient = 0.5
        if (vector[value] > 0):
            gradient = scale

        gradients.append(gradient)

    gradientMatrix = []
    for vVal in range(len(vector)):
        newVector = []
        for space in range(len(vector)):
            if (space == vVal):
                newVector.append(gradients[vVal])
            else:
                newVector.append(0)

        gradientMatrix.append(newVector)

    return gradientMatrix


def activationSELU(output, usePyTorch):
    alpha = 0
    if (usePyTorch):
        alpha = pyTorchAlpha
    else:
        alpha = tensorFlowAlpha

    newOutput = []
    for x in range(len(output)):
        if (output[x] >= 0):
            newOutput.append(scale * output[x])
        else:
            newOutput.append(scale * alpha * (math.exp(output[x]) - 1))

    return newOutput
