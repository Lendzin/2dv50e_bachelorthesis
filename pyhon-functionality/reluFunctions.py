def activationJacobianRELU(vector, usePyTorch):
    # usePyTorch does nothing here (to not get complaints using this function with an extra variable for generic structure)
    gradients = []
    for value in range(len(vector)):
        gradient = 0
        if (vector[value] > 0):
            gradient = 1
        if (vector[value] == 0):
            print(
                '** There is a 0 value!! ** @Index: ' + str(value))
            gradient = 0.5

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


def activationRELU(output, usePyTorch):
    # usePyTorch does nothing here (to not get complaints using this function with an extra variable for generic structure)
    newOutput = []
    for x in range(len(output)):
        if (output[x] <= 0):
            newOutput.append(0)
        else:
            newOutput.append(output[x])

    return newOutput
