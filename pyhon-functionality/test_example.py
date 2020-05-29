from seluFunctions import activationJacobianSELU, activationSELU
#from reluFunctions import activationJacobianRELU, activationRELU
from scipy.linalg import eigh as largest_eigh
import array
import json
import numpy as np
from numpy import linalg as LA
import os
from numpy import array
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
usePyTorch = True
##################################################################### SMALL EXAMPLE'
# variables = []
# for i in range(2):
#     variables.append(1.0)


# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(2, input_shape=(len(variables),)),
#     tf.keras.layers.Dense(3),
#     tf.keras.layers.Dense(len(variables))
# ])
# # model.summary()

# xyArray = []

# for var in range(len(variables)):
#     v = tf.Variable([[variables[var]]])
#     xyArray.append(v)

# with tf.GradientTape(persistent=True) as t:
#     t.watch(xyArray)
#     z = tf.concat(xyArray, 1)

#     fArray = []
#     for f in range(len(variables)):
#         fVal = model(z)[0][f]
#         fArray.append(fVal)

# vArray = []
# for f in range(len(fArray)):
#     dArray = []

#     for val in range(len(xyArray)):
#         d = t.gradient(fArray[f], xyArray[val]).numpy()
#         d = d[0][0]
#         dArray.append(d)

#     vArray.append(dArray)

# del t
# # for p1 in range(len(vArray)):
#     # print("")
#     # for p2 in range(len(vArray[p1])):
#         # print(vArray[p1][p2], end=" ")
# # print(vArray)
# jacobian = np.asmatrix(vArray)
# matrix = np.dot(jacobian, jacobian.T)  # create a symmetric matrix
# print(matrix)
# eigenvalues = LA.eigvals(matrix)
# print('')

# # nsArray = []  # normSquared
# # for ev_count in range(len(eigenvalues)):
# #    ev = eigenvalues[ev_count]
# #    nsArray.append(ev.real*ev.real + ev.imag*ev.imag)

# evals_large, evecs_large = largest_eigh(matrix, eigvals=(len(variables)-1, len(variables)-1))
# print(eigenvalues)
# # np.savetxt('eigenvalues.txt', eigenvalues, delimiter=',')
# # print(nsArray)
# # np.savetxt('nsArray.txt', nsArray, delimiter=',')
# # print('max eigenvalue: ' + str(np.amax(eigenvalues)))
# # print('number of eigenvalues: ' + str(len(eigenvalues)))
# print('------------------------')
# print(evals_large)
# print(evecs_large)
# # my_matrix = np.loadtxt(open("eigenvalues.txt", "rb"), delimiter=",", skiprows=0)
# # print(float((my_matrix[0].lstrip())))
# # print(my_matrix[1])


#################################################################### END SMALL EXAMPLE
# path_to_model = "d:/kurser/project/backend/public/images/python/relu_1000epochs_mse_rmsp_1_16_lr001_biaspy"
# model1 = tf.keras.models.load_model(path_to_model) #using this way to load the model triggers more usability.
# with open('d:/kurser/project/backend/routes/jsons/image3.json') as f:  # choose the number on the picture
#     picture = json.load(f)
picture = [1.0, 1.0]

picture1 = [1.0, 1.0]
pictureForModel = array([[1.0, 1.0]])
model = tf.keras.Sequential([
    tf.keras.layers.Dense(2, activation='selu', input_shape=(len(picture1),)),
 #   tf.keras.layers.Dense(3),
    tf.keras.layers.Dense(len(picture1), activation='selu'),
    tf.keras.layers.Dense(len(picture1), activation='selu'),
    tf.keras.layers.Dense(len(picture1), activation='selu'),
    tf.keras.layers.Dense(len(picture1), activation='selu'),
    tf.keras.layers.Dense(len(picture1), activation='selu'),
    tf.keras.layers.Dense(len(picture1), activation='selu'),
    tf.keras.layers.Dense(len(picture1), activation='selu'),
    tf.keras.layers.Dense(len(picture1), activation='selu'),
    tf.keras.layers.Dense(len(picture1), activation='selu'),
    tf.keras.layers.Dense(len(picture1), activation='selu')
])
print('-------prediction2-------')
print(model.predict(pictureForModel))

def runModel(model, variables):
    xyArray = []

    for var in range(len(variables)):
        v = tf.Variable([[variables[var] * 1.0]])
        xyArray.append(v)

    with tf.GradientTape(persistent=True) as t:
        t.watch(xyArray)
        z = tf.concat(xyArray, 1)

        fArray = []
        for f in range(len(variables)):
            fVal = model(z)[0][f]
            fArray.append(fVal)

    vArray = []
    for f in range(len(fArray)):
        # for f in range(490,510):
        dArray = []

        for val in range(len(xyArray)):
            d = t.gradient(fArray[f], xyArray[val]).numpy()
            d = d[0][0]
            dArray.append(d)

        vArray.append(dArray)

    del t



    jacobian = np.asmatrix(vArray)
    print('-------TF jacobian-------')
    print(jacobian)
    matrix = np.dot(jacobian, jacobian.T)  # create a symmetric matrix
    
    evals_large, evecs_large = largest_eigh(
        matrix, eigvals=(len(variables)-1, len(variables)-1))
    # eigenvalues = LA.eigvals(matrix)

    # nsArray = []  # normSquared
    # for ev_count in range(len(eigenvalues)):
    #     ev = eigenvalues[ev_count]
    #     nsArray.append(ev.real*ev.real + ev.imag*ev.imag)

    # print('max eigenvalue: ' + str(np.amax(nsArray)))
    # print('number of eigenvalues: ' + str(len(eigenvalues)))


    # print('---------------------------------evals_large--------------------------------------------')
    # print(evals_large)
    # print('---------------------------------evacs_large--------------------------------------------')
    # print(evecs_large)


def updateJacobian(layer, output, jacobian):
    weightMatrix, biasVector = layer.get_weights()
    weightMatrix = np.transpose(weightMatrix)
    print('-------weight matrix-------')
    print(weightMatrix)

    jacobian = np.dot(weightMatrix, jacobian)

    output = np.add(np.dot(weightMatrix, output), biasVector)

    jacobian = np.dot(activationJacobianSELU(output, usePyTorch), jacobian)
    output = activationSELU(output, usePyTorch)

    return output, jacobian


def jacobian(pic, model):
    currentOutput = pic
    currentJacobian = []
    for i in range(len(pic)): #create identity matrix
        cArray = []
        for x in range(len(pic)):
            if (i == x):
                cArray.append(1)
            else:
                cArray.append(0)

        currentJacobian.append(cArray) 

    for layer in range(len(model.layers)):
        currentLayer = model.layers[layer]
        currentOutput, currentJacobian = updateJacobian(currentLayer, currentOutput, currentJacobian)
    print('------Our output--------')
    print(currentOutput)
    return currentJacobian


result = jacobian(picture, model)
print('----- our jacobian ----')
print(result)

runModel(model, picture1)


