import copy
import tensorflow as tf
from reluFunctions import activationJacobianRELU, activationRELU
from seluFunctions import activationJacobianSELU, activationSELU
from os.path import isfile, join
from os import listdir
import time
import json
from numpy import linalg as LA
import array
import math
import codecs
import json
from scipy.linalg import eigh as largest_eigh
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# path to the pictures used to caluclate eigenvalues, has to be changed depending on the system.
path = ''  # path to pictures saved as json
pathToModels = ''  # path to the folder with models to compute for'

# Make sure that the files are named correctly, or they might be loaded in the wrong order here. (1->10->11...->2->20->21... , or 01->02->03->...)
fileNames = [f for f in listdir(path) if isfile(join(path, f))]

pictureArray = []

for file in range(len(fileNames)):
    filepath = path + fileNames[file]
    with open(filepath) as pic:
        picture = json.load(pic)
        pictureArray.append(picture)

# set this value with a function from other file // activationJacobianSELU, activationJacobianRELU
# set this value with a function from other file // activationSELU, activationRELU
currentActivationJacobian = activationJacobianRELU
currentActivation = activationRELU
# this only alters the 'alpha'-value used by the computations, True = the one used while training in Javascript TensorFlow and PyTorch.
usePyTorch = True
# (other options are used at other places of the TensorFlow framework, however not in the models)


def addVectorSavePicture(eigenvector, picture):
    perturbationVector = eigenvector.real * 0.1
    perturbationAmount = 0
    averagePerturbation = 0
    for i in range(len(perturbationVector)):
        perturbationAmount = perturbationAmount + pow(perturbationVector[i], 2)
        averagePerturbation = averagePerturbation + abs(perturbationVector[i])
    perturbationAmount = math.sqrt(perturbationAmount)
    averagePerturbation = averagePerturbation/3072
    print('Euclidean norm of perturbation vector:')
    print(perturbationAmount)
    print('Average absolute values of the perturbations on each of the 3072 entries:')
    print(averagePerturbation)
    eigenpicture = np.add(perturbationVector, picture)
    convertedList = eigenpicture.tolist()
    json.dump(convertedList, codecs.open('d:/kurser/project/backend/routes/jsons/python/image.json', 'w', encoding='utf-8'),
              separators=(',', ':'), sort_keys=True, indent=4)


def updateJacobian(layer, output, jacobian, updateFunctionJacobianAct, updateFunctionAct):
    biasVector = []
    weightMatrix = layer.get_weights()[0]
    if (len(layer.get_weights()) == 2):
        biasVector = layer.get_weights()[1]

    # flip the matrix (saved differently)
    weightMatrix = np.transpose(weightMatrix)

    # take the previous jacobian and multiply it with the weight matrix
    jacobian = np.dot(weightMatrix, jacobian)
    if (len(biasVector) != 0):
        # update the compressed picture with weights and bias
        output = np.add(np.dot(weightMatrix, output), biasVector)
    else:
        # update the compressed picture with just the weights
        output = np.dot(weightMatrix, output)

    # multiply the updated jacobian (weight*jacob) with the jacobian of the activation function (updatefuncjacobact)
    jacobian = np.dot(updateFunctionJacobianAct(output, usePyTorch), jacobian)
    # apply the activation function to update the compressed image
    output = updateFunctionAct(output, usePyTorch)
    # returns the updated picture and the updated jacobian at the current layer
    return output, jacobian


def jacobian(pic, model, updateFunctionJacobianAct, updateFunctionAct):
    currentOutput = pic
    currentJacobian = []
    # create identity matrix for the first layer as there is no jacobian to multiply
    for i in range(len(pic)):
        cArray = []
        for x in range(len(pic)):
            if (i == x):
                cArray.append(1)
            else:
                cArray.append(0)

        currentJacobian.append(cArray)

    for layer in range(len(model.layers)):
        currentLayer = model.layers[layer]
        currentOutput, currentJacobian = updateJacobian(
            currentLayer, currentOutput, currentJacobian, updateFunctionJacobianAct, updateFunctionAct)
    return currentJacobian


# This function is not really used except for investigative purposes, thus comments stay inside.
def calculateOneEigenvaluesForOnePicture(picture, model, activationJacobianFunction, activationFunction):
    jacobianMatrix = jacobian(
        picture, model, activationJacobianFunction, activationFunction)
    # matrix = np.dot(jacobianMatrix, jacobianMatrix.T)
    eigenvalues, eigenvectors = LA.eig(jacobianMatrix)
    # getHigestEigenvalueFromAll(jacobianMatrix)
    # eigenvalues = LA.eigvals(jacobianMatrix)

    # print('--------eigenvectors---------')
    # print(eigenvectors[0])
    print('--------eigenvalues---------')
    print(eigenvalues[0])
    addVectorSavePicture(eigenvectors[0], picture)

    # evals_large, evecs_large = largest_eigh(
    #     matrix, eigvals=(len(picture)-1, len(picture)-1))
    # print(evals_large)
    # print(evecs_large)


def calculateEigenValues(picture, model, activationJacobianFunction, activationFunction):
    jacobianMatrix = (
        jacobian(picture, model, activationJacobianFunction, activationFunction))
    highestEigenvalue = getHigestEigenvalueFromAll(jacobianMatrix)
    # getHigestEigenvalueFromSymmetric(jacobianMatrix, len(picture))
    return highestEigenvalue


def getHigestEigenvalueFromSymmetric(jacobianMatrix, inputLength):
    matrix = np.dot(jacobianMatrix, jacobianMatrix.T)
    evals_large, evecs_large = largest_eigh(
        matrix, eigvals=(inputLength-1, inputLength-1))
    # print('-------------------------------length of variables---------------------------------------------')
    # print(len(picture))
    # print('---------------------------------evals_large--------------------------------------------')
    print('max singular_value: ' + str(evals_large))
    # print('---------------------------------evacs_large--------------------------------------------')
    # print(evecs_large)


def getHigestEigenvalueFromAll(jacobianMatrix):
    eigenvalues = LA.eigvals(jacobianMatrix)
    nsArray = []  # normSquared
    for ev_count in range(len(eigenvalues)):
        ev = eigenvalues[ev_count]
        nsArray.append(ev.real*ev.real + ev.imag*ev.imag)
    highestEigenvalue = np.amax(nsArray)
    # print('max eigenvalue: ' + str(highestEigenvalue))
    # print('length of array: ' + str(len(nsArray)))
    return highestEigenvalue


def saveAndPrintHighEigenValues(model, modelName, folderName):
    eigenValuesAbove1 = []
    highestEigenValues = []
    for picture in range(len(pictureArray)):
        highEig = calculateEigenValues(
            pictureArray[picture], model, currentActivationJacobian, currentActivation)
        highestEigenValues.append(highEig)
        if (highEig >= 1):
            eigenValuesAbove1.append([highEig, picture])
    print('--------HE_Above1---------')
    print(eigenValuesAbove1)
    print('-------HE_ALL--------')
    print(highestEigenValues)
    json.dump(highestEigenValues, codecs.open('d:/kurser/project/backend/public/images/' + folderName + '/highestEigenvalues/' + modelName + '.json', 'w', encoding='utf-8'),
              separators=(',', ':'), sort_keys=True, indent=4)


def calculateForFourModelsInFolder(folderName, pathToModels):
    modelName1 = folderName + '1'
    modelName2 = folderName + '2'
    modelName3 = folderName + '3'
    modelName4 = folderName + '4'

    path_to_model1 = pathToModels + \
        folderName + '/pythonModels/' + modelName1
    path_to_model2 = pathToModels + \
        folderName + '/pythonModels/' + modelName2
    path_to_model3 = pathToModels + \
        folderName + '/pythonModels/' + modelName3
    path_to_model4 = pathToModels + \
        folderName + '/pythonModels/' + modelName4

    model1 = tf.keras.models.load_model(path_to_model1)
    model2 = tf.keras.models.load_model(path_to_model2)
    model3 = tf.keras.models.load_model(path_to_model3)
    model4 = tf.keras.models.load_model(path_to_model4)

    saveAndPrintHighEigenValues(model1, modelName1, folderName)
    saveAndPrintHighEigenValues(model2, modelName2, folderName)
    saveAndPrintHighEigenValues(model3, modelName3, folderName)
    saveAndPrintHighEigenValues(model4, modelName4, folderName)
