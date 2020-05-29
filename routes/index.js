const express = require('express')
const router = express.Router()
const fetch = require('node-fetch')
var fs = require('fs')

const comparison = require('../functions/comparisonFunctions')
const helpers = require('../functions/helperfunctions.js')
const {
  // functions used when generating 'comparison LaTeX table'.
  getAverageITFs,
  runComparison,
  makeBatchFileForComparison,
  getResultsFromBatch,
  createTableFromThreeBatches,
} = comparison
const {
  // functions used to solve non-regular actions
  generateRandomNoisePictures,
  shuffle,
  convertPicturesToGray_3072To1024,
  getRandomPicturesColor,
  getArrayOfOnePeturbedPicture,
  peturbArrayWithPictures,
  compareDistanceArrays,
  getDistanceToTrainingPicture,
  logDistanceOfPictures,
  // functions used regularly:
  getClosestTrainingPicture,
  getCifar10ArrayAtIndex,
  savePNGImageData,
  getXTimesCifar10ArraysFromIndex,
  getDistanceBetweenTwoPictures,
  getSaved100RandomPictures,
} = helpers
const trainAutoEncoder = require('../functions/train.js').trainAutoEncoder
const predictWithTrainedEncoder = require('../functions/predict.js')
  .predictWithTrainedEncoder

router.get('/', async function (req, res, next) {
  const firstPictureFromEachClass = await getCifar10ArrayAtIndex(1)
  const first10PicturesFromEachClass = await getXTimesCifar10ArraysFromIndex(
    10,
    1
  )
  const secondPictureFromEachClass = await getCifar10ArrayAtIndex(3)

  let randomPictures = await getSaved100RandomPictures()

  const trainingSetup = {
    activation: 'relu', //  Activations that can be set: 'elu','hardSigmoid','linear','relu',
    //  'relu6', 'selu', 'sigmoid','softmax','softplus','softsign','tanh'
    bias: false, // false or true
    hiddenLayers: 10, //  # of hidden layers = x + 1, because of using a Dense layer as input.
    unitsPerLayer: 32, // units per layer, also known as width
    lossFunction: 'meanSquaredError', // loss function used
    stopAtLoss: 0.000001, // experiment will stop, reaching this loss.
    stopAtEpochs: 50000, // experiment will stop, reaching this many epochs
    optimizer: 'rmsp', // adam, rmsp, sgd
    modelName: 'tp100_relu_color_d11_w32_rmsp_random_nobias', // name of the model to be saved as training finishes.
    oldModelToLoad: '', // can be other than current model, to be loaded before training.
    folderToLoad: '', // folder to load old model from
    trainingImages: randomPictures, // which pictures to load as training images for the model
    loadOldModel: false, // if an older model should be loaded
    trainingEpochs: 1000, // amount of epochs before reloading the model
    inputShape: 3072, // 1024 for gray pictures, 3072 for colored pictures
    saveTrainingImagesAsJson: false,
    learningRate: 0.001, // the learning rate of the optimizer
  }
  let trainingIterations = 3

  let training = false // if set to true, training will commence with given parameters

  let runRepeatPrediction = false // if set to true, normal prediction will run, "amountOfRepeats"
  let amountOfRepeats = 3000 //  need to be at least 1

  let repeatPredictForDistance = false // this starts the iterative fixed points finding predictions if true. (should be combined with: 'reRunUndefinedPredictions')
  let trainingIndex = 0
  let forcePredictTimes = 0
  let forcePredictStop = 1000
  let repeatThreshold = 0.00005
  let startPredictDistancePoint = 0
  let endPredictDistancePoint = 100

  let peterbRate = 1
  const predictionSetup = {
    testImages: first10PicturesFromEachClass, // needs to be changed based on what pictures you want to run the prediction with.
    modelNameToPredict: 'tp100_relu_color_d11_w128_rmsp1', // should most likely not be the same as train.
    picturesTrainedWith: first10PicturesFromEachClass, // needs to be changed based on the pictures used for the model trained.
    predictFolder: '', // if the files are not recently trained and thus moved, a folder is necessary
  }

  const {
    picturesTrainedWith,
    predictFolder,
    modelNameToPredict,
  } = predictionSetup

  let modelsToRepeatPredict = [
    predictFolder + '1',
    predictFolder + '2',
    predictFolder + '3',
    predictFolder + '4',
  ]

  if (training) {
    res.render('index')
    let count = 1
    const modelName = trainingSetup.modelName
    let trainingFinished = true
    while (count <= trainingIterations) {
      if (trainingFinished) {
        trainingFinished = false
        try {
          trainingSetup.modelName = modelName + count
          trainingFinished = await trainAutoEncoder(trainingSetup)
          if (trainingFinished) {
            count++
            console.log(
              '----------------------- TRAINING ENDED --------------------'
            )
          }
        } catch (error) {
          console.log('Crashed with.. ' + error)
        }
      }
    }
  } else {
    console.log(
      '----------------------- PREDICTION STARTED--------------------'
    )
    let filesIn = []
    let filesOut = []
    for (let i = 0; i < testImages.length; i++) {
      if (testImages.input) {
        savePNGImageData(testImages[i].input, `in${i}`)
        filesIn.push(`in${i}`)
      } else {
        savePNGImageData(testImages[i], `in${i}`)
        filesIn.push(`in${i}`)
      }
    }

    let predictionPics = []
    let trainingIndicies = []

    if (repeatPredictForDistance && !runRepeatPrediction) {
      res.render('index', {
        title: 'Finished',
      })

      for (let model = 0; model < modelsToRepeatPredict.length; model++) {
        let modelName = modelsToRepeatPredict[model]
        let imagesToTest = JSON.parse(JSON.stringify(testImages))
        let iterativeFixedPointStatusByIndex = []
        for (
          let i = startPredictDistancePoint;
          i < endPredictDistancePoint;
          i++
        ) {
          predictionSetup.testImages = [imagesToTest[i]]
          predictionSetup.modelNameToPredict = modelName
          let index = i
          if (trainingIndicies.length === imagesToTest.length) {
            index = trainingIndicies[i]
          }
          let iterativeFixedPointStatus = await predictForDistance(
            predictionSetup,
            index,
            forcePredictTimes,
            forcePredictStop,
            repeatThreshold
          )
          if (iterativeFixedPointStatus[0] === null) {
            console.log(iterativeFixedPointStatus[1], index)
          }
          iterativeFixedPointStatusByIndex.push(iterativeFixedPointStatus[0])
        }

        let jsonToSave = JSON.stringify(iterativeFixedPointStatusByIndex)
        fs.writeFile(
          `public/images/${predictFolder}/iterativeFixedPoints/${modelName}.json`,
          JSON.stringify(jsonToSave),
          (err) => {
            if (err) throw err
            console.log(`${modelName}.json is saved in IterativeFixedPoints`)
          }
        )
      }
    }

    if (runRepeatPrediction && !repeatPredictForDistance) {
      for (let i = 0; i < amountOfRepeats; i++) {
        predictionPics = await predictWithTrainedEncoder(predictionSetup)
        predictionSetup.testImages = predictionPics
        if (i % 100 === 0) {
          console.log(i) // to get a grip of time passed.
        }
      }
    }

    if (!repeatPredictForDistance) {
      let distanceArray = []
      if (predictionPics.length > 0) {
        distanceArray = getClosestTrainingPicture(
          predictionPics,
          picturesTrainedWith
        )
      }

      if (predictionPics) {
        for (let i = 0; i < predictionPics.length; i++) {
          savePNGImageData(predictionPics[i], `out${i}`)
          filesOut.push(`out${i}`)
        }
      }
      const filesTrainingLength = picturesTrainedWith.length

      res.render('index', {
        title: 'Express',
        filesIn,
        filesOut,
        modelNameToPredict,
        predictFolder,
        distanceArray,
        filesTrainingLength,
      })
    }
  }
})

/**
 * This functions looks for iterative fixed points, if this runs on all pictures in a set, it can stall as
 * this function can create memory issues in the Node environment using only 2GB of memory, especially on bad models that run thousands of predictions.
 * @param {*} setup
 * @param {*} trainingIndex
 * @param {*} forcePredictTimes
 * @param {*} forcePredictStop
 * @param {*} repeatThreshold
 */
async function predictForDistance(
  setup,
  trainingIndex,
  forcePredictTimes,
  forcePredictStop,
  repeatThreshold
) {
  const {picturesTrainedWith} = setup

  let startPic = JSON.parse(JSON.stringify(setup.testImages[0]))
  let previousPic = JSON.parse(JSON.stringify(setup.testImages[0]))
  predictionPics = await predictWithTrainedEncoder(setup)
  let currentPic = predictionPics[0]
  let count = 1 // one re-prediction is already done

  while (
    getDistanceBetweenTwoPictures(previousPic, currentPic) > repeatThreshold ||
    forcePredictTimes !== 0
  ) {
    previousPic = currentPic
    setup.testImages = predictionPics
    predictionPics = await predictWithTrainedEncoder(setup)
    currentPic = predictionPics[0]

    if (forcePredictTimes > 0) {
      forcePredictTimes--
    }

    count++

    if (count % 1000 === 0) {
      console.log(count)
      console.log(
        'Distance between previous and current pic: ' +
          getDistanceBetweenTwoPictures(previousPic, currentPic)
      )
      console.log(
        'from Startpic: ' + getDistanceBetweenTwoPictures(currentPic, startPic)
      )
    }
    if (count > forcePredictStop) {
      console.log(
        `Stopped at ${forcePredictStop}, distance: ` +
          getDistanceBetweenTwoPictures(previousPic, currentPic)
      )
      break
    }
  }

  let currentDistanceArray = getClosestTrainingPicture(
    predictionPics,
    picturesTrainedWith
  )

  let distance = getDistanceBetweenTwoPictures(currentPic, startPic)
  console.log('--------------------------------------')
  console.log('Iterations until stable: ' + count)
  console.log(
    'trainingIndex: ' + trainingIndex,
    ' | currentClosestIndex: ' + currentDistanceArray[0][1]
  )

  if (currentDistanceArray[0][1] === trainingIndex) {
    return count > forcePredictStop ? [null, distance] : [true, distance]
  } else {
    return [false, distance]
  }
}

/**
 * Should be run after the 'predictForDistance' to remove undefined iterative fixed points in the JSON files.
 * This can create memory issues in the Node environment as standard is only 2GB of memory.
 * @param {*} setup
 * @param {*} repeatThreshold
 */
async function reRunUndefinedPredictions(setup, repeatThreshold) {
  const {modelNameToPredict, predictFolder, picturesTrainedWith} = setup
  let iterativeFixedPointsResults = await fetch(
    `http://localhost:${process.env.PORT}/images/${predictFolder}/iterativeFixedPoints/${modelNameToPredict}.json`
  )
  let iterativeFixedPoints = JSON.parse(
    await iterativeFixedPointsResults.json()
  )

  let undefinedIndicies = []
  iterativeFixedPoints.forEach((itf, index) => {
    if (itf === null) {
      undefinedIndicies.push(index)
    }
  })
  let picturesToRePredict = picturesTrainedWith.filter((pic, index) => {
    return undefinedIndicies.includes(index)
  })
  console.log(undefinedIndicies)
  if (undefinedIndicies.length > 0) {
    let index = 0
    setup.testImages = [picturesToRePredict[index]]
    try {
      let result = await predictForDistance(
        setup,
        undefinedIndicies[index],
        0,
        32000,
        repeatThreshold
      )
      let newStatus = result[0] === true ? true : false
      console.log(
        'Null changed to ' +
          newStatus +
          ' @ index : ' +
          undefinedIndicies[index]
      )

      iterativeFixedPoints[undefinedIndicies[index]] = newStatus

      let jsonToSave = JSON.stringify(iterativeFixedPoints)
      fs.writeFile(
        `public/images/${predictFolder}/iterativeFixedPoints/${modelNameToPredict}.json`,
        JSON.stringify(jsonToSave),
        (err) => {
          if (err) throw err
          console.log(
            `${modelNameToPredict}.json is saved in IterativeFixedPoints`
          )
        }
      )
    } catch (error) {
      console.log(error)
    }
  } else {
    console.log(`No indicies for ${modelNameToPredict} to run...`)
  }
}

/**
 * Used to retrieve all iterative fixed points from a model.
 * @param {*} predictFolder
 * @param {*} modelNameToPredict
 */
async function getIterativeFixedPointsForModel(
  predictFolder,
  modelNameToPredict
) {
  let iterativeFixedPointsResults = await fetch(
    `http://localhost:${process.env.PORT}/images/${predictFolder}/iterativeFixedPoints/${modelNameToPredict}.json`
  )
  let iterativeFixedPoints = JSON.parse(
    await iterativeFixedPointsResults.json()
  )
  let array = []
  iterativeFixedPoints.forEach((fp, index) => {
    if (fp) {
      array.push(index)
    }
  })
  return array
}

/**
 * Used to retrieve non-iterative fixed points with low eigenvalues from a model
 * @param {*} predictFolder
 * @param {*} modelNameToPredict
 */
async function getNonIterativeFixedPointsLowEigenvaluesForModel(
  predictFolder,
  modelNameToPredict
) {
  let iterativeFixedPointsResults = await fetch(
    `http://localhost:${process.env.PORT}/models/${predictFolder}/iterativeFixedPoints/${modelNameToPredict}.json`
  )
  let highestEigenvalues = await fetch(
    `http://localhost:${process.env.PORT}/models/${predictFolder}/highestEigenvalues/${modelNameToPredict}.json`
  )
  let iterativeFixedPoints = JSON.parse(
    await iterativeFixedPointsResults.json()
  )
  let eigenvalues = await highestEigenvalues.json()
  let array = []
  iterativeFixedPoints.forEach((fp, index) => {
    if (!fp) {
      if (eigenvalues[index] < 1) {
        console.log(
          'index: ' +
            index +
            '(' +
            fp +
            ') ' +
            ' eigenvalue: ' +
            eigenvalues[index]
        )
        array.push(index)
      }
    }
  })
  return array
}

/**
 * Used in conjuction with getNonIterativeFixedPointsLowEigenvaluesForModel or getIterativeFixedPointsForModel
 *  to generate images based on these outputs for prediction to view.
 * @param {*} picturesTrainedWith
 * @param {*} itfpArray
 */
async function generateTestImagesFromItfpsArray(
  picturesTrainedWith,
  itfpArray
) {
  let testImages = []
  itfps.forEach((itfp) => {
    testImages.push(picturesTrainedWith[itfp])
  })
  return testImages
}

module.exports = router
