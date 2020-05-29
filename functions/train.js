var getSequentialModel = require('./helperfunctions.js').getSequentialModel
var savePNGImageData = require('./helperfunctions.js').savePNGImageData
const tf = require('@tensorflow/tfjs-node')
var fs = require('fs')

module.exports = {
  trainAutoEncoder: async function (setup) {
    // setup breakdown:
    const {
      trainingImages,
      modelName,
      trainingEpochs,
      loadOldModel,
      oldModelToLoad,
      folderToLoad,
      activation,
      bias,
      hiddenLayers,
      unitsPerLayer,
      lossFunction,
      learningRate,
      stopAtLoss,
      stopAtEpochs,
      optimizer,
      inputShape,
      saveTrainingImagesAsJson,
      momentum,
      decay,
    } = setup
    if (saveTrainingImagesAsJson) {
      trainingImages.forEach((image, index) => {
        // write json file
        let json = JSON.stringify(image.input)
        fs.writeFile(
          __dirname + `/jsons/random/100/image${index}.json`,
          json,
          'utf8',
          () => {
            console.log('written file: ' + index)
          }
        )
      })
    }

    let preData = await trainingImages.map((item) => item.input || item)
    const trainingData = await tf.tensor2d(
      await JSON.parse(JSON.stringify(preData))
    )

    const outputData = await tf.tensor2d(
      await JSON.parse(JSON.stringify(preData))
    )

    console.log(
      `----------------------- TRAINING STARTING, model: ${modelName} --------------------`
    )

    // set configuration for optimizers
    let currentOptimizer = setOptimizer(
      optimizer,
      learningRate,
      momentum,
      decay
    )

    console.log(
      '----- ModelSetup - activation: ' +
        activation +
        ' bias: ' +
        bias +
        ' hlayers: ' +
        hiddenLayers +
        ' units: ' +
        unitsPerLayer +
        ' -----'
    )

    let model = getSequentialModel(
      activation,
      bias,
      hiddenLayers,
      unitsPerLayer,
      inputShape
    )

    if (
      oldModelToLoad !== `${modelName}LowestLoss` &&
      oldModelToLoad !== `${modelName}`
    ) {
      console.log('--creating new lowestloss folder--')
      let worked = await model.save(
        `file://public/images/${modelName}LowestLoss`
      ) // save empty model to create folder-structure
      if (worked) {
        for (let i = 0; i < trainingImages.length; i++) {
          if (trainingImages[i].input) {
            savePNGImageData(
              trainingImages[i].input,
              `${modelName}LowestLoss/${modelName}LowestLosstraining${i}`
            )
          } else {
            savePNGImageData(
              trainingImages[i],
              `${modelName}LowestLoss/${modelName}LowestLosstraining${i}`
            )
          }
        }
      }
    }

    if (loadOldModel) {
      console.log(
        `---------------------------- loading OLD model: ${oldModelToLoad}--------------------------------`
      )
      if (folderToLoad != '') {
        model = await tf.loadLayersModel(
          `file://public/images/${folderToLoad}/${oldModelToLoad}/model.json`
        )
      } else {
        model = await tf.loadLayersModel(
          `file://public/images/${oldModelToLoad}/model.json`
        )
      }
    }
    // train/fit our network
    let currentLoss = Number.MAX_SAFE_INTEGER
    let lowestLoss = Number.MAX_SAFE_INTEGER
    let something = null
    let epochCount = 0
    const startTime = Date.now()

    while (currentLoss > stopAtLoss && epochCount < stopAtEpochs) {
      if (currentLoss !== Number.MAX_SAFE_INTEGER) {
        // console.log(
        //   `---------------------------- loading (current) model: ${modelName}--------------------------------`
        // )
        model.dispose()
        model = await tf.loadLayersModel(
          `file://public/images/${modelName}/model.json`
        )
      }
      // if(epochCount === 15000) {
      //   currentOptimizer = setOptimizer(optimizer, learningRate*10)
      // }
      // if (epochCount === 35000) {
      //   currentOptimizer = setOptimizer(optimizer, learningRate/10)
      // }

      model.compile({
        loss: lossFunction,
        optimizer: currentOptimizer,
      })
      let history = await model.fit(trainingData, outputData, {
        epochs: trainingEpochs,
        verbose: 0,
        callbacks: {
          onEpochEnd: async (epoch, logs) => {
            epochCount++
            if (lowestLoss > logs.loss) {
              lowestLoss = logs.loss
              let justbecause = await model.save(
                `file://public/images/${modelName}LowestLoss`
              )
            }
            currentLoss = logs.loss
            if (loadOldModel && epochCount === 1) {
              console.log(currentLoss)
            } else if (epochCount === 1) {
              console.log(currentLoss)
            }
            if (epochCount % 10000 === 0) {
              console.log('epoch: ' + epochCount + ' loss: ' + currentLoss)
              console.log('Lowest loss recorded: ' + lowestLoss)
            }
          },
        },
      })
      something = await model.save(`file://public/images/${modelName}`)
    }

    const endTime = Date.now()
    if (something) {
      console.log((endTime - startTime) / 1000 + ' seconds')
      console.log('Current loss: ' + currentLoss)
      console.log('Lowest loss recorded: ' + lowestLoss)

      for (let i = 0; i < trainingImages.length; i++) {
        if (trainingImages[i].input) {
          savePNGImageData(
            trainingImages[i].input,
            `${modelName}/${modelName}training${i}`
          )
          savePNGImageData(
            trainingImages[i].input,
            `${modelName}LowestLoss/${modelName}LowestLosstraining${i}`
          )
        } else {
          savePNGImageData(
            trainingImages[i],
            `${modelName}/${modelName}training${i}`
          )
          savePNGImageData(
            trainingImages[i],
            `${modelName}LowestLoss/${modelName}LowestLosstraining${i}`
          )
        }
      }
    }

    return true
  },
}

function setOptimizer(optimizer, learningRate, momentum, decay) {
  let currentOptimizer
  if (optimizer === 'adam') {
    currentOptimizer = tf.train.adam(learningRate) // other settings: beta1, beta2, epsilon
  }
  if (optimizer === 'rmsp') {
    currentOptimizer = tf.train.rmsprop(learningRate) // other settings: decay, momentum, epsilon, centered
  }
  if (optimizer === 'sgd') {
    currentOptimizer = tf.train.sgd(learningRate)
  }
  return currentOptimizer
}
