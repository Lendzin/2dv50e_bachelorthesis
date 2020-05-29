const tf = require('@tensorflow/tfjs-node')


module.exports = {
  predictWithTrainedEncoder: async function(predictionSetup) {
    tf.engine().startScope()
    const {testImages, modelNameToPredict, predictFolder} = predictionSetup
    let images = await JSON.parse(JSON.stringify(testImages))
    let preData = await images.map(item => item.input || item)
    
    let testingData = await tf.tensor2d(preData)
    let model
    if (predictFolder !== '') {
      model = await tf.loadLayersModel(
        `file://public/images/${predictFolder}/${modelNameToPredict}/model.json`
      )
    } else {
      model = await tf.loadLayersModel(
        `file://public/images/${modelNameToPredict}/model.json`
      )
    }
    let predictions = await model.predict(testingData)

    let predictionsDesynced = await predictions.array()
    model.dispose()
    tf.dispose(model)
    tf.dispose(testingData)
    tf.dispose(predictions)
    tf.disposeVariables()
    tf.engine().endScope()
    
    return predictionsDesynced
  },
}
