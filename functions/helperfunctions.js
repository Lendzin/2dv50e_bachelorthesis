const CIFAR10 = require('cifar10')({
  dataPath: __dirname + '/../node_modules/cifar10/data',
})
const fs = require('fs')
const util = require('util')
const readFile = util.promisify(fs.readFile)

const tf = require('@tensorflow/tfjs-node')
PNG = require('pngjs2').PNG

const fetch = require('node-fetch')

module.exports = {
  getSaved100RandomPictures: async function () {
    let pictures = []
    for (let i = 0; i < 100; i++) {
      let dirname = __dirname + '/jsons/random/100/image'
      dirname = i > 9 ? dirname + i + '.json' : dirname + '0' + i + '.json'
      let pic = JSON.parse(await readFile(dirname, 'utf8'))
      pictures.push(pic)
    }
    return pictures
  },

  compareDistanceArrays: function (previous, current) {
    let currentStates = []
    for (let i = 0; i < previous.length; i++) {
      let better = previous[i][0] >= current[i][0] ? true : false
      let samePicture = previous[i][1] === current[i][1] ? true : false
      let change = Math.abs(previous[i][0] - current[i][0])
      let previousPic = previous[i][1]
      let currentPic = current[i][1]
      currentStates.push([better, change, samePicture, previousPic, currentPic])
    }
    return currentStates
  },

  generateRandomNoisePictures: function (pic_size, nr_images) {
    let images = []
    for (let img = 0; img < nr_images; img++) {
      let picture = generateRandomPicture(pic_size)
      images.push(picture)
    }
    return images
  },

  getArrayOfOnePeturbedPicture: function (picture, peturbRate, number) {
    let pictures = []
    for (i = 0; i < number; i++) {
      let newPicture = perturbPicture(picture, peturbRate)
      pictures.push(newPicture)
    }
    return pictures
  },

  peturbArrayWithPictures: function (picturesIn, peturbRate) {
    let pictures = []
    picturesIn.forEach((picture) => {
      let newPicture = perturbPicture(picture.input || picture, peturbRate)
      pictures.push(newPicture)
    })
    return pictures
  },

  logDistanceOfPictures: async function (picturesTrainedWith) {
    return await logDistancesFromPics(picturesTrainedWith)
  },

  getSequentialModel: function (
    activation,
    bias,
    hiddenLayers,
    units,
    inputShape
  ) {
    let model = tf.sequential()
    model.add(
      tf.layers.dense({
        inputShape: [inputShape],
        activation: activation,
        units: units,
        useBias: bias,
      })
    )

    for (let layer = 0; layer < hiddenLayers; layer++) {
      model.add(
        tf.layers.dense({
          activation: activation,
          useBias: bias,
          units: units,
        })
      )
    }

    model.add(
      tf.layers.dense({
        activation: activation,
        useBias: bias,
        units: inputShape,
      })
    )

    return model
  },

  getDistanceBetweenTwoPictures: function (picture1, picture2) {
    let distance = getDistance(picture1, picture2)
    return distance
  },

  getDistanceToTrainingPicture: function (
    predictionPics,
    trainingImages,
    index
  ) {
    let currentDistanceArray = []
    predictionPics.forEach((x, xIndex) => {
      trainingImages.forEach((y, yIndex) => {
        let distance = getDistance(x, y.input || y)
        if (yIndex === index) {
          currentDistanceArray.push(distance)
        }
      })
    })
    return currentDistanceArray
  },

  getClosestTrainingPicture: function (predictionPics, trainingImages) {
    let closestPictures = []
    predictionPics.forEach((x, xIndex) => {
      closestPictures[xIndex] = []
      let bestDistance = Number.MAX_SAFE_INTEGER
      let bestIndex = null
      trainingImages.forEach((y, yIndex) => {
        let distance = getDistance(x, y.input || y)
        if (distance < bestDistance) {
          bestDistance = distance
          bestIndex = yIndex
        }
      })
      closestPictures[xIndex] = [bestDistance, bestIndex]
    })
    return closestPictures
  },

  getCifar10ArrayAtIndex: async function (index) {
    return await returnCifar10ArrayAtIndex(index)
  },

  getXTimesCifar10ArraysFromIndex: async function (number, startIndex) {
    let imagesToReturn = []
    for (i = startIndex; i < number + startIndex; i++) {
      let array = await returnCifar10ArrayAtIndex(i)
      imagesToReturn = [...array, ...imagesToReturn]
    }
    return imagesToReturn
  },
  getRandomPicturesColor: async function (number) {
    return await CIFAR10.training.get(number)
  },

  savePNGImageData: async function (data, name) {
    if (data.input) {
      data = data.input
    }

    if (data.length < 3072) {
      // important if saving a gray scaled picture
      data = setMultipleChannelsToGrayForPicture_1024To3072(data)
    }
    let inputData = data.map(function (v) {
      return v * 255
    })
    let imageDataBuffer = new Uint8ClampedArray(32 * 32 * 4)

    for (let rowI = 0; rowI < 32; rowI++) {
      for (let colI = 0; colI < 32; colI++) {
        let pos = (rowI * 32 + colI) * 4
        imageDataBuffer[pos] = inputData[rowI * 32 + colI]
        imageDataBuffer[pos + 1] = inputData[rowI * 32 + colI + 1024]
        imageDataBuffer[pos + 2] = inputData[rowI * 32 + colI + 2048]
        imageDataBuffer[pos + 3] = 255
      }
    }

    let img_png = new PNG({width: 32, height: 32})
    img_png.data = Buffer.from(imageDataBuffer)
    img_png.pack().pipe(fs.createWriteStream(`public/images/${name}.png`))
  },

  convertPicturesToGray_3072To1024: async function (pictures) {
    let newPictures = pictures.map((picture) => {
      let count = 0
      let newPicture = []
      while (count < picture.input.length / 3) {
        let red = picture.input[count]
        let green = picture.input[count + 1024]
        let blue = picture.input[count + 2048]
        let gray = (red + green + blue) / 3
        newPicture.push(gray)
        count++
      }
      return newPicture
    })
    return newPictures
  },

  shuffle: function (orgArray) {
    let array = JSON.parse(JSON.stringify(orgArray))
    let currentIndex = array.length,
      temporaryValue,
      randomIndex

    // While there remain elements to shuffle...
    while (0 !== currentIndex) {
      // Pick a remaining element...
      randomIndex = Math.floor(Math.random() * currentIndex)
      currentIndex -= 1

      // And swap it with the current element.
      temporaryValue = array[currentIndex]
      array[currentIndex] = array[randomIndex]
      array[randomIndex] = temporaryValue
    }

    return array
  },
}

function generateRandomPicture(pic_size) {
  let image = {}
  image.input = []
  for (let channelValue = 0; channelValue < pic_size; channelValue++) {
    image.input.push(Math.random())
  }
  return image
}

function getDistance(xArray, yArray) {
  if (xArray.input) {
    xArray = xArray.input
  }
  if (yArray.input) {
    yArray = yArray.input
  }
  let sum = 0
  xArray.forEach((xVal, xIndex) => {
    let yVal = yArray[xIndex]
    sum += Math.pow(xVal - yVal, 2)
  })
  sum = Math.sqrt(sum)

  return sum
}

function setMultipleChannelsToGrayForPicture_1024To3072(picture) {
  let newPicture = []
  for (i = 0; i < 3072; i++) {
    newPicture.push(0)
  }

  picture.forEach((pixel, count) => {
    let currentColor = pixel
    newPicture[count] = currentColor
    newPicture[count + 1024] = currentColor
    newPicture[count + 2048] = currentColor
  })
  return newPicture
}

function perturbPicture(picture, peturbRate) {
  if (picture.input) {
    picture = picture.input
  }
  newPicture = picture.map((value) => {
    let randomizedValue = (Math.random() - 0.5) * peturbRate
    return value + randomizedValue
  })

  return newPicture
}

async function returnCifar10ArrayAtIndex(index) {
  index = index == 0 ? 1 : index
  let imagesToReturn = []
  imagesToReturn = [
    ...(await CIFAR10.cat.range(index - 1, index)),
    ...imagesToReturn,
  ]
  imagesToReturn = [
    ...(await CIFAR10.airplane.range(index - 1, index)),
    ...imagesToReturn,
  ]
  imagesToReturn = [
    ...(await CIFAR10.automobile.range(index - 1, index)),
    ...imagesToReturn,
  ]
  imagesToReturn = [
    ...(await CIFAR10.bird.range(index - 1, index)),
    ...imagesToReturn,
  ]
  imagesToReturn = [
    ...(await CIFAR10.deer.range(index - 1, index)),
    ...imagesToReturn,
  ]
  imagesToReturn = [
    ...(await CIFAR10.dog.range(index - 1, index)),
    ...imagesToReturn,
  ]
  imagesToReturn = [
    ...(await CIFAR10.frog.range(index - 1, index)),
    ...imagesToReturn,
  ]
  imagesToReturn = [
    ...(await CIFAR10.horse.range(index - 1, index)),
    ...imagesToReturn,
  ]
  imagesToReturn = [
    ...(await CIFAR10.ship.range(index - 1, index)),
    ...imagesToReturn,
  ]
  imagesToReturn = [
    ...(await CIFAR10.truck.range(index - 1, index)),
    ...imagesToReturn,
  ]
  return imagesToReturn
}

async function logDistancesFromPics(picturesTrainedWith) {
  let maximumDistance = 0
  let minimumDistance = Number.MAX_SAFE_INTEGER
  let bestPics = []
  let worstPics = []
  let count = 0
  for (let index1 = 0; index1 < 99; index1++) {
    for (let index2 = index1 + 1; index2 < 100; index2++) {
      let distance = await getDistance(
        picturesTrainedWith[index1],
        picturesTrainedWith[index2]
      )
      if (maximumDistance < distance) {
        worstPics = [index1, index2]
        maximumDistance = distance
      }
      count++
      if (minimumDistance > distance) {
        bestPics = [index1, index2]
        minimumDistance = distance
      }
    }
  }
  console.log(
    'MaxDistance, count, indexes for pics: ' + maximumDistance,
    count,
    worstPics
  )
  console.log(
    'MinDistance, count, indexes for pics: ' + minimumDistance,
    count,
    bestPics
  )
}
