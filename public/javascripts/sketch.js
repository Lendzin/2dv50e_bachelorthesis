/**
 * Renders the different pictures form the application on a Canvas with its euclidian distances.
 * if files are misplaced, this will not run, and errors can be watched in the console of your web-client.
 */

let imagesIn = []
let imagesOut = []
let imagesTraining = []

const filesIn = JSON.parse(
  window.document.currentScript.getAttribute('filesIn')
)
const filesOut = JSON.parse(
  window.document.currentScript.getAttribute('filesOut')
)
const distanceArray = JSON.parse(
  window.document.currentScript.getAttribute('distanceArray')
)
const trainingLength = JSON.parse(
  window.document.currentScript.getAttribute('filesTrainingLength')
)

const modelName = window.document.currentScript.getAttribute('modelName')
const folder = window.document.currentScript.getAttribute('folder')

function preload() {
  filesIn.forEach((name) => {
    imagesIn.push(loadImage(`../images/${name}.png`))
  })
  filesOut.forEach((name) => {
    imagesOut.push(loadImage(`../images/${name}.png`))
  })
  for (let i = 0; i < trainingLength; i++) {
    if (folder) {
      imagesTraining.push(
        loadImage(
          `../images/${folder}/${modelName}/${modelName}training${i}.png`
        )
      )
    } else {
      imagesTraining.push(
        loadImage(`../images//${modelName}/${modelName}training${i}.png`)
      )
    }
  }
}

async function setup() {
  createCanvas(1500, 2000)
}

function draw() {
  let positionX = 0
  let positionY = 30
  let batchSize = filesOut.length > 10 ? 20 : 10
  let filesLeft = filesOut.length

  while (filesLeft > 0) {
    line(0, positionY - 20, 1500, positionY - 20)
    let currentIValue = filesOut.length - filesLeft
    text('Training images', 10, positionY - 4)
    for (let i = currentIValue; i < currentIValue + batchSize; i++) {
      if (imagesTraining[i]) {
        image(imagesTraining[i], positionX, positionY, 64, 64)
      }
      positionX += 64
    }
    positionX = 0
    positionY += 80
    text('Testing images', 10, positionY - 4)
    for (let i = currentIValue; i < currentIValue + batchSize; i++) {
      if (imagesIn[i]) {
        image(imagesIn[i], positionX, positionY, 64, 64)
      }
      positionX += 64
    }
    positionX = 0
    positionY += 80
    text('Result images', 10, positionY - 4)
    for (let i = currentIValue; i < currentIValue + batchSize; i++) {
      if (imagesOut[i]) {
        image(imagesOut[i], positionX, positionY, 64, 64)
      }
      text(
        distanceArray[i][0].toFixed(2) + ' / ' + distanceArray[i][1],
        positionX,
        positionY + 84
      )
      if (imagesTraining[distanceArray[i][1]]) {
        image(
          imagesTraining[distanceArray[i][1]],
          positionX,
          positionY + 100,
          64,
          64
        )
      }

      positionX += 64
    }
    positionX = 0
    positionY += 200
    filesLeft -= batchSize
  }
}
