const fetch = require('node-fetch')

/**
 * These are used in conjuction with each other:
 * #1 (first)     --> create a batch of 128 experiments,
 *                --> then a batch of 64 experiments,
 *                --> then a batch of 32 experiments.
 *                --> Using the 'makeBatchFileForComparison'.
 * #2 (secondly)  --> use 'getResultsFrombatch' function on each of the results to get the data.
 * #3 (thirdly)   --> use 'createTableFromThreeBatches' with the three generated resultbatches (in order: 128-->32).
 *  (these funtions can be used in other ways but this is how it was planned for the thesis)
 */

module.exports = {
  createTableFromThreeBatches: function (data128, data64, data32) {
    let LaTeXTable = `\\begin{table}[h]
    \\centering
    {\\tabulinesep=1.6mm
    \\begin{tabu}{!{\\vrule width 1pt}c|c:c|c:c|c:c!{\\vrule width 1pt}} 
    \\Xhline{1pt}
        \\backslashbox{depth}{width} & \\multicolumn{2}{c|}{128} & \\multicolumn{2}{c|}{64} & \\multicolumn{2}{c!{\\vrule width 1pt}}{32} \\\\ \\hline
        \\multirow{2}{*}{11} & \\textbf{\\cellcolor{orange!25}{${data128[0][1].itf_low}}} & ${data128[0][1].itf_high} & \\textbf{\\cellcolor{orange!25}{${data64[0][1].itf_low}}} & ${data64[0][1].itf_high} & \\textbf{\\cellcolor{orange!25}{${data32[0][1].itf_low}}} & ${data32[0][1].itf_high} \\\\ 
        \\cdashline{2-7}
        & ${data128[0][1].noitf_low} & ${data128[0][1].noitf_high} & ${data64[0][1].noitf_low} & ${data64[0][1].noitf_high} & ${data32[0][1].noitf_low} & ${data32[0][1].noitf_high} \\\\ \\hline
        \\multirow{2}{*}{6} & \\textbf{\\cellcolor{orange!25}{${data128[1][1].itf_low}}} & ${data128[1][1].itf_high} & \\textbf{\\cellcolor{orange!25}{${data64[1][1].itf_low}}} & ${data64[1][1].itf_high} & \\textbf{\\cellcolor{orange!25}{${data32[1][1].itf_low}}} & ${data32[1][1].itf_high} \\\\ 
        \\cdashline{2-7}
        & ${data128[1][1].noitf_low} & ${data128[1][1].noitf_high} & ${data64[1][1].noitf_low} & ${data64[1][1].noitf_high} & ${data32[1][1].noitf_low} & ${data32[1][1].noitf_high} \\\\ \\hline
        \\multirow{2}{*}{3} & \\textbf{\\cellcolor{orange!25}{${data128[2][1].itf_low}}} & ${data128[2][1].itf_high} & \\textbf{\\cellcolor{orange!25}{${data64[2][1].itf_low}}} & ${data64[2][1].itf_high} & \\textbf{\\cellcolor{orange!25}{${data32[2][1].itf_low}}} & ${data32[2][1].itf_high} \\\\ 
        \\cdashline{2-7}
        & ${data128[2][1].noitf_low} & ${data128[2][1].noitf_high} & ${data64[2][1].noitf_low} & ${data64[2][1].noitf_high} & ${data32[2][1].noitf_low} & ${data32[2][1].noitf_high} \\\\ \\hline
        \\multirow{2}{*}{2} & \\textbf{\\cellcolor{orange!25}{${data128[3][1].itf_low}}} & ${data128[3][1].itf_high} & \\textbf{\\cellcolor{orange!25}{${data64[3][1].itf_low}}} & ${data64[3][1].itf_high} & \\textbf{\\cellcolor{orange!25}{${data32[3][1].itf_low}}} & ${data32[3][1].itf_high} \\\\ 
        \\cdashline{2-7}
        & ${data128[3][1].noitf_low} & ${data128[3][1].noitf_high} & ${data64[3][1].noitf_low} & ${data64[3][1].noitf_high} & ${data32[3][1].noitf_low} & ${data32[3][1].noitf_high} \\\\ \\hline
        \\multirow{2}{*}{1} & \\textbf{\\cellcolor{orange!25}{${data128[4][1].itf_low}}} & ${data128[4][1].itf_high} & \\textbf{\\cellcolor{orange!25}{${data64[4][1].itf_low}}} & ${data64[4][1].itf_high} & \\textbf{\\cellcolor{orange!25}{${data32[4][1].itf_low}}} & ${data32[4][1].itf_high} \\\\ 
        \\cdashline{2-7}
        & ${data128[4][1].noitf_low} & ${data128[4][1].noitf_high} & ${data64[4][1].noitf_low} & ${data64[4][1].noitf_high} & ${data32[4][1].noitf_low} & ${data32[4][1].noitf_high} \\\\ \\hline
         \\Xhline{1pt}
    \\end{tabu}}
    \\caption{Average number of training images in the four categories shown in Table~\\ref{tab:four_categories}.}
    \\label{tab:resultsWithBiasCIFAR}
\\end{table}`
    console.log(LaTeXTable)
  },

  makeBatchFileForComparison: function (predictFolders) {
    let batch = []

    predictFolders.forEach((predictFolder) => {
      let modelsToRepeatPredict = [
        predictFolder + '1',
        predictFolder + '2',
        predictFolder + '3',
        predictFolder + '4',
      ]
      batch.push([predictFolder, modelsToRepeatPredict])
    })
    return batch
  },

  getResultsFromBatch: async function (batch) {
    let batchResults = await Promise.all(
      batch.map(async (batch) => {
        let predictFolder = batch[0]
        let resultArray = await Promise.all(
          batch[1].map(async (modelName) => {
            return runComparisonFunc(predictFolder, modelName)
          })
        )
        console.log(resultArray)
        return [predictFolder, getAverageITFFunc(resultArray)]
      })
    )
    return batchResults
  },

  getAverageITFs: function (resultArrays) {
    return getAverageITFFunc(resultArrays)
  },

  runComparison: async function (predictFolder, modelName) {
    return await runComparisonFunc(predictFolder, modelName)
  },
}

async function runComparisonFunc(predictFolder, modelName) {
  let results = {
    itf_low: 0,
    itf_high: 0,
    noitf_low: 0,
    noitf_high: 0,
  }
  let highEigResults = await fetch(
    `http://localhost:${process.env.PORT}/images/${predictFolder}/highestEigenvalues/${modelName}.json`
  )
  let iterativeFixedPointsResults = await fetch(
    `http://localhost:${process.env.PORT}/images/${predictFolder}/iterativeFixedPoints/${modelName}.json`
  )
  let iterativeFixedPoints = JSON.parse(
    await iterativeFixedPointsResults.json()
  )
  let highestEigenValues = await highEigResults.json()

  if (iterativeFixedPoints) {
    iterativeFixedPoints.forEach((result, index) => {
      if (result === true && highestEigenValues[index] < 1) {
        results.itf_low++
        // console.log(index, highestEigenValues[index])
      }
      if (result === true && highestEigenValues[index] > 1) {
        results.itf_high++
        // console.log(index, highestEigenValues[index])
      }
      if (result === false && highestEigenValues[index] < 1) {
        results.noitf_low++
        //   console.log(index, highestEigenValues[index])
      }
      if (result === false && highestEigenValues[index] > 1) {
        results.noitf_high++
        // console.log(index, highestEigenValues[index])
      }
    })

    //   console.log(modelName + ' : ' + results.itf_low + ' Attractors found.')
  }
  return results
}

function getAverageITFFunc(resultArrays) {
  let sum = {
    itf_low: 0,
    itf_high: 0,
    noitf_low: 0,
    noitf_high: 0,
  }
  resultArrays.forEach((result) => {
    sum.itf_low += result.itf_low
    sum.itf_high += result.itf_high
    sum.noitf_low += result.noitf_low
    sum.noitf_high += result.noitf_high
  })
  let average = {}
  Object.keys(sum).forEach((key) => {
    average[key] = sum[key] / resultArrays.length
  })
  return average
}
