import * as tf from '@tensorflow/tfjs-node-gpu'
// import * as tf from "@tensorflow/tfjs-node"

const one = tf.tensor([1, 2, 3, 4, 5, 6])
const floatingTensor = tf.tensor([3.5151500701904297, 5.51515007019043])
const manualTwoDTensor = tf.tensor([
  [1, 2, 3],
  [4, 5, 6],
])
const autoTwoDTensor2x3 = tf.tensor([1, 2, 3, 4, 5, 6], [2, 3])
const autoTwoDTensor3x2 = tf.tensor([1, 2, 3, 4, 5, 6], [3, 2])

console.log('one', one.print())
console.log('floatTensor', floatingTensor.dataSync())
// Data sync flattens the 2d array
console.log('manual2dTensor', manualTwoDTensor.dataSync())

const originalManualTwoD = manualTwoDTensor.arraySync()
console.log(
  '🚀 ~ file: tensorData.ts:14 ~ originalManualTwoD:',
  originalManualTwoD
)

const originalAutoTwoD2x3 = autoTwoDTensor2x3.arraySync()
console.log(
  '🚀 ~ file: tensorData.ts:18 ~ originalAutoTwoD2x3:',
  originalAutoTwoD2x3
)

const originalAutoTwoD3x2 = autoTwoDTensor3x2.arraySync()
console.log(
  '🚀 ~ file: tensorData.ts:21 ~ originalAutoTwoD3x2:',
  originalAutoTwoD3x2
)

tf.dispose([
  one,
  floatingTensor,
  manualTwoDTensor,
  autoTwoDTensor2x3,
  autoTwoDTensor3x2,
])
console.log('No of Tensors', tf.memory().numTensors)
