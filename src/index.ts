// GPU Based Tensorflow for Node
import * as tf from '@tensorflow/tfjs-node-gpu'
// CPU Based Tensorflow for Node
// import * as tf from "@tensorflow/tfjs-node";

console.log('🚀 version', tf.version.tfjs)
const firstTensore = tf.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9], [3, 3], 'float32')
console.log('🚀 ~ file: index.js:5 ~ firstTensor:', firstTensore)
console.log('🚀 ~ file: index.js:5 ~ size:', firstTensore.size)
console.log('🚀 ~ file: index.js:5 ~ ranks:', firstTensore.rank)

const dataArray = Array.from(Array(1e6).keys())
const largeTensor = tf.tensor(dataArray, [1e3, 1e3])
console.log('🚀🚀 ~ file: index.js:11 ~ largeTensor:', largeTensor.size)

console.log('original', firstTensore.dataSync())
const integerTensore = firstTensore.asType('int32')
console.log('result', integerTensore.dataSync())

console.log('Memory used', tf.memory().numBytes)
console.log('No Tensors', tf.memory().numTensors)
// Garbage collects the Tensor
largeTensor.dispose()
console.log('Memory used', tf.memory().numBytes)
console.log('No Tensors', tf.memory().numTensors)
