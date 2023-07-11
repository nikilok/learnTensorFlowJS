import * as tf from '@tensorflow/tfjs-node-gpu'

const getArrayOfSize = (length): number[] =>
  Array.from({ length }, () => Math.floor(Math.random() * 100))

const convertBytesToMb = (num: number): number => num / 1048576
let result

tf.tidy(() => {
  const mat1 = tf.tensor([91, 82, 13, 15, 23, 62, 25, 66, 63], [3, 3])
  const mat2 = tf.tensor([1, 23, 83, 33, 12, 5, 7, 23, 61], [3, 3])
  tf.matMul(mat1, mat2).print()

  const matA = tf.tensor([1, 2, 3], [3, 1])
  const matB = tf.tensor([2, 4, 6], [1, 3])
  tf.matMul(matA, matB).print()

  const matARan100 = tf.tensor(getArrayOfSize(10e3), [250, 40])
  const matBRan100 = tf.tensor(getArrayOfSize(10e3), [40, 250])
  result = tf.matMul(matARan100, matBRan100)
  return result
})

console.log(result.arraySync())

console.log('Memory of tensors (mb)', convertBytesToMb(tf.memory().numBytes))
result.dispose()
console.log('Memory of tensors (mb)', convertBytesToMb(tf.memory().numBytes))
// if 9 -> 3 x 3
// if 100 -> 50 x 2
