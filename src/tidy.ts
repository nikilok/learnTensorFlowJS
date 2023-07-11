/**
 * An example on how tidy works
 */
import * as tf from '@tensorflow/tfjs-node'

let keeper, chaser, seeker, beater

tf.tidy(() => {
  keeper = tf.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9], [3, 3], 'int32')
  chaser = tf.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9], null, 'int32')
  seeker = tf.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9], null, 'int32')
  beater = tf.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9], null, 'int32')
  // Now we have 4 tensors in memory
  console.log(
    'No of tensors inside tidy',
    tf.memory().numTensors,
    tf.memory().numBytes
  )

  // protect a tensor
  // tf.keep(keeper)
  console.log('keeper rank', keeper.rank)

  // by returning tensors from a tidy block you protect those tensors
  // the rest gets GC.
  return { chaser, keeper }
})

console.log('After Tidy', tf.memory().numTensors, tf.memory().numBytes)
// Manually dispose of tensors with tensor.dispose()
keeper.dispose()
chaser.dispose()
console.log('After Dispose', tf.memory().numTensors, tf.memory().numBytes)
