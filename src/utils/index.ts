import fs from 'fs'
import * as tf from '@tensorflow/tfjs-node-gpu'

/**
 * Convert a tensor into a jpg image on disk
 * @param img
 * @param fileName
 */
export const encodeJpeg = async (img: tf.Tensor<tf.Rank>, fileName: string) => {
  const data = await tf.node.encodeJpeg(img as tf.Tensor3D)
  fs.writeFileSync(fileName, data)
}

/**
 * Convert a tensor into an png image on disk
 * @param img
 * @param fileName
 */
export const encodePng = async (img: tf.Tensor<tf.Rank>, fileName: string) => {
  const data = await tf.node.encodePng(img as tf.Tensor3D)
  fs.writeFileSync(fileName, data)
}
