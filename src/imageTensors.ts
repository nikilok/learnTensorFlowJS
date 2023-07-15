import * as tf from '@tensorflow/tfjs-node-gpu'
import { encodeJpeg, encodePng, decodeImg, decodeGifImg } from './utils'

tf.tidy(() => {
  // create a 4 x 3 image (4 pixels wide, 3 pixels tall)
  const imageTensor = tf.tensor([
    [
      //  r, g, b
      [1, 1, 1], // first pixel 1st row (white pixel)
      [0, 0, 0], // second pixel 1st row (black pixel)
      [1, 1, 1], // third pixel 1st row (white pixel)
      [0, 0, 0], // 4th pixel 1st row (black pixel)
    ],
    [
      [0, 0, 0], // first pixel 2nd row (black pixel)
      [1, 1, 1], // ...
      [0, 0, 0],
      [1, 1, 1],
    ],
    [
      [1, 1, 1], // first pixel 3rd row (white pixel)
      [0, 0, 0], // ...
      [1, 1, 1],
      [0, 0, 0],
    ],
  ])
  console.log('flatImageTensor size', imageTensor.shape, imageTensor.dtype)

  console.log('imageTensor size', imageTensor.shape, imageTensor.dtype)
  // output: [3, 4, 3] float32

  // another way to represent the same information above in a flat manner
  const pixels = [
    1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1,
    1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0,
  ]

  // pixels.length and the dimensions of the matrix multiplied would need to be
  // the same in order for tensorflow to initalize the tensor.
  console.log('no of pixels', pixels.length, 3 * 4 * 3)
  //                              image height, img width, rgb (3)
  const flatImageTensor = tf.tensor(pixels, [3, 4, 3])
  console.log(
    '🚀 ~ file: imageTensors.ts:37 ~ flatImageTensor:',
    flatImageTensor.shape
  )
  // removing similar pixels
  const uniqueTensor = tf.tensor(pixels).unique()
  console.log('unique values', uniqueTensor.values.dataSync())

  // create a 1024 x 768 black pixel
  const blackImage = tf.zeros([768, 1024, 1])
  console.log('black image', blackImage.arraySync())

  const whiteImage = tf.ones([768, 1024, 3])
  console.log('black image', whiteImage.arraySync())

  const specificFill = tf.fill([20, 20, 3], 0.5)
  console.log('specific image', specificFill.arraySync())

  /**
   Filling a tile pattern
   */
  const pattern = tf.tensor([
    [[255], [0]],
    [[0], [255]],
  ])

  const patternImage = pattern.tile([2e3, 2e3, 1])
  console.log('patternImage src shape ->', patternImage.shape)

  /*
  Random pixel fill
  */
  const randomImage = tf.randomUniform([10, 10, 3])
  // console.log('randomImage', randomImage.arraySync())

  // Random number range from 0 - 255
  const randomImageBlack = tf.randomUniform([200, 200, 1], 0, 255, 'int32')
  const randomImageColorNoise = tf.randomUniform([10, 10, 3], 0, 255, 'int32')
  const patternColor = randomImageColorNoise.tile([300, 300, 1])
  const randomImageV2 = tf.randomUniform([50, 200, 4], 0, 255, 'int32')

  encodeJpeg(patternImage, './src/images/checkeredPattern.jpg')
  encodeJpeg(randomImageBlack, './src/images/sampleBlack.jpg')
  encodeJpeg(randomImageColorNoise, './src/images/sampleColor.jpg')
  encodePng(randomImageV2, './src/images/sample-with-alpha.png')
  encodeJpeg(patternColor, './src/images/color-pattern.jpg')

  // Read an image back into a tensor

  const imgTensore = decodeImg('./src/images/checkeredPattern.jpg')
  console.log(
    '🚀 ~ file: imageTensors.ts:93 ~ tf.tidy ~ imgTensore:',
    imgTensore.shape
  )

  // Read a gif image from disk
  const jumpingCake = decodeGifImg('./src/img/jumping-cake.gif')
  console.log(
    '🚀 ~ file: imageTensors.ts:100 ~ tf.tidy ~ jumpingCake:',
    jumpingCake.shape
  )
  const reversedTensor = tf
    .tensor([
      [
        [1, 2, 3],
        [2, 2, 3],
        [3, 2, 3],
      ],
      [
        [1, 2, 3],
        [2, 2, 3],
        [3, 2, 3],
      ],
      [
        [1, 2, 3],
        [2, 2, 3],
        [3, 2, 3],
      ],
    ])
    .reverse(2)
  console.log(
    '🚀 ~ file: imageTensors.ts:105 ~ tf.tidy ~ reversedTensor:',
    reversedTensor.arraySync()
  )

  // Loop through each frame of the gif and write it out as jpg
  // since we saw 7 frames in the previous log we should have 7 jpg images
  jumpingCake.arraySync().map((frame, i) => {
    // reverse (2) - flips the color.. gets a negative.
    // reverse (1) - flips the image horizontally
    // reverse (0) - flips the image vertically along the y axis.
    const imageFrameTensor = tf.tensor(frame).reverse(2)
    encodeJpeg(imageFrameTensor, `./src/images/frame${i + 1}.jpg`)
  })
})
