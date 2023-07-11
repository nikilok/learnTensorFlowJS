/**
 * A simple recommendation engine based on matrix multiplication
 */
import * as tf from '@tensorflow/tfjs-node-gpu'
tf.tidy(() => {
  const users = ['user a', 'user b', 'user c', 'user d']

  // only for reference purposes, as users are asked to rate
  // the below bands
  const bands = [
    'Nivana',
    'Rocker Smith',
    'DJ boys',
    'Smith A',
    'Basilica',
    'Vanity Fair',
    'Michal Tors',
  ]

  // certain broad features bands can be related to
  const features = new Map([
    [0, 'Grunge'],
    [1, 'Rock'],
    [2, 'Industrial'],
    [3, 'Boy Band'],
    [4, 'Dance'],
    [5, 'Jazz'],
  ])

  console.log('feature size', features.size)

  // each users vote for the different bands
  // row 1 = user a's vote against all 7 bands.
  const userVotes = tf.tensor([
    [10, 9, 1, 1, 8, 7, 8],
    [6, 8, 2, 2, 0, 10, 0],
    [0, 2, 10, 9, 3, 7, 0],
    [7, 4, 2, 3, 6, 5, 5],
  ])

  // bands co-relation to a specific feature
  // 1 indicates the band is related to a feature and
  // 0 indicates its not.
  const bandFeats = tf.tensor([
    [1, 1, 0, 0, 0, 0], // Nivana
    [1, 0, 1, 0, 0, 0], // Rocker Smith
    [0, 0, 0, 1, 1, 0], // DJ boys
    [0, 0, 0, 1, 0, 0], // Smith A
    [0, 0, 1, 0, 0, 1], // Basilica
    [0, 0, 1, 0, 0, 1], // Vanity Fair
    [1, 1, 0, 0, 0, 0], // Michal Tors
  ])

  // dot product of user vote to band feature
  const userFeatures = tf.matMul(userVotes, bandFeats)
  userFeatures.print()

  // get the top n user features
  // const topUserFeatures = tf.topk(userFeatures, features.length)
  // tensorflow lets you access most methods from either tf or the tensor directly
  const topUserFeatures = userFeatures.topk(features.size)
  /**
   * Output: topUserFeatures
   * [[27, 18, 24, 2 , 1 , 15],
     [14, 6 , 18, 4 , 2 , 10],
     [2 , 0 , 12, 19, 10, 10],
     [16, 12, 15, 5 , 2 , 11]]
   */

  /**
   * tensore.topk returns back the highest to lowest values after the matrix
   * multiplication. Each vector in the matrix has an index that we are
   * interested in as the index would point to the particular feature index.
   */
  const topRecommendationByFeatureID = topUserFeatures.indices.arraySync()
  console.log(
    '🚀 ~ file: recommendation.ts:39 ~ topGenres:',
    topRecommendationByFeatureID
  )
  /**
   * Output: topRecommendationByFeatureID
   * [ 0, 2, 1, 5, 3, 4 ],
     [ 2, 0, 5, 1, 3, 4 ],
     [ 3, 2, 4, 5, 0, 1 ],
     [ 0, 2, 1, 5, 3, 4 ]
   */

  /**
   * we then map users array, and then traverse the recommendationbyfeatureid
   * to access the feature, which is what gets recommended to the user.
   */
  users.map((u, i) => {
    const rankedCategories = topRecommendationByFeatureID[i].map((v) =>
      features.get(v)
    )
    console.log(u, rankedCategories)
    /**
     * Output:
     * user a [ 'Grunge', 'Industrial', 'Rock', 'Jazz', 'Boy Band', 'Dance' ]
       user b [ 'Industrial', 'Grunge', 'Jazz', 'Rock', 'Boy Band', 'Dance' ]
       user c [ 'Boy Band', 'Industrial', 'Dance', 'Jazz', 'Grunge', 'Rock' ]
       user d [ 'Grunge', 'Industrial', 'Rock', 'Jazz', 'Boy Band', 'Dance' ]
     */
  })
})
