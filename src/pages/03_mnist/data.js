/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

const tf = require('@tensorflow/tfjs');
const fetchWechat = require('fetch-wechat');

const IMAGE_SIZE = 784;
const NUM_CLASSES = 10;
const NUM_DATASET_ELEMENTS = 65000;

const NUM_TRAIN_ELEMENTS = 55000;
const NUM_TEST_ELEMENTS = NUM_DATASET_ELEMENTS - NUM_TRAIN_ELEMENTS;

const MNIST_IMAGES_SPRITE_PATH =
  'https://ilego.club/ai/dataset/mnist_images.png';
const MNIST_LABELS_PATH =
  'https://ilego.club/ai/dataset/mnist_labels_uint8';

/**
 * A class that fetches the sprited MNIST dataset and returns shuffled batches.
 *
 * NOTE: This will get much easier. For now, we do data fetching and
 * manipulation manually.
 */
export class MnistData {
  constructor() {
    this.shuffledTrainIndex = 0;
    this.shuffledTestIndex = 0;
    this.page;
  }

  async load(canvasId, imgWidth, imgHeight) {
    const ctx = wx.createCanvasContext(canvasId);
    
    const datasetBytesBuffer =
      new ArrayBuffer(NUM_DATASET_ELEMENTS * IMAGE_SIZE * 4);

    const chunkSize = 5000;

    let drawJobs = [];
    for (let i = 0; i < NUM_DATASET_ELEMENTS / chunkSize; i++) {
      const datasetBytesView = new Float32Array(
        datasetBytesBuffer, i * IMAGE_SIZE * chunkSize * 4,
        IMAGE_SIZE * chunkSize);
      ctx.drawImage(
        MNIST_IMAGES_SPRITE_PATH, 0, i * chunkSize, imgWidth, chunkSize, 0, 0, imgWidth,
        chunkSize);

      drawJobs.push(new Promise((resolve, reject) => {
        ctx.draw(false, () => {
          console.log("ctx.draw");

          // API 1.9.0 获取图像数据
          wx.canvasGetImageData({
            canvasId: canvasId,
            x: 0,
            y: 0,
            width: imgWidth,
            height: chunkSize,
            success(imageData) {
              for (let j = 0; j < imageData.data.length / 4; j++) {
                // All channels hold an equal value since the image is grayscale, so
                // just read the red channel.
                datasetBytesView[j] = imageData.data[j * 4] / 255;
              }
              resolve();
            },
            fail: e => {
              console.error(e);
              resolve();
            },
          });
        });
      }));
    }
    await Promise.all(drawJobs);

    this.datasetImages = new Float32Array(datasetBytesBuffer);

    console.log("all images drawn!");
    const fetch = fetchWechat.fetchFunc();
    const labelsResponse = await fetch(MNIST_LABELS_PATH);

    this.datasetLabels = new Uint8Array(await labelsResponse.arrayBuffer());

    // Create shuffled indices into the train/test set for when we select a
    // random dataset element for training / validation.
    this.trainIndices = tf.util.createShuffledIndices(NUM_TRAIN_ELEMENTS);
    this.testIndices = tf.util.createShuffledIndices(NUM_TEST_ELEMENTS);

    // Slice the the images and labels into train and test sets.
    this.trainImages =
      this.datasetImages.slice(0, IMAGE_SIZE * NUM_TRAIN_ELEMENTS);
    this.testImages = this.datasetImages.slice(IMAGE_SIZE * NUM_TRAIN_ELEMENTS);
    this.trainLabels =
      this.datasetLabels.slice(0, NUM_CLASSES * NUM_TRAIN_ELEMENTS);
    this.testLabels =
      this.datasetLabels.slice(NUM_CLASSES * NUM_TRAIN_ELEMENTS);
    console.log("data loaded!");
  }

  nextTrainBatch(batchSize) {
    return this.nextBatch(
      batchSize, [this.trainImages, this.trainLabels], () => {
        this.shuffledTrainIndex =
          (this.shuffledTrainIndex + 1) % this.trainIndices.length;
        return this.trainIndices[this.shuffledTrainIndex];
      });
  }

  nextTestBatch(batchSize) {
    return this.nextBatch(batchSize, [this.testImages, this.testLabels], () => {
      this.shuffledTestIndex =
        (this.shuffledTestIndex + 1) % this.testIndices.length;
      return this.testIndices[this.shuffledTestIndex];
    });
  }

  nextBatch(batchSize, data, index) {
    const batchImagesArray = new Float32Array(batchSize * IMAGE_SIZE);
    const batchLabelsArray = new Uint8Array(batchSize * NUM_CLASSES);

    for (let i = 0; i < batchSize; i++) {
      const idx = index();

      const image =
        data[0].slice(idx * IMAGE_SIZE, idx * IMAGE_SIZE + IMAGE_SIZE);
      batchImagesArray.set(image, i * IMAGE_SIZE);

      const label =
        data[1].slice(idx * NUM_CLASSES, idx * NUM_CLASSES + NUM_CLASSES);
      batchLabelsArray.set(label, i * NUM_CLASSES);
    }

    const xs = tf.tensor2d(batchImagesArray, [batchSize, IMAGE_SIZE]);
    const labels = tf.tensor2d(batchLabelsArray, [batchSize, NUM_CLASSES]);

    return { xs, labels };
  }

  // 传递页面对象
  setPageObject(page) {
    this.page = page;
    page.setData({
      src: MNIST_IMAGES_SPRITE_PATH
    });
  }
}