const tf = require('@tensorflow/tfjs');
const fetchWechat = require('fetch-wechat');

// 从网络加载数据
async function getData() {
  const fetch = fetchWechat.fetchFunc();
  const houseDataReq = await fetch('https://ilego.club/ai/dataset/house.json');
  const houseData = await houseDataReq.json();
  const cleaned = houseData.map(house => ({
    price: house.Price,
    rooms: house.AvgAreaNumberofRooms,
  }))
    .filter(house => (house.price != null && house.rooms != null));

  return cleaned;
}

function convertToTensor(data) {
  return tf.tidy(() => {
    // Step 1\. Shuffle the data    
    tf.util.shuffle(data);
    // Step 2\. Convert data to Tensor
    const inputs = data.map(d => d.rooms)
    const labels = data.map(d => d.price);
    const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
    const labelTensor = tf.tensor2d(labels, [labels.length, 1]);
    //Step 3\. Normalize the data to the range 0 - 1 using min-max scaling
    const inputMax = inputTensor.max();
    const inputMin = inputTensor.min();
    const labelMax = labelTensor.max();
    const labelMin = labelTensor.min();
    const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
    const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));
    return {
      inputs: normalizedInputs,
      labels: normalizedLabels,
      // Return the min/max bounds so we can use them later.
      inputMax,
      inputMin,
      labelMax,
      labelMin,
    }
  });
}

function createModel() {
  // Create a sequential model
  const model = tf.sequential();

  // Add a single hidden layer
  model.add(tf.layers.dense({ inputShape: [1], units: 1, useBias: true }));

  // Add an output layer
  model.add(tf.layers.dense({ units: 1, useBias: true }));
  return model;
}

async function trainModel(model, inputs, labels) {
  // Prepare the model for training.  
  model.compile({
    optimizer: tf.train.adam(),
    loss: tf.losses.meanSquaredError,
    metrics: ['mse'],
  });

  const batchSize = 28;
  const epochs = 50;

  return await model.fit(inputs, labels, {
    batchSize,
    epochs
  });
}

module.exports = {
  getData: getData,
  convertToTensor: convertToTensor,
  createModel: createModel,
  trainModel: trainModel,
}