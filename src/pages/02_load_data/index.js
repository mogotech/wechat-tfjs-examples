// pages/02_load_data/index.js
const helper = require('script.js');
const tf = require('@tensorflow/tfjs');

Page({

  /**
   * 页面的初始数据
   */
  data: {

  },

  /**
   * 生命周期函数--监听页面加载
   */
  onLoad: function (options) {
    async function testLoadData() {
      // 从网络加载boston house数据
      var data = await helper.getData();
      console.log("数据加载完毕");

      // 数据转换为tensor   
      const { inputs, labels, inputMax, inputMin, labelMax,labelMin } = helper.convertToTensor(data);

      // 创建模型
      const model = helper.createModel();

      // 训练模型
      await helper.trainModel(model, inputs, labels);
      console.log("模型训练完毕！");

      const inputTensor = tf.tensor2d([5], [1, 1]);
      const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
      const preds = model.predict(normalizedInputs);
      const unNormPreds = preds.mul(labelMax.sub(labelMin)).add(labelMin);
      unNormPreds.print();
    }

    testLoadData();
  },

  /**
   * 生命周期函数--监听页面初次渲染完成
   */
  onReady: function () {

  },

  /**
   * 生命周期函数--监听页面显示
   */
  onShow: function () {
  },

  /**
   * 生命周期函数--监听页面隐藏
   */
  onHide: function () {

  },

  /**
   * 生命周期函数--监听页面卸载
   */
  onUnload: function () {

  },

  /**
   * 页面相关事件处理函数--监听用户下拉动作
   */
  onPullDownRefresh: function () {

  },

  /**
   * 页面上拉触底事件的处理函数
   */
  onReachBottom: function () {

  },

  /**
   * 用户点击右上角分享
   */
  onShareAppMessage: function () {

  }
})