// pages/03_mnist/index.js
const tf = require('@tensorflow/tfjs');

const dataHelper = require('data.js');
const modelHelper = require('script.js');

const mnist = new dataHelper.MnistData();

Page({

  /**
   * 页面的初始数据
   */
  data: {

  },

  imageError: function (e) {
    console.log('加载图片出错，原因：', e.detail.errMsg)
  },

  /**
   * 生命周期函数--监听页面加载
   */
  onLoad: function (options) {
    mnist.setPageObject(this);
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

  },

  onImageLoaded: async function (e) {
    console.log("image loaded!" + e.detail);
    await mnist.load("mnistCanvas", e.detail.width, e.detail.height);

    const model = modelHelper.getModel();

    console.log("training ...");
    await modelHelper.train(model, mnist);
    console.log("train successful!");
    console.log("predicting ...");
    const [preds, labels] = await modelHelper.doPrediction(model, mnist);
    console.log("after predict!");
    const predsArray = preds.dataSync();
    const labelsArray = labels.dataSync();
    console.log('preds:', predsArray);
    console.log('labels:', labelsArray);
    var n = 0;
    for (var i = 0; i < predsArray.length; i++) {
      console.log(predsArray[i]);
      console.log(labelsArray[i]);
      if (predsArray[i] == labelsArray[i])
        n++;
    }
    const accuracy = n / predsArray.length;
    console.log(accuracy);
  }
})