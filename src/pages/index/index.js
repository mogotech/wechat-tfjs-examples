//index.js


//获取应用实例
const app = getApp()

Page({
  data: {
  },
  onLoad: function () {
  },
  go01_linear_regression: function () {
    wx.navigateTo({
      url: '../01_linear_regression/index'
    })
  },
  go02_load_data: function () {
    wx.navigateTo({
      url: '../02_load_data/index'
    })
  }
})