# Hello World
web用C#写，人脸检测用python写，检测到结果就向某一个web的某一个端口post一次（或者用rpc，没想好），直播就在另一个端口用流媒体。

检测人脸然后调用接口的机制是每帧都判断方框的个数，如果个数发生改变就截图post并且通知，截图机制是如果稳定方框数与上一个稳定状态的最后时刻间隔为2s，则判断更新的方框数是增还是减，如果是增则对屏幕进行截图。
