# 项目踩坑总结--MNIST机器学习入门

### 问题：从tensorflow.examples.tutorials.mnist引入模块时发生错误。
**解决方法**
首先进入到自己的python路径下（不知道的win+R -> cmd，命令行下输入：where python），进入Lib\site-packages文件夹下找tensorflow，很多网上的解决方法都要要我们去找tensorflow_core，但是我发现我的路径下并没有这个文件夹。大概时因为tensorflow版本不同的原因，导致目录结构又所改变，所以我们直接进入tensorflow文件夹里面，然后再进入core里面。然后再进入example里面，然后你会发现里面确实没有所谓的tutorials文件夹，所以直接从网上下载该文件夹粘贴进example中。
下载语句要做对应修改：
`from tensorflow.core.example.tutorials.mnist import input_data`
---
