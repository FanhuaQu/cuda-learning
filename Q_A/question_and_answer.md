https://zhuanlan.zhihu.com/p/1920946738270810330

Q：什么是transpose kernel



Q：解决bank conflict有哪些方法？



Q：写kernel的时候，会用到哪些优化手段

A：合并访存、消除bank conflict、减少分支、Persistent Kernel、Specialized Warp、TMA trans、向量化计算



Q：flash attention 1/2/3的优化点



Q：大模型推理引擎



Q：为什么只对weight做量化，不对activate做量化？

A：权重训练好了就固定了，而activate是动态的，会随着输入样本batch、分布变化而变化。对前者量化收益大风险小，对后者量化难度高且精度损失大。还有原因就是在很多场景下只量化权重就可以满足需求了。