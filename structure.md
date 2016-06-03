###毕业论文结构提纲


#####绪论
	课题背景介绍

	实体关系分类任务的内容与价值
		实体分类任务的介绍
		数据集介绍
		任务本身的意义
		模型转作他用的可能性
	attention model的介绍
		attention model的思想源流
		attention model的主要methodology
		attention model解决的问题

	本文的主要工作
		根据已发表的一些工作，在最短依存路径工作的基础上进行了一些实验。
		受到SDP思路的启发，结合Deepmind的文章提出了新的决策链+attention模型。
		模型融合实验

#####第一章 基线系统介绍
	LSTM模型介绍
	GLOVE介绍
	基线系统介绍
	benchmark
	错误分析

#####第二章 最短依存路径上的一系列实验
	
	依存句法介绍
	standford parser介绍
	SDP介绍
	三个模型介绍
		SDP baseline（with/without deprel）
		extended sdp （with/without deprel）
		deep sdp
	benchmark比较，速度比较，错误分析（optional）

#####第三章 attention model
	介绍attention model
	介绍deepmind的工作
	介绍改进版模型
	benchmark
	错误分析

#####第四章 融合系统
	融合的介绍
	简单融合
	复杂融合
		三种融合模型
	benchmark
	和state of art的表现比较
	错误分析


#####结论
