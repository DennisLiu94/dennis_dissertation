	实体关系分类任务通常被视作一个分类任务，在过去的研究中研究者们提出了各种分类特征。如在2010年的评测任务当时表现最好的系统的作者设计了Lexical（词法），Dependency（依存关系）等8组共45个特征。[2]更早的研究表明，在判断实体关系的任务中，句法特征的加入对于提升系统表现是非常重要的。事实上，大量有效信息集中在两个目标实体的最短依存路径上。[3]
	在NLP技术进入深度学习时代后，很多关于实体关系分类任务的研究中，研究者们都把整个句子的原始单词序列或者整个句子的句法树作为神经网络模型的系统输。如国内学者曾道建等的CDNN模型[4],国外学者C´ıcero Nogueira dos Santos等人的CR-CNN模型等。这种做法在需要判断关系的两个实体距离较远时，会收到较多无关信息的影响而表现下降。[1]