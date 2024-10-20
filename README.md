这是一份关于本代码的说明。
﻿
﻿
1.本代码中TRC_train.py为主体训练代码，consistency.py为添加权重和知识蒸馏损失函数后的代码。
﻿
2.本代码依赖原版Consistency的开源代码运行，将原版代码中的consistency.py替换为本网址提供的consistency.py，并将TRC_train.py放入原版Consistency代码文件夹中。运行TRC_train.py即可开始训练模型。
﻿
3.作者需要额外新建logs_teacher文件夹存放预训练的教师模型。
﻿
4.训练的超参数可在已发表论文“Adaptive Distillation on Hard Samples Benefits Consistency for Certified Robustness”中找到。
