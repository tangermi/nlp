
#   测试BERT中文文本分类
—————————————————————————————————————————
1.下载BERT中文预训练模型
	bert_config.json
	bert_model.ckpt.data-00000-of-00001
	bert_model.ckpt.index
	bert_model.ckpt.meta
	vocab.txt

2.下载BERT源码

3.修改run_classifier.py
	针对自己的任务数据重载 DataProcessor类
	这里添加了 class NewsProcessor(DataProcessor)
	将 NewsProcessor 添加到主函数的 processor字典
	
4.添加验证集输出指标Precision，Recall
	
5.运行 run_classifier.py
	命令行 python run_classifier.py 
				--task_name="news" 
				--do_train=False 
				--do_eval=True 
				--do_predict= False
				--data_dir="./news_data/" 
				--vocab_file="./chinese_L-12_H-768_A-12/vocab.txt" 
				--bert_config_file="./chinese_L-12_H-768_A-12/bert_config.json" 
				--init_checkpoint="./chinese_L-12_H-768_A-12/bert_model.ckpt" 
				--output_dir="./output/" 
				--train_batch_size=2 
				--eval_batch_size=2 
				--predict_batch_size=2 
				--learning_rate=1e-4 
				--num_train_epochs=3.0
				--save_checkpoints_steps=150
				
				
				
Note: 
	Dataset包含3个类别（game，fashion，houseliving）的中文新闻
		train_data 300 examples
		val_data 100 examples
		必须转为UTF-8编码
		
	


