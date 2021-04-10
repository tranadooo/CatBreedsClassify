import tensorflow as tf
convert=tf.contrib.lite.TFLiteConverter.from_frozen_graph("model/frozen_model/incp_finetuned.pb",
                                                          input_arrays=["Cast"],output_arrays=["Softmax_1"],input_shapes={"Cast":[64,64,3]} )
convert.post_training_quantize=True
tflite_model=convert.convert()
open("model/frozen_model/incp_finetuned.tflite","wb").write(tflite_model)


python D:\Python\Lib\site-packages\tensorflow\python\tools\strip_unused.py ^
	--input_graph=incp_finetuned.pb ^
	--output_graph=incp_finetuned_stripped.pb ^
	--input_node_names=Cast ^
	--output_node_names=Softmax_1 ^
	--input_binary=true
