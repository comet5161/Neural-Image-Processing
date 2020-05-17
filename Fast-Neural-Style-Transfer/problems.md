- ImportError: libgthread-2.0.so.0: cannot open shared object file: No such file or directory
- ---
- ImportError: libSM.so.6: cannot open shared object file: No such file or directory

---
- UnknownError:  Failed to get convolution algorithm. This is probably because cuDNN failed to initialize, so try looking to see if a warning log message was printed above.
	 [[node model/block1_conv1/Conv2D (defined at :31) ]] [Op:__inference_distributed_function_669]

解决：
  可能是显存不够，tf2设置
  gpu按需使用:
  ```python
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
```

- Failed to import pydot. You must install pydot and graphviz for `pydotprint` to work.

解决：
pip install pydot
pip install graphviz
sudo apt install graphviz