# Transformer (PyTorch)
Attention Is All You Need (Vaswani et al., 2017) implementation in PyTorch.


## Build and run with Docker

```console
$ git clone  https://github.com/ioarun/pytorch-transformer.git
$ cd pytorch-transformer
$ docker build -t pytorch-transformer .
$ docker run -it --rm --gpus all -v $(pwd):/workspace -p 8888:8888 pytorch-transformer
```

Then open your browser and go to `http://localhost:8888` to access Jupyter Notebook.