if __name__ == "__main__":
    import torch

    print("Torch version: ", torch.__version__)
    print("Torch cuda version: ", torch.version.cuda)
    print("Torch backends cudnn version: ", torch.backends.cudnn.version())

    if torch.cuda.is_available():
        print("CUDA works")
        exit(0)
    else:
        print("CUDA not available")
        exit(1)
