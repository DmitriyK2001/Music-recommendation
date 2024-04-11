import fire
from music.infer import infer
from music.train import train

def main():
    fire.Fire({"infer": infer, "train": train,})

if __name__ == "__main__":
    main()