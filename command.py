import fire
from music.infer import infer
from music.train import train

if __name__ == '__main__':
    fire.Fire(train)
    fire.Fire(infer)
