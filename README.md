This is a program where pong can be played, the players can be controlled by humans or neural networks. The architecture for the neural networks is based on Deep Q Learning.

# How to run

Numpy needs to be installed

## Training

```zsh
python3 train.py [load] [model]
```

Training plays 10,000 points and saves every 10 points, so you can stop it early.

- `[load]` is `1` to keep training an existing model, and the file must already exist. With anything else, it starts a new network and replaces any model saved in `[model]`. The default is a new network.
- `[model]` is the file to load from and save to. The default is `model.json`. To choose a model, you also have to give `[load]`.

Examples

```zsh
python3 train.py                # new model, saved to model.json
python3 train.py 1              # keep training model.json
python3 train.py 0 new.json     # new model, saved to new.json
python3 train.py 1 old.json     # keep training old.json
```

## Playing

```zsh
python3 ui.py [players] [model1] [model2]
```

A game lasts 60 seconds. The left paddle moves with `W`/`S` and the right paddle with `I`/`K`.

- `[players]` is two digits, the left paddle first. `1` is a human and `0` is a network. The default is `11`, and anything that isn't two characters long is ignored.
- `[model1]` is the left paddle's model. The default is `model.json`.
- `[model2]` is the right paddle's model. The default is `model.json`.

Both models are always loaded, even for a human paddle, so both files must exist. To choose a model, you also have to give `[players]`.

Examples

```zsh
python3 ui.py                             # two humans
python3 ui.py 10                          # you on the left, model.json on the right
python3 ui.py 01                          # model.json on the left, you on the right
python3 ui.py 00                          # model.json against itself
python3 ui.py 00 first.json second.json   # first.json against second.json
python3 ui.py 10 model.json second.json   # you against second.json
```
