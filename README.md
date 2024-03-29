## Requirements

libtorch

cmake

## Training

cd src & mkdir build & cd build

cmake -DCMAKE_PREFIX_PATH=[LIBTORCH_DIR] ..

cd .. & cd ..

python train/actor_train.py

## Evaluation

ABSTRACTION_TYPE = 0 --> DEFAULT ACTION ABSTRACTION

ABSTRACTION_TYPE = 1 --> RL-CFR ACTION ABSTRACTION

### Abstraction Evaluation

./src/build/CFR -1 [EVALUATION_NUMBER]

### Exploxity Evaluation

./src/build/CFR -2 [ABSTRACTION_TYPE] [ABSTRACTION_TYPE] [EVALUATION_NUMBER]

### Heads-up Evaluation

./src/build/CFR -3 [THREAD_ID] [ABSTRACTION_TYPE] [ABSTRACTION_TYPE] [EVALUATION_NUMBER]

OR

python train/AI_test.py