## Training

cd src
mkdir build
cmake -DCMAKE_PREFIX_PATH=/home/liboning/anaconda3/lib/python3.9/site-packages/torch ..
make
cd ..
cp build/CFR CFR
cd ..
python train/actor_train.py
