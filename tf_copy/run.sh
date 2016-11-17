vocab_size=1000

data_dir=data_dir
train_dir=train_dir_$vocab_size

ipython -- translate.py --data_dir $data_dir --train_dir $train_dir --en_vocab_size=$vocab_size --fr_vocab_size=$vocab_size --quick_and_dirty --max_train_data_size=1000 --steps_per_checkpoint=4 --batch_size=4  --size=64 --num_layers=2

ipython -- translate.py --decode --data_dir $data_dir --train_dir $train_dir