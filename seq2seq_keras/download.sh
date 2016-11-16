wget http://www.statmt.org/wmt10/training-giga-fren.tar
wget http://www.statmt.org/wmt15/dev-v2.tgz
tar zxvf dev-v2.tgz
tar -xvf training-giga-fren.tar 
gunzip giga-fren.release2.fixed.en.gz
gunzip giga-fren.release2.fixed.fr.gz

ln giga-fren.release2.fixed.en giga-fren.release2.en
ln giga-fren.release2.fixed.fr giga-fren.release2.fr

# ln giga-fren.release2.fixed.en wmt.en
# ln giga-fren.release2.fixed.fr wmt.fr
