import os

f = open('/content/drive/My Drive/monodepth2/splits/7scenes/train_files.txt','w')
for i in range(0,980):
  f.write('chess/seq-01 {}\n'.format(int(i)))
  f.write('chess/seq-02 {}\n'.format(int(i)))
  f.write('chess/seq-04 {}\n'.format(int(i)))
  f.write('chess/seq-06 {}\n'.format(int(i)))

for i in range(0,980):
  f.write('fire/seq-03 {}\n'.format(int(i)))
  f.write('fire/seq-04 {}\n'.format(int(i)))

for i in range(0,980):
  f.write('heads/seq-02 {}\n'.format(int(i)))

    
f.close()

f = open('/content/drive/My Drive/monodepth2/splits/7scenes/val_files.txt','w')
for i in range(0,980):
  f.write('chess/seq-03 {}\n'.format(int(i)))
  f.write('chess/seq-05 {}\n'.format(int(i)))

for i in range(0,980):
  f.write('fire/seq-01 {}\n'.format(int(i)))
  f.write('fire/seq-02 {}\n'.format(int(i)))
  
for i in range(0,980):
  f.write('heads/seq-01 {}\n'.format(int(i)))

f.close()
