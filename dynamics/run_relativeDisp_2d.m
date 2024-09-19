af = '65';
path = strcat('/media/hdd2/softness/P2-Entropy_2d/pos_binary/0.',af,'/');

dia=3.34;
%
big_x = load([path 'big_af-',af,'_x.dat'])/dia;
big_y = load([path 'big_af-',af,'_y.dat'])/dia;
small_x = load([path 'small_af-',af,'_x.dat'])/dia;
small_y = load([path 'small_af-',af,'_y.dat'])/dia;
%}

compute_relativeDisp_2d(small_x,big_x,small_y,big_y)
