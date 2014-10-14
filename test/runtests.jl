using RPCA
using Base.Test

D = [0.462911    0.365901  0.00204357    0.692873    0.935861;
    0.0446199    0.108606   0.0664309   0.0736707    0.264429;
     0.320581    0.287788    0.073133    0.188872    0.526404;
     0.356266    0.197536 0.000718338    0.513795    0.370094;
     0.677814    0.011651    0.818047   0.0457694    0.471477]

A = [0.462911   0.365901  0.00204356   0.345428   0.623104;
     0.0446199  0.108606  0.0429271    0.0736707  0.183814;
     0.320581   0.203777  0.073133     0.188872   0.472217;
     0.30725    0.197536  0.000717701  0.201626   0.370094;
     0.234245   0.011651  0.103622     0.0457694  0.279032]

E = [0.0        0.0        0.0        0.347445  0.312757 ;
     0.0        0.0        0.0235038  0.0       0.0806151;
     0.0        0.0840109  0.0        0.0       0.0541868;
     0.0490157  0.0        6.5061e-7  0.312169  0.0      ;
     0.443569   0.0        0.714425   0.0       0.192445]

Â, Ê = inexact_alm_rpca(D, nonnegativeE=true, nonnegativeA=true)
 
@test_approx_eq_eps Â A 1.0e-6
@test_approx_eq_eps Ê E 1.0e-6
# reconstruction eror
@test norm(D - (Â + Ê))/norm(D) < 1.0e-7
