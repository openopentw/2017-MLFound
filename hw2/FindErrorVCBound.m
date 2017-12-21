function output = FindErrorVCBound(N, VC, Dow, Err)
    Ori(N, VC, Dow)
    Rad(N, VC, Dow)
    Par(N, VC, Dow, Err)
    dev = Dev(N, VC, Dow, Err)
    var = Var(N, VC, Dow, Err)
    function ori = Ori(N, VC, Dow)
        ori = sqrt((8/N) * log((4*(2*N)^VC) / (Dow)));
    end
    function rad = Rad(N, VC, Dow)
       rad = sqrt((2*log(2*(N)^VC)) / (N)) + sqrt((2/N)*log(1/Dow)) + 1/N;
    end
    function par = Par(N, VC, Dow, Err)
       par = sqrt((1/N)*(2*Err + log((6*(2*N)^VC)/(Dow)))); 
    end
    function dev = Dev(N, VC, Dow, Err)
       dev = sqrt((1/(2*N)) * (4*Err*(1+Err) + log(4/(Dow))+VC*log(N^2))); 
    end
    function var = Var(N, VC, Dow, Err)
        var = sqrt((16/N) * log((2*(N)^VC)/(sqrt(Dow))));
    end
end