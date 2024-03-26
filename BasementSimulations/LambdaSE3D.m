function LamSE3D=LambdaSE3D(m1,m2,m3,sigma_SE,l_SE,lL1,uL1,lL2,uL2,lL3,uL3)

Lambda1=Lambda(m1,sqrt(sigma_SE),l_SE,uL1,lL1);
Lambda2=Lambda(m2,sqrt(sigma_SE),l_SE,uL2,lL2);
Lambda3=Lambda(m3,sqrt(sigma_SE),l_SE,uL3,lL3);
LamSE2D=kron(Lambda1,Lambda2);
LamSE3D=kron(LamSE2D,Lambda3);

end


function Lambda=Lambda(m,sigma_SE,l_SE,uL,lL)
%Eigenvalues for the 1D squared exponential basis functions
Lambda=ones(m,1);
for j=1:m
    eigv=(pi*j/(uL-lL)).^2;
    Lambda(j)=spectral_density_SE(sqrt(eigv),sigma_SE,l_SE);
end
end

