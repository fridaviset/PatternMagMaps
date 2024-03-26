function PhiSE3D=Phi3D(m1,m2,m3,x,lL1,uL1,lL2,uL2,lL3,uL3)

Phi1=Phi(x(1,:),m1,uL1,lL1);
Phi2=Phi(x(2,:),m2,uL2,lL2);
Phi3=Phi(x(3,:),m3,uL3,lL3);
N=size(x,2);
M=m1*m2*m3;
PhiSE3D=zeros(N,M);
for t=1:N
    PhiSE2D=kron(Phi1(t,:),Phi2(t,:));
    PhiSE3D(t,:)=kron(PhiSE2D,Phi3(t,:));
end


end

function phi=Phi(x,m,uL,lL)
%Basis functions for the SE kernel function

N=size(x,2);
phi=zeros(N,m);
for i=1:N
    for j=1:m
        phi(i,j)=1/sqrt(0.5*(uL-lL))*sin(pi*j*(x(i)-lL)/(uL-lL));
    end
end

end