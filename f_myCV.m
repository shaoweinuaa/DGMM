function [tcv fcv]=f_myCV(gnd,kfold,krand)

% Startified k-fold CV partition
% [fcv tcv]=myCV(gnd,kfold,krand)
% Inputs
%  gnd   - class labels
%  kflod - k folds
%  krand - seed for random number
% 
% Outputs
%  tcv - training fold (kfold-1)
%  fcv - test fold (1)

c=length(unique(gnd));
scv=cell(c,kfold);
for i=1:c,
    t=find(gnd==i);
    rand('state',krand);
    rp=randperm(length(t));
    t=t(rp);
    a=fix(length(t)/kfold);
    for j=1:kfold-1
        scv{i,j}=t((j-1)*a+1:j*a);
    end
    scv{i,kfold}=t((kfold-1)*a+1:length(t));
end
fcv=cell(kfold,1);
tcv=cell(kfold,1);
for k=1:kfold,
    for i=1:c,
        fcv{k}=[fcv{k} scv{i,k}];
        t=[];
        for j=1:kfold,
            if k~=j
                t=[t scv{i,j}];
            end
        end
        tcv{k}=[tcv{k} t];
    end
end