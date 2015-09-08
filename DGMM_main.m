
clear all
clc

tic

% load dataset.mat; %Partition Dataset for the 5 fold test
load Partition3M20150401.mat;

% rng('default');     % reset random generator.
opts.tFlag = 1;     % terminate after relative objective value does not changes much.
opts.tol = 10^-5;   % tolerance.  
opts.maxIter = 1000; % maximum iteration number of optimization.


for n=1:5
    test_FDG=Xt_FDG{1,n};
    test_VBM=Xt_VBM{1,n};
    test_AV=Xt_AV{1,n};
    test_Yt=Yt_SNP{1,n}(:,1); 

    task.DT{1}=X_FDG{1,n};
    task.DT{2}=X_VBM{1,n};
    task.DT{3}=X_AV{1,n};
    respons=Y_SNP{1,n}(:,1);
    task.target{1}=Y_SNP{1,n}(:,1); %1:rs429358
    task.target{2}=Y_SNP{1,n}(:,1); %2:rs429358
    task.target{3}=Y_SNP{1,n}(:,1); %3:rs429358
    task.lab{1}=Y{1,n};
    task.lab{2}=Y{1,n};
    task.lab{3}=Y{1,n};
    gnd=task.lab{1};

    task.num=3;

    paraset=[0.00001 0.00003 0.0001 0.0003 0.001 0.003 0.01 0.003 0.1 0.3 1 3 10];


    for j=1:length(paraset)
        opts.rho1=paraset(j);
        for k=1:length(paraset)
            opts.rho_L3=paraset(k);
            opts.init =2; % guess start point from data ZERO.
            kfold=5;
            kk=1;     
            % construct the index of cross_validation for each task.
            [tcv fcv]=f_myCV(gnd',kfold,kk); 
            %% begin to 5-fold.
            for cc=1:kfold 
                task.X = cell(task.num,1);
                task.Y = cell(task.num,1);
                for i=1:task.num                  
                    trLab=tcv{cc}';
                    % generate the task.              
                    task.X{i}=task.DT{i}(trLab,:);               
                    task.Y{i}=task.target{i}(trLab);
                    task.label{i}=task.lab{i}(trLab);
                end
                %----------Main Algorithm---------------        
                task.MC=f_lapLabelDistMatrix(task.X, task.label);
                [W, epsvalue] = f_GMTM_APG(task.X,task.Y,opts,task.MC);

                % find the selected features for each task. 
                trLab=tcv{cc}';
                teLab=fcv{cc}';
                pl=task.DT{1}(teLab,:)*W(:,1);
                pl2=task.DT{2}(teLab,:)*W(:,2);      
                pl3=task.DT{3}(teLab,:)*W(:,3);
                et(cc)=sqrt(mean((pl-respons(teLab,1)).^2))+sqrt(mean((pl2-respons(teLab,1)).^2))+sqrt(mean((pl3-respons(teLab,1)).^2));                                        
                a=respons(teLab,1)-mean(respons(teLab,1));b=pl-mean(pl);
                a2=respons(teLab,1)-mean(respons(teLab,1));b2=pl2-mean(pl2);
                a3=respons(teLab,1)-mean(respons(teLab,1));b3=pl3-mean(pl3);  
                co(cc)=abs(sum(a.*b)/sqrt(sum(a.^2)*sum(b.^2)))+abs(sum(a2.*b2)/sqrt(sum(a2.^2)*sum(b2.^2)))+abs(sum(a3.*b3)/sqrt(sum(a3.^2)*sum(b3.^2)));
            end       
            res_kfold_CO(kk)=mean(co);
            res_kfold_RMSE(kk)=mean(et);  
            res_CO(j,k)=mean(res_kfold_CO);
            res_RMSE(j,k)=mean(res_kfold_RMSE);
        end
    end
    ndim=size(res_RMSE);
    tempRMSE=10;
    tempCO=0;
    for ii=1:ndim(1)
        for jj=1:ndim(2)
    %         for k=1:ndim(3)
    %            if  res_RMSE(ii,jj)<tempRMSE
    %                tempRMSE=res_RMSE(ii,jj);
    %                paraSet=[ii,jj];
    %            end
               if  res_CO(ii,jj)>tempCO
                 tempCO=res_CO(ii,jj);
                 paraSet=[ii,jj];
               end
    %         end
        end
    end
    paraSet

    test_opts.rho1=paraset(paraSet(1));
    test_opts.rho_L3=paraset(paraSet(2));
    Final_para{n}=paraSet;
    test_opts.tFlag = 1;     % terminate after relative objective value does not changes much.
    test_opts.tol = 10^-5;   % tolerance. 
    test_opts.maxIter = 1000; % maximum iteration number of optimization.
    test_opts.init = 2;  % guess start point from data ZERO.

    newtask.MC=f_lapLabelDistMatrix(task.DT,task.lab);
    [newW, epsvalue] = f_GMTM_APG(task.DT,task.target,test_opts,newtask.MC);

    trainpl=task.DT{1}*newW(:,1);
    trainpl2=task.DT{2}*newW(:,2);
    trainpl3=task.DT{3}*newW(:,3);
    trainRMSE(n)=sqrt(mean((trainpl-task.target{1}).^2));
    trainRMSE2(n)=sqrt(mean((trainpl2-task.target{2}).^2));
    trainRMSE3(n)=sqrt(mean((trainpl3-task.target{3}).^2));
    aa=task.target{1}-mean(task.target{1});bb=trainpl-mean(trainpl);
    aa2=task.target{2}-mean(task.target{2});bb2=trainpl2-mean(trainpl2);
    aa3=task.target{3}-mean(task.target{3});bb3=trainpl3-mean(trainpl3);
    trainCO(n)=sum(aa.*bb)/sqrt(sum(aa.^2)*sum(bb.^2));
    trainCO2(n)=sum(aa2.*bb2)/sqrt(sum(aa2.^2)*sum(bb2.^2));
    trainCO3(n)=sum(aa3.*bb3)/sqrt(sum(aa3.^2)*sum(bb3.^2));


    testpl=test_FDG*newW(:,1);
    testpl2=test_VBM*newW(:,2);  
    testpl3=test_AV*newW(:,3); 
    testRMSE(n)=sqrt(mean((testpl-test_Yt).^2));
    testRMSE2(n)=sqrt(mean((testpl2-test_Yt).^2)); 
    testRMSE3(n)=sqrt(mean((testpl3-test_Yt).^2)); 

    Weight{n}=newW;

    aa=test_Yt-mean(test_Yt);bb=testpl-mean(testpl);
    aa2=test_Yt-mean(test_Yt);bb2=testpl2-mean(testpl2);
    aa3=test_Yt-mean(test_Yt);bb3=testpl3-mean(testpl3);
    testCO(n)=sum(aa.*bb)/sqrt(sum(aa.^2)*sum(bb.^2));
    testCO2(n)=sum(aa2.*bb2)/sqrt(sum(aa2.^2)*sum(bb2.^2));
    testCO3(n)=sum(aa3.*bb3)/sqrt(sum(aa3.^2)*sum(bb3.^2));

    p{n}=testpl;
    p2{n}=testpl2;
    p3{n}=testpl3;
end

toc

RMSE_FDG=[mean(testRMSE) std(testRMSE)]
RMSE_VBM=[mean(testRMSE2) std(testRMSE2)]
RMSE_AV45=[mean(testRMSE3) std(testRMSE3)]

CC_FDG=[mean(testCO) std(testCO)]
CC_VBM=[mean(testCO2) std(testCO2)]
CC_AV45=[mean(testCO3) std(testCO3)]

