% Farinaz Fallahpour
% Date: 2012
% https://github.com/FarinazFallahpour

function SVM()
clc;clear all;close all;
dataset1 = load('hw4_dataset1');
dataset2 = load('hw4_dataset2');
[vehicle_data vehicle_label]=load_vehicle();
[heart_data heart_label]=load_heart();
[diabets_data diabets_label]=load_diabets();
[a b c d]=split_data(vehicle_data,vehicle_label,.9);
%Normalize: normalize all the samples to make their energy is 1
[accuracy] = SVM_Linear(dataset1);
data_norm = Normalize(dataset2.X);
RBF_kernel_SVM(data_norm,dataset2.y);
[scaled] = Scale(diabets_label, min(diabets_label), max(diabets_label));
RBF_kernel_SVM(diabets_data,diabets_label);
RBF_kernel_SVM(heart_data,heart_label);
vehicle_norm = Normalize((vehicle_data));
kernel_SVM_Multiclass(vehicle_data,vehicle_label);
end
function [data label]=load_vehicle()
f=fopen('xaa.dat');temp=textscan(f,'%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d  %s');fclose(f);
data=[temp{1} temp{2} temp{3} temp{4} temp{5} temp{6} temp{7} temp{8} temp{9} temp{10} temp{11} temp{12} temp{13} temp{14} temp{15} temp{16} temp{17} temp{18} ];
label=[temp{19}];
f=fopen('xab.dat');temp=textscan(f,'%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d  %s');fclose(f);
data=[data; [temp{1} temp{2} temp{3} temp{4} temp{5} temp{6} temp{7} temp{8} temp{9} temp{10} temp{11} temp{12} temp{13} temp{14} temp{15} temp{16} temp{17} temp{18} ]];
label=[label; [temp{19}]];
f=fopen('xac.dat');temp=textscan(f,'%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d  %s');fclose(f);
data=[data; [temp{1} temp{2} temp{3} temp{4} temp{5} temp{6} temp{7} temp{8} temp{9} temp{10} temp{11} temp{12} temp{13} temp{14} temp{15} temp{16} temp{17} temp{18} ]];
label=[label; [temp{19}]];
f=fopen('xad.dat');temp=textscan(f,'%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d  %s');fclose(f);
data=[data; [temp{1} temp{2} temp{3} temp{4} temp{5} temp{6} temp{7} temp{8} temp{9} temp{10} temp{11} temp{12} temp{13} temp{14} temp{15} temp{16} temp{17} temp{18} ]];
label=[label; [temp{19}]];
f=fopen('xae.dat');temp=textscan(f,'%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d  %s');fclose(f);
data=[data; [temp{1} temp{2} temp{3} temp{4} temp{5} temp{6} temp{7} temp{8} temp{9} temp{10} temp{11} temp{12} temp{13} temp{14} temp{15} temp{16} temp{17} temp{18} ]];
label=[label; [temp{19}]];
f=fopen('xaf.dat');temp=textscan(f,'%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d  %s');fclose(f);
data=[data; [temp{1} temp{2} temp{3} temp{4} temp{5} temp{6} temp{7} temp{8} temp{9} temp{10} temp{11} temp{12} temp{13} temp{14} temp{15} temp{16} temp{17} temp{18} ]];
label=[label; [temp{19}]];
f=fopen('xag.dat');temp=textscan(f,'%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d  %s');fclose(f);
data=[data; [temp{1} temp{2} temp{3} temp{4} temp{5} temp{6} temp{7} temp{8} temp{9} temp{10} temp{11} temp{12} temp{13} temp{14} temp{15} temp{16} temp{17} temp{18} ]];
label=[label; [temp{19}]];
f=fopen('xah.dat');temp=textscan(f,'%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d  %s');fclose(f);
data=[data; [temp{1} temp{2} temp{3} temp{4} temp{5} temp{6} temp{7} temp{8} temp{9} temp{10} temp{11} temp{12} temp{13} temp{14} temp{15} temp{16} temp{17} temp{18} ]];
label=[label; [temp{19}]];
f=fopen('xai.dat');temp=textscan(f,'%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d  %s');fclose(f);
data=[data; [temp{1} temp{2} temp{3} temp{4} temp{5} temp{6} temp{7} temp{8} temp{9} temp{10} temp{11} temp{12} temp{13} temp{14} temp{15} temp{16} temp{17} temp{18} ]];
label=[label; [temp{19}]];

end
function [data label]=load_heart()
ds=load('heart.dat');
data=ds(:,1:end-1);
label=ds(:,end);
end
function [data label]=load_diabets()
ds=load('diabets.dat');
data=ds(:,1:end-1);
label=ds(:,end);
end
function [test_data test_label train_data train_label]=split_data(data,label,percent)
[m n]=size(data);
indices=randperm(m);
test_data=data(1,:);test_label=label(1,:);
train_data=data(1,:);train_label=label(1,:);
for i=1:m
    if i/m<=percent
        test_data(i,:)=data(indices(i),:);
        test_label(i,:)=label(indices(i));
    else
        train_data(m-i+1,:)=data(indices(i),:);
        train_label(m-i+1,:)=label(indices(i),:);
    end

end
end
function [accu] = SVM_Linear(dataset1)
    % Normalize: normalize all the samples to make their energy is 1
    data_norm = Normalize(dataset1.X);

    % Use ten-times-ten-fold cross validation
    % SVM classifier Trainer:
    %
    %   LinearSVC
    C = [1 100];
    for i=1:size(C,2)
        [AlphaY, SVs, Bias, Parameters, nSV, nLabel] = LinearSVC(dataset1.X',dataset1.y',C(i));
        [ClassRate, DecisionValue, Ns, ConfMatrix, PreLabels]= SVMTest(dataset1.X', dataset1.y', AlphaY, SVs, Bias,Parameters, nSV, nLabel);
        res=PreLabels' - dataset1.y;
        res=abs(res);
        accu(i,1)=100-(sum(res)/size(dataset1.y,1))*100;
        disp('###################################');
        str = sprintf('%s%d','ACCURACY With C :',C(i));
        disp(str);
        disp(accu(i,1));
    end
    
    % Plot the data & Decision Boundary
    SVMPlot2(AlphaY, SVs, Bias, Parameters,dataset1.X', dataset1.y');
    
end
function RBF_kernel_SVM(dataset,label)
    % Use ten-times-ten-fold cross validation
    % SVM classifier Trainer:
    %
    %   Kernel SVM
    C=[0.01 0.04 0.1 0.4 1 4 10 40];
    Gamma=[0.01 0.04 0.1 0.4 1 4 10 40];
    accu_test=zeros(64,12);
    accu_train=zeros(64,12);
    acc_tr = zeros(size(Gamma,2),3);
    acc_te = zeros(size(Gamma,2),3);
    indices = crossvalind('Kfold',label,10);
%     [a b c d]=split_data(vehicle_data,vehicle_label,.9);
    l = 1;        
    for i=1:size(C,1)
        for j=1:size(Gamma,2)
            for k=1:10
                %test = (indices == k); train = ~test;
                [a b c d]=split_data(dataset,label,.9);
                [AlphaY, SVs, Bias, Parameters, nSV, nLabel] = u_RbfSVC(c',d', Gamma(1,j), C(1,i));
                [ClassRate, DecisionValue, Ns, ConfMatrix, PreLabels]= SVMTest(a', b', AlphaY, SVs, Bias,Parameters, nSV, nLabel);
                temp = PreLabels' - b;
                temp = abs(temp);
                accu_test(l,k) = 100-(sum(temp)/size(d,1))*100;
                accu_test(l,11) = Gamma(1,j);
                accu_test(l,12) = C(1,i);

                [ClassRate, DecisionValue, Ns, ConfMatrix, PreLabels]= SVMTest(c',d', AlphaY, SVs, Bias,Parameters, nSV, nLabel);
                temp = PreLabels' - d;
                temp=abs(temp);
                accu_train(l,k)=100-(sum(temp)/size(d,1))*100;
                accu_train(l,11)=Gamma(1,j);
                accu_train(l,12)=C(1,i);
                
            end
            acc_te(l,1)=mean(accu_test(l,1:10));
            acc_te(l,2)=Gamma(1,j);
            acc_te(l,3)=C(1,i);
            acc_tr(l,1)=mean(accu_train(l,1:10));
            acc_tr(l,2)=Gamma(1,j);
            acc_tr(l,3)=C(1,i);
            disp('###################################');
            str = sprintf('%s%d','ACCURACY Test With C :',C(i));
            disp(str);
            disp(acc_te(i,1));
            l = l + 1;
        end
    end
    % best model
    j=1;
    m=max(acc_te);
    for i=1:size(acc_te,1)
        if m(1,1)==acc_te(i,1)
            BestModel(j,:)=acc_te(i,:);
            j=j+1;
        end
    end
    
    [AlphaY, SVs, Bias, Parameters, nSV, nLabel] = u_RbfSVC(dataset', label', BestModel(1,2), BestModel(1,3));
    figure;
    SVMPlot2(AlphaY, SVs, Bias, Parameters, dataset', label');
    
    figure;
%     surf(acc_tr(:,1),acc_tr(:,2),acc_tr);
    plot(acc_tr(:,2),acc_tr(:,1),'-b');
    xlabel('Sigma');
    ylabel('Accuracy Train');
    figure;
    plot(acc_te(:,2),acc_te(:,1),'-r');
    xlabel('Sigma');
    ylabel('Accuracy Test');
    figure;
    plot(std(accu_test'),'-g');
    xlabel('for different value of C and Sigma');
    ylabel('Variances Test');
    figure;
    plot(std(accu_train'),'-c');
    xlabel('for different value of C and Sigma');
    ylabel('Variances Test');
    
end
function kernel_SVM_Multiclass(data,label)
    [~,~,labels] = unique(label);   %# labels: 1/2/3/4
   % data = zscore(dataset);        %# scale features
    [numInst m] = size(data);
    numLabels = max(labels);

    %# split training/testing
    idx = randperm(numInst);
    numTrain = 100; numTest = numInst - numTrain;
    trainData = data(idx(1:numTrain),:);  testData = data(idx(numTrain+1:end),:);
    trainLabel = labels(idx(1:numTrain)); testLabel = labels(idx(numTrain+1:end));
    [lab l_place] = sort(labels);

    trainData = cell(numLabels,1);
    trainLabel = cell(numLabels,1);
    l = 1;
    for j=1:numLabels
        [train place] = find(labels == j);
        n = size(train,1);
        trainD = zeros(n,m);
        trainL = zeros(n,1);
        for i=1:n
            
            trainD(l,1:m) = data(train(i,1),:); 
            trainL(l,1) = labels(train(i,1));
            l = l + 1;
        end
        trainData{j,1} = trainD;
        trainLabel{j,1} = trainL;
    end
    %# train one-against-all models
    C=[0.01 0.04 0.1 0.4 1 4 10 40];
    Gamma=[0.01 0.04 0.1 0.4 1 4 10 40];
    accu_test=zeros(64,12);
    accu_train=zeros(64,12);
    acc_tr = zeros(size(Gamma,2),3);
    acc_te = zeros(size(Gamma,2),3);
    model = cell(numLabels,1);
    model_test = cell(numLabels,1);
    model{1,1} = [trainData{2,1}; trainData{3,1}; trainData{4,1}];
    model{2,1} = [trainData{1,1}; trainData{3,1}; trainData{4,1}];
    model{3,1} = [trainData{1,1}; trainData{2,1}; trainData{4,1}];
    model{4,1} = [trainData{1,1}; trainData{2,1}; trainData{3,1}];
    m1 = size(model{1,1},1);
    m2 = size(model{2,1},1);
    m3 = size(model{3,1},1);
    m4 = size(model{4,1},1);
    model_test{1,1} = repmat(2,m1,1);
    model_test{2,1} = repmat(1,m2,1);
    model_test{3,1} = repmat(2,m3,1);
    model_test{4,1} = repmat(3,m4,1);
    l = 1;
    for k=1:numLabels
        for i=1:size(C,1)
            for j=1:size(Gamma,2)

                [AlphaY, SVs, Bias, Parameters, nSV, nLabel] = u_RbfSVC(model{k,1}',model_test{k,1}', Gamma(1,j), C(1,i));
                [ClassRate, DecisionValue, Ns, ConfMatrix, PreLabels]= SVMTest(trainData{k,1}', trainLabel{k,1}', AlphaY, SVs, Bias,Parameters, nSV, nLabel);
                temp = PreLabels' - trainLabel{k,1};
                temp = abs(temp);
                accu_test(l,k) = 100-(sum(temp)/size(model_test{k,1},1))*100;
                accu_test(l,11) = Gamma(1,j);
                accu_test(l,12) = C(1,i);

                [ClassRate, DecisionValue, Ns, ConfMatrix, PreLabels]= SVMTest(model{k,1}',model_test{k,1}', AlphaY, SVs, Bias,Parameters, nSV, nLabel);
                temp = PreLabels' - model_test{k,1};
                temp=abs(temp);
                accu_train(l,k)=100-(sum(temp)/size(model_test{k,1},1))*100;
                accu_train(l,11)=Gamma(1,j);
                accu_train(l,12)=C(1,i);

               % model{k} = svmtrain(double(trainLabel==k), trainData, '-c 1 -g 0.2 -b 1');
            end
            acc_te(l,1)=mean(accu_test(l,1:10));
            acc_te(l,2)=Gamma(1,j);
            acc_te(l,3)=C(1,i);
            acc_tr(l,1)=mean(accu_train(l,1:10));
            acc_tr(l,2)=Gamma(1,j);
            acc_tr(l,3)=C(1,i);
            disp('###################################');
            str = sprintf('%s%d','ACCURACY Test With C :',C(i));
            disp(str);
            disp(acc_te(i,1));
            l = l + 1;
        end
    end

    % best model
    j=1;
    m=max(acc_te);
    for i=1:size(acc_te,1)
        if m(1,1)==acc_te(i,1)
            BestModel(j,:)=acc_te(i,:);
            j=j+1;
        end
    end
    
    [AlphaY, SVs, Bias, Parameters, nSV, nLabel] = u_RbfSVC(data', label', BestModel(1,2), BestModel(1,3));
    figure;
    SVMPlot2(AlphaY, SVs, Bias, Parameters, data', label');
    
end

