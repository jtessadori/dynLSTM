classdef dynFC_LSTM < handle
    % December 2019
    % Try performing classification using LSTM on connectivity matrices
    
    properties
        dataPath;
        p;
        n;
        FCfeats;
        autoencNet;
        Y;
    end
    
    methods
        function this=dynFC_LSTM(path)
            % Constructor for dynFC class.
            % Only argument (required) is absolut path of data file.
            this.dataPath=path;
        end
        
        function prepareData(this)
            % Split data according to patient
            inData=load(this.dataPath);
            this.p=size(inData.cm_all_subj_corr{1},1);
            
            % Define labels and remove unlabeled data
            y=cat(1,zeros(length(inData.idHC),1),ones(length(inData.idRR),1));
            inData.cm_all_subj_corr=inData.cm_all_subj_corr(union(inData.idHC,inData.idRR));
            this.Y=y;
            
            % Pack data
            allPdata=cat(3,inData.cm_all_subj_corr{:});
            this.n=cellfun(@(x)size(x,3),inData.cm_all_subj_corr);
            
            % Remove missing entries
            toBeRemoved=sum(diff(allPdata,[],3)==0,3)>this.p;
            toBeRemoved=sum(toBeRemoved)>this.p/2;
            allPdata(toBeRemoved,:,:)=[];
            allPdata(:,toBeRemoved,:)=[];
            this.p=size(allPdata,1);
            
            % Remove variables missing in some subjects and repack
            this.FCfeats=mat2cell(reshape(allPdata,[],size(allPdata,3)),size(allPdata,1)^2,this.n);
            
            % Normalize each subject dataset
            this.FCfeats=cellfun(@(x)normalize(x,2),this.FCfeats,'UniformOutput',false);
        end
        
        function trainAutoencoder(this)
            % Define network structure
            numInputs=size(this.FCfeats{1},1);
            numOutputs=numInputs;
            layers = [ ...
                sequenceInputLayer(numInputs)
                lstmLayer(500,'OutputMode','sequence')
                lstmLayer(this.p,'OutputMode','sequence')
                lstmLayer(500,'OutputMode','sequence')
                fullyConnectedLayer(numOutputs)
                regressionLayer];
            maxEpochs = 5000;
            
            % Define train options and train network
            miniBatchSize=1;
            options = trainingOptions('adam', ...
                'ExecutionEnvironment','gpu', ...
                'MaxEpochs',maxEpochs, ...
                'MiniBatchSize',miniBatchSize, ...
                'GradientThreshold',1, ...
                'Verbose',true, ...
                'Plots','training-progress', ... %training-progress
                'Shuffle','every-epoch', ... % every-epoch
                'ValidationData',{this.FCfeats(1:2:end),this.FCfeats(1:2:end)}, ...
                'ValidationFrequency',50, ...
                'ValidationPatience',50, ...
                'InitialLearnRate',0.001, ...
                'LearnRateSchedule','piecewise', ...
                'LearnRateDropPeriod',10, ...
                'LearnRateDropFactor',0.7);
            
            % Train network
            this.autoencNet=trainNetwork(this.FCfeats(2:2:end),this.FCfeats(2:2:end),layers,options);
        end
        
        function computeJointEigenspace(this,algorithm)
            % Compute Laplacian and Joint Eigenspace
            
            disp('Computing the Laplacian and the Joint Eigenspace');
            % These Functions are in the JoinDiagonalziation Folder
            % W set of connectivity matrices [(nxn)]x N_subjs
            this.L=ld_ComputeLaplacianGraph(this.CMdemean,'Normalized');
%             this.L=ld_ComputeLaplacianGraph(this.CMdemean,'UnNormalized');
            % L Laplacian matrices [(nxn)x N_subjs]
            % U eigenspace [(nxn)x N_subjs ]
            % E eigenvalues [(nxn)x N_subjs ]
            
            if isempty(this.V)
                if isempty(algorithm)||algorithm==1
                    % This SHOULD be an upgrade over previous joint
                    % diagonalization code. Code from: "Pierre Ablin,
                    % Jean-François Cardoso, Alexandre Gramfort. Beyond
                    % Pham’s algorithm for joint diagonalization. 2018.
                    % ffhal-01936887f"
                    insert(py.sys.path,int32(0),'D:\Code\2019_12_oldPaperJT\qndiag-master\qndiag');
                    mod=py.importlib.import_module('qndiag');
                    C=permute(this.L,[3,1,2]);
                    C(C==0)=eps;
                    C=C(1:205,:,:);
                    Ctest=mod.np.array(C);
                    B0=mod.np.array(eye(size(this.L,1)));
                    maxIter=int32(2e4);
                    tol=1e-8;
                    lambdaMin=1e-4;
                    maxTries=int32(10);
                    returnBlist=false;
                    verbose=true;
                    tempV=mod.qndiag(Ctest,B0,maxIter,tol,lambdaMin,maxTries,returnBlist,verbose);
                    tempV=cell(tempV);
                    this.V=double(tempV{1});
                else
                    Q = [];
                    for i = 1:size(this.L,3)
                        Q = cat(3,Q,this.L(:,:,i));
                    end
                    thr = 10^-8;
                    %                 thr = 1e-2;
                    % We used 10^-8. Higher thr makes the algorithm faster but
                    % the joint eigenspace is less close to the original
                    % eispaces of laplacians
                    this.V=jointD(Q,thr); % V is the Joint Eigenspace between two or more laplacians
%                 else
%                     temp=load(sprintf('%s\\V.mat',this.dataPath),'V');
%                     this.V=temp.V;
                end
            end
            disp('done');
        end
        
        function reorderJointEigenspace(this)
            % Given the joineigensapce....
            disp('Computing the Approximates Eigenvalues');
            
            LambdaTilde=zeros(size(this.V,1),size(this.L,3));
            for j = 1:size(this.L,3)
                [LambdaNew]= ld_reorderJointEigenspace_v2(this.V,this.L(:,:,j));
                LambdaTilde(:,j) = diag(LambdaNew);
                clear LambdaNew
            end
            
            ii = 0;
            step = 205;
            for i = 1:step:size(LambdaTilde,2)
                ii = ii+1;
                this.eigenLL{1,ii} = LambdaTilde(:,i:(i+step-1));
            end
            disp('done');
        end
        
        function extractFeatures(this)
            % Extracting Features from eigenvalues Timeseries
            disp('Extracting Features');
            standardDev =[];
            meanEig=[];
            for s = 1:size(this.eigenLL,2)
                tmpS = std(this.eigenLL{s},[],2);
                tmpM = mean(this.eigenLL{s},2);
                standardDev =cat(2,standardDev,tmpS);
                meanEig = cat(2,meanEig,tmpM);
            end
            standardDev=standardDev';
            meanEig = meanEig';
            this.allfeatures = [standardDev  meanEig];
%             this.allfeaturesNorm=featureNormSVM(allfeatures);
            disp('done');
        end
        
        function svmClassify(this)
            % Perform leave one out svm classification
            y = ones(13,1); 
            y(14:28,1)=-1;
            Healthy=13;
            norm =1;
            [this.all_res,Accuracy,overall] = SVMLOO_vecMS(this.allfeatures,y,Healthy,norm);
            disp('done');
        end
        
        function extractClassifierWeights(this)
            disp('Extracting Classifier weights and bootstraping');
            c = 5;
            loo_model = this.all_res{c};
            for i = 1:size(loo_model,2)
                model = loo_model(i).m;
                w(i,:) = (model.sv_coef' * full(model.SVs));
            end
            
            MeanW= mean(w,1);
            stdW = std(w,1);
            boot = MeanW./stdW;
            zboot = zscore(boot);
            figure;plot(zboot)
            
            zneg  = -1.96;
            zpos = 1.96;
            label = 1:88;
            label = label';
            label = [label;label];
            zP = label(zboot>zpos);
            zN = label(zboot<zneg);
            z = sort([zN;zP]);
            z = unique(z);
            keyboard;
        end
    end
end

