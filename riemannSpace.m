classdef riemannSpace < handle
    %% January, 2019 Jacopo Tessadori
    %% From: https://ieeexplore.ieee.org/abstract/document/6046114
    properties
        Rmean;
        sqrtMean;
        invSqrtMean;
    end
    
    methods
        function obj=riemannSpace(relWins,varargin)
            % If only one argument is passed, relWins are expected to be
            % time windows. If a second argument reading 'cov' is passed,
            % relWins are expected to be covariance matrices, already
            if nargin==2&&strcmp(varargin{1},'cov')
                if length(size(relWins))==3
                    covMats=mat2cell(relWins,ones(size(relWins,1),1),size(relWins,2),size(relWins,3));
                    covMats=cellfun(@(x)squeeze(x),covMats,'UniformOutput',0);
                else
                    covMats{1}=relWins;
                end
            else
                % Compute covariance matrices
                if iscell(relWins)
                    covMats=cell(size(relWins));
                    for currTrial=1:length(relWins)
                        covMats{currTrial}=cov(relWins{currTrial});
                    end
                else
                    covMats=cell(size(relWins,1),1);
                    for currTrial=1:size(relWins,1)
                        covMats{currTrial}=cov(squeeze(relWins(currTrial,:,:)));
                    end
                end
            end
            
            % Compute Riemannian mean
            obj.Rmean=karcher(covMats{1:size(covMats,1)});
            
            % Compute sqrt of Rmean and inv(Rmean)
            obj.sqrtMean=obj.sqrt(obj.Rmean);
            obj.invSqrtMean=obj.sqrt(inv(obj.Rmean));
        end
        
        function Si=invMap(obj,Pi)
            Si=obj.sqrtMean*obj.log(obj.invSqrtMean*Pi*obj.invSqrtMean)*obj.sqrtMean;
        end
        
        function Pi=map(obj,Si)
            Pi=obj.sqrtMean*obj.exp(obj.invSqrtMean*Si*obj.invSqrtMean)*obj.sqrtMean;
        end
        
        function si=project(obj,Pi)
            si=triu(obj.invSqrtMean*obj.invMap(Pi)*obj.invSqrtMean);
            si=si(triu(true(size(si))));
        end
    end
    
    methods (Static)
        function expP=exp(inData)
            % Performs exp of symmetric positive-definite matrices
            [V,D]=eig(inData);
            expP=V*diag(exp(abs(diag(D))))*V';
        end
        
        function logP=log(inData)
            % Performs exp of symmetric positive-definite matrices
            [V,D]=eig(inData);
            logP=V*diag(log(abs(diag(D))))*V';
        end
        
        function sqrtP=sqrt(inData)
            % Performs (extended) square root of symmetric
            % positive-definite matrices
            [V,D]=eig(inData);
            sqrtP=V*diag(sqrt(abs(diag(D))))*V';
        end
    end
end