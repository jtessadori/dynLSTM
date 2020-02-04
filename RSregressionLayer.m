classdef RSregressionLayer < nnet.layer.RegressionLayer
        
    properties
        % (Optional) Layer properties.

        % Layer properties go here.
    end
 
    methods
        function loss = forwardLoss(layer, Y, T)
            % Return the loss between the predictions Y and the 
            % training targets T.
            %
            % Inputs:
            %         layer - Output layer
            %         Y     – Predictions made by network
            %         T     – Training targets
            %
            % Output:
            %         loss  - Loss between Y and T
            
            % Inputs are supposed to be positive-definite matrices, but
            % they have to be passed to here as a vector of inputs. Resize
            % them
            p=sqrt(size(Y,2));
            Ymat=reshape(Y,size(Y,1),p,p);
            Tmat=reshape(T,size(T,1),p,p);
            
            % Need to use Riemannian geodesic distance here
            lossVec=zeros(size(Ymat,1),1);
            for currMat=1:length(lossVec)
                P1P2=pinv(squeeze(Tmat(currMat,:,:)))*squeeze(Ymat(currMat,:,:));
                lossVec(currMat)=sqrt(sum(log(eig(P1P2)).^2));
            end
            loss=sum(lossVec);
        end
        
        function dLdY = backwardLoss(layer, Y, T)
            % Backward propagate the derivative of the loss function.
            %
            % Inputs:
            %         layer - Output layer
            %         Y     – Predictions made by network
            %         T     – Training targets
            %
            % Output:
            %         dLdY  - Derivative of the loss with respect to the predictions Y        

            % Inputs are supposed to be positive-definite matrices, but
            % they have to be passed to here as a vector of inputs. Resize
            % them
            p=sqrt(size(Y,2));
            Ymat=reshape(Y,size(Y,1),p,p);
            Tmat=reshape(T,size(T,1),p,p);
            
            % Need to use Riemannian geodesic distance here
            lossVec=zeros(size(Ymat,1),1);
            for currMat=1:length(lossVec)
                P1P2=pinv(squeeze(Tmat(currMat,:,:)))*squeeze(Ymat(currMat,:,:));
                lossVec(currMat)=sqrt(sum(log(eig(P1P2)).^2));
            end
            loss=sum(lossVec);
        end
    end
end