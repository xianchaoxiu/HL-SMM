%%%%%%%%%%% 生成算子A的伴随
function [AT] = adjoint (Y,X,d)
    AT=0;
    for i=1:length(d)
        AT = AT+Y(i)*X(i,:,:,:)*d(i);
    end
end
 


