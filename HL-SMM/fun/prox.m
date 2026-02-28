function [z] = prox(s,alpha)  
     for i=1:length(s)
         if 0< s(i) && s(i) <= sqrt(2*alpha)
             z(i)=0;
         else
             z(i) = s(i);
         end  
     end
     z = z';
end
