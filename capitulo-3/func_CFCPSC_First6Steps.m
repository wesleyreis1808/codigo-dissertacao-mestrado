function ravg = func_CFCPSC_First6Steps(Y,nSubBands)
% CFCPSC algorithm

dim1Size = size(Y,1)*size(Y,3); % dim1Size = nSUSs*nAnts
dim2Size = size(Y,2); % dim2Size = nSamples

Vcfcpsc = dim2Size/(2*nSubBands); % CFCPSC number of samples per sub-band

% Step 1) PSDE
Fline = abs(fft(Y,dim2Size,2)).^2/dim2Size;
% Fline = zeros(size(Y));
% for i = 1:dim1Size
%     Fline(i,:) = abs(fft(Y(i,:),dim2Size)).^2/dim2Size;
% end

% Step 2) modified circular-even component of F'u
F(:,1) = (Fline(:,1) + Fline(:,dim2Size/2+1))/2; 
F(:,2:dim2Size) = (Fline(:,2:dim2Size) + Fline(:,dim2Size-(2:dim2Size)+2))/2; 
% F = zeros(dim1Size,dim2Size);
% for i = 1:dim1Size
%     F(i,1) = (Fline(i,1) + Fline(i,dim2Size/2+1))/2;
%     for k = 2:dim2Size
%         F(i,k) = (Fline(i,k) + Fline(i,dim2Size-k+2))/2;
%     end
% end
F = F(:,1:dim2Size/2);

% Step 3) Divide the sensed band into 'nSubBands' sub-bands and compute the signal power in the l-th sub-band, ell = 1, 2, ..., nSubBands, as
Fell = zeros(dim1Size,nSubBands);
for ell = 1:nSubBands
    for k = 1:Vcfcpsc
        Fell(:,ell) = Fell(:,ell) + F(:,(ell-1)*Vcfcpsc + k);
        %for i = 1:dim1Size
            %Fell(i,ell) = Fell(i,ell) + F(i,(ell-1)*Vcfcpsc + k);
        %end
    end
end

% Step 4) Compute the total signal power in the sensed band,
Ffull = sum(F(:,1:dim2Size/2),2);
% Ffull = zeros(dim1Size,1);
% for i = 1:dim1Size
%     Ffull(i) = sum(F(i,1:dim2Size/2),2);
% end

% Step 5) Compute the average of the ratio Fell_u/Ffull_u, where the noise variance influence is canceled-out, yielding
r = Fell./(Ffull*ones(1,nSubBands));
% r = zeros(dim1Size,nSubBands);
% for i = 1:dim1Size
%     r(i,:) = Fell(i,:)./(Ffull(i));
% end

% Step 6) For both the partial and the total sample fusion strategies, the adapted CF-CPSC test statistic for the ell-th subband is formed at the FC, yielding
ravg = sum(r)/dim1Size;
% Tcfcpsc_ell = zeros(1,nSubBands);
% for i = 1:dim1Size
%     Tcfcpsc_ell(1,:) = Tcfcpsc_ell(1,:) + r(i,:)/dim1Size;
% end
% ravg = Tcfcpsc_ell;

% % Step 7) Compare the test statistics with a decision threshold to reach the decision on the occupation state of the ell-th sub-band, that is,
% decision = zeros(1,nSubBands);
% for ell = 1:nSubBands
%     if T_ell(1,ell) < Thres
%         %decide H0 for ell-th sub-band
%         decision(1,ell) = 0;
%     else
%         %decide H1 for ell-th sub-band
%         decision(1,ell) = 1;
%     end
% end
%
% % Step 8) Finally, make the global decision on the occupation of the sensed band according to
% if sum(decision) == 0
%     final_decision = 0;
% else
%     final_decision = 1;
% end
%
% dec = final_decision;
end