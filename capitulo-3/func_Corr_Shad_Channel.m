%% Gera amostras de canal com distribuicao rayleigh lognormal, com a
%% parcela lognormal correlacionada
%
% parametros:
%
% m => numero de radios cognitivos,
% N => numero de amostra por cada radio cognitivo,
% sigma_dB => desvio padrao da variavel lognormal em dB
% b0 => variancia das gaussianas que geram a rayleigh
% L => disposicao dos radios no espaco L x L x L
% isCorr =>1, para ativar a correlacao
%          0, para gerar o canal descorrelacionado
function HLN = func_Corr_Shad_Channel(m,N,sigma_dB,mu_dB,L,Dcorr,isCorr)

if isCorr == 0
    HLN = randn(m,N).*sigma_dB+mu_dB;
else
    if L/Dcorr <= 0.01
        HLN = repmat(randn(1,N).*sigma_dB+mu_dB,m,1);
    else
        
        % gera pontos descorrelacionados no grid nxn
        n = ceil(L/Dcorr) + 1;
        
        
        % gera pontos uniformemente distribuidos no espaco LxLxL
        xi = unifrnd(0,L,1,m);
        yi = unifrnd(0,L,1,m);
        zi = unifrnd(0,L,1,m);
        
        HLN = zeros(m,N);
        
        z = randn(n,n,n,N).*sigma_dB;
        
        for i = 1: m
            
            % cordenadas normalizadas
            X = xi(i)/Dcorr;
            Y = yi(i)/Dcorr;
            Z = zi(i)/Dcorr;
            
            % ponto descorrelacionado de referencia para x, y e z
            gridx = floor(X) + 1;
            gridy = floor(Y) + 1;
            gridz = floor(Z) + 1;
            
            % correcao das cordenadas
            X = X - floor(X);
            Y = Y - floor(Y);
            Z = Z - floor(Z);
            
            % erro na estremidade do grid, correcao necessaria
            if gridx + 1 > n
                gridx = n - 1;
                X = X +1;
            end
            
            % erro na estremidade do grid, correcao necessaria
            if gridy + 1 > n
                gridy = n - 1;
                Y = Y +1;
            end
            
            % erro na estremidade do grid, correcao necessaria
            if gridz + 1 > n
                gridz = n - 1;
                Z = Z +1;
            end
            
            % pega correspondentes pontos descorrelacionados
            Sa = reshape(z(gridx,gridy,gridz,:),[1 N]);
            Sb = reshape(z(gridx + 1,gridy,gridz,:),[1 N]);
            Sc = reshape(z(gridx,gridy + 1,gridz,:),[1 N]);
            Sd = reshape(z(gridx + 1,gridy + 1,gridz,:),[1 N]);
            Se = reshape(z(gridx,gridy,gridz + 1,:),[1 N]);
            Sf = reshape(z(gridx + 1,gridy,gridz + 1,:),[1 N]);
            Sg = reshape(z(gridx,gridy + 1,gridz + 1,:),[1 N]);
            Sh = reshape(z(gridx + 1,gridy + 1,gridz + 1,:),[1 N]);
            
            % fator de normalizacao para que HG tenha sigma^2 = var
            G = sqrt((1 - 2*X + 2*(X^2))*(1 - 2*Y + 2*(Y^2))*(1 - 2*Z + 2*(Z^2)));
            HLN(i,:) = (((Sa*(1 - X) + Sb*X)*(1 - Y) + (Sc*(1 - X) + Sd*X)*Y)*(1 - Z) + ((Se*(1 - X) + Sf*X)*(1 - Y) + (Sg*(1 - X) + Sh*X)*Y)*Z)/G;
        end
        HLN = HLN + mu_dB;
    end    
end

HLN = 10.^(HLN./20); % transforma de gaussiana para lognormal
LN_phase = unifrnd(-pi,pi,[m N]); % gera fase uniforme para HLN
HLN = HLN.*(cos(LN_phase) + 1i*sin(LN_phase));

% HR = sqrt(b0)*randn(m,N) + 1i * sqrt(b0)*randn(m,N);
% H =  HLN + HR;

% d0 = (sigma_dB/(20*log10(exp(1))))^2; % conversao para escala linear
% mu = (mu_dB/(20*log10(exp(1))));

% G = 2*b0 + exp(2*mu+2*d0);
end