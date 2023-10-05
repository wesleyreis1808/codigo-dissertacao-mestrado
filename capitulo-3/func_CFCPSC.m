% U = Number of secondary users (SUs)
% M = Number of sensing samples
% snr = Signal-to-noise ratio in dB scale in each SU
% L = Number of sub bands
% sps = Samples per symbol
% mod_ord = Order of modulation
% nEvent = Number of Monte Carlo events
% nPtROC = Number of pont on ROC curve
% algorith = algorith used to sensoring
%   1 = CFCPSC
%   2 = WCFCPSC
%   3 = GLRT
% channel = Model of channel comunication
%   1 = Flat, no shadowing, equal noise
%   2 = Selective, no shadowing, equal noise
%   3 = Selective, shadowing, equal noise
%   4 = Selective, shadowing, unequal noise
% adjustThres = Adjustment in the limit (min and max) to improve the 
%   distribution of points on the curve

function [pf, pd, AUC, Thres] = func_CFCPSC(U, M, snr, L, sps, mod_ord, nEvent, nPtROC, algorith, channel, adjustThres)

    nSUs = double(U); % Number of secondary users (SUs)
    nSamples = double(M); % Number of sensing samples collected by each antena of each SU in each sensing time (st)

    %%% Impulse response
    gns = 10.^(-[-20 * log10((exp(-0/1.303))) -20 * log10((exp(-1/1.303))) -20 * log10((exp(-2/1.303))) -20 * log10((exp(-3/1.303)))] / 20); %Gains, linear scale
    ngns = gns ./ sqrt(sum(gns.^2)); %Normalized gains
    index_paths = [1 2 3 4]; npaths = numel(ngns);

    %%% 3D correlated shadowed fading sensing channel
    sigma_dB = 4; % Standard deviation of the shadowing
    L_area = 60; % 3D area of SUs in meters
    Dcorr = 30; % Decorrelation distance
    %mu_dB = -Inf;%0 or -Inf (-Inf to cancel the shadowing)

    snr = double(snr); % Signal-to-noise ratio in dB scale in each SU
    snr = db2pow(snr); % Signal-to-noise ratio in linear scale in each SU
    sigma2n = ones(1, nSUs); % Additive white Gaussian noise power in each SU

    % Spectrum sensing scenarios
    % 1 = Flat, no shadowing, equal noise
    % 2 = Selective, no shadowing, equal noise
    % 3 = Selective, shadowing, equal noise
    % 4 = Selective, shadowing, unequal noise
    SSSCenario = channel;

    if SSSCenario == 4
        sigma2n = [0.8, 0.9, 0.95, 1.1, 0.85, 1.15]; % Additive white Gaussian noise power in each SU
    end

    sigma2s = mean(sigma2n) * snr; % Received PU signal power

    nSubBands = L; % CFCPSC number of sub-bands
    nSamplesSymb = sps; % CFCPSC number of samples per symbol

    weights = repmat(((nSubBands - (1:nSubBands)) + 1) / nSubBands, nSUs, 1);

    mOrd = 2^mod_ord; % Modulation order
    avpow = mean(abs(pskmod(0:(mOrd - 1), mOrd, pi / mOrd)).^2); % Average power of mOrdPSK primary user (PU) signal
    nSymbols = nSamples / nSamplesSymb; %  Number of PU signal symbols (CFCPSC)

    nMCEvents = nEvent; % Number of Monte Carlo events
    nROCPts = nPtROC; % Number of ROC points
    TX = randi([0, 1], 1, nMCEvents); % Bernoulli random PU transmitter activity (50% in H0 and 50% in H1 hypothesis)

    Tcfcpsc = zeros(nMCEvents, nSubBands); % Pre-allocate variable

    for st = 1:nMCEvents
        % Sensing time (st) loop
        puSignal = reshape(repmat(pskmod(randi([0, (mOrd - 1)], nSymbols, 1), mOrd, pi / mOrd) ./ sqrt(avpow), [1, nSamplesSymb]).', nSamples, 1).'; % mOrdPSK PU signal
        X = TX(st) * sqrt(sigma2s) * puSignal;
        V = sqrt(diag(sigma2n) / 2) * complex(randn(nSUs, nSamples), randn(nSUs, nSamples)); % Noise samples (AWGN) in each SU

        if SSSCenario == 1
            H = sqrt(1/2) * complex(randn(nSUs, 1), randn(nSUs, 1)); % Channel samples (Nonfrequence selective Rayleigh fading)
            Y = H * X + V; % Received signal at each SU
        elseif SSSCenario == 2
            mu_dB = -Inf; %0 or -Inf (-Inf to cancel the shadowing)
            H = func_Selective_Channel(index_paths, nSUs, npaths, ngns, sigma_dB, mu_dB, L_area, Dcorr);
            Y = zeros(nSUs, nSamples);

            for u = 1:nSUs
                HX = conv(X, H(u, :));
                Y(u, :) = HX(1:nSamples) + V(u, :);
            end

        elseif SSSCenario == 3
            mu_dB = 0; %0 or -Inf (-Inf to cancel the shadowing)
            H = func_Selective_Channel(index_paths, nSUs, npaths, ngns, sigma_dB, mu_dB, L_area, Dcorr);
            Y = zeros(nSUs, nSamples);

            for u = 1:nSUs
                HX = conv(X, H(u, :));
                Y(u, :) = HX(1:nSamples) + V(u, :);
            end

        elseif SSSCenario == 4
            mu_dB = 0; %0 or -Inf (-Inf to cancel the shadowing)
            H = func_Selective_Channel(index_paths, nSUs, npaths, ngns, sigma_dB, mu_dB, L_area, Dcorr);
            Y = zeros(nSUs, nSamples);

            for u = 1:nSUs
                HX = conv(X, H(u, :));
                Y(u, :) = HX(1:nSamples) + V(u, :);
            end

        end

        %%% CFCPSC
        if algorith == 1
            Tcfcpsc(st, :) = func_CFCPSC_First6Steps(Y, nSubBands); % CFCPSC statistics per sub-band
        elseif algorith == 2
            Tcfcpsc(st, :) = func_WCFCPSC_First6Steps(Y, nSubBands, weights); % WCFCPSC statistics per sub-band
        end

    end

    %Thres_cfcpsc = linspace(min(min(Tcfcpsc(TX'==0,:))),max(max(Tcfcpsc(TX'==0,:))),nROCPts); % Thesholds
    Thres_cfcpsc = linspace(min(Tcfcpsc(:))*adjustThres(1), max(Tcfcpsc(:))*adjustThres(2), nROCPts); % Thesholds

    nH0 = sum(TX == 0); % Number of H0 events
    nH1 = nMCEvents - nH0; % Number of H1 events

    %%% CFCPSC
    subBandDec = zeros(nMCEvents,nROCPts,nSubBands); % Pre-allocate variable
    for i = 1:nSubBands
        subBandDec(:,:,i) = Tcfcpsc(:,i)>Thres_cfcpsc; % Decision per sub-band for each st and each threshold
    end
    Dec = sum(subBandDec,3)>0;

    pf_cfcpsc = sum(Dec&TX'==0,1)/nH0; % Probability of false alarm
    pd_cfcpsc = sum(Dec&TX'==1,1)/nH1; % Probability of detection
    AUC_cfcpsc = abs(trapz(pf_cfcpsc, pd_cfcpsc)); % Area under curve

    pf = pf_cfcpsc;
    pd = pd_cfcpsc;
    AUC = AUC_cfcpsc;
    Thres = Thres_cfcpsc;

end
