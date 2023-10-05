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

function [pf, pd, AUC, Thres] = func_SpectrumSensing(U, M, snr, L, sps, mod_ord, nEvent, nPtROC, algorith, channel, adjustThres)

    if algorith == 1 || algorith == 2
        [pf, pd, AUC, Thres] = func_CFCPSC(U, M, snr, L, sps, mod_ord, nEvent, nPtROC, algorith, channel, adjustThres);
    elseif algorith == 3
        [pf, pd, AUC, Thres] = func_GLRT(U, M, snr, sps, mod_ord, nEvent, nPtROC, channel, adjustThres);
    end

end