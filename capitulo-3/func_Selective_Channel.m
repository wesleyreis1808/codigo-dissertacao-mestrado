function H = func_Selective_Channel(index_paths, ncrs, npaths, ngns, sigma_dB, mu_dB, L, Dcorr)
    HR(:, index_paths) = sqrt(0.5) .* complex(randn(ncrs, npaths), randn(ncrs, npaths)) .* repmat(ngns, [ncrs 1]); %H Rayleigh
    HLN(:, index_paths) = repmat(func_Corr_Shad_Channel(ncrs, 1, sigma_dB, mu_dB, L, Dcorr, 1), [1 npaths]) .* exp(-1i .* unifrnd(0, 2 * pi, ncrs, npaths)); %H lognormal
    H = HLN + HR;
    d0 = (sigma_dB / (20 * log10(exp(1))))^2; %dB to linear scale
    mu = (mu_dB / (20 * log10(exp(1))));
    G = 1 + npaths * exp(2 * mu + 2 * d0);
    H = H ./ sqrt(G);
end
