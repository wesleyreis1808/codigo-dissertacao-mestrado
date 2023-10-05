clearvars
close all
clc

CFCPSC = 1;
WCFCPSC = 2;
GLRT = 3;

FLAT_NO_SHADOWING_EQUAL_NOISE = 1;
SELECTIVE_NO_SHADOWING_EQUAL_NOISE = 2;
SELECTIVE_SHADOWING_EQUAL_NOISE = 3;
SELECTIVE_SHADOWING_UNEQUAL_NOISE = 4;

%==========================================================================================================
%==========================================================================================================
figure;
grid on;
hold on;

tit = strcat('Flat, no shadowing, equal noise');
title(tit);
xlabel('Pf');
ylabel('Pd');

%[pf, pd, AUC, Thres] = func_SpectrumSensing(U, M, snr, L, sps, mod_ord, nEvent, nPtROC, algorith, channel)
[pf_cfcpsc, pd_cfcpsc, AUC_cfcpsc, Thres] = func_SpectrumSensing(6, 160, -10, 5, 4, 2, 50e3, 60, CFCPSC, FLAT_NO_SHADOWING_EQUAL_NOISE, [1 1]);
plot(pf_cfcpsc, pd_cfcpsc, 'DisplayName', strcat('CFCPSC (AUC = ', num2str(AUC_cfcpsc), ')'));

[pf_wcfcpsc, pd_wcfcpsc, AUC_wcfcpsc, Thres] = func_SpectrumSensing(6, 160, -10, 5, 4, 2, 50e3, 60, WCFCPSC, FLAT_NO_SHADOWING_EQUAL_NOISE, [1 1]);
plot(pf_wcfcpsc, pd_wcfcpsc, 'o-', 'DisplayName', strcat('WCFCPSC (AUC = ', num2str(AUC_wcfcpsc), ')'));

[pf_glrt, pd_glrt, AUC_glrt, Thres] = func_SpectrumSensing(6, 160, -10, 5, 4, 2, 50e3, 60, GLRT, FLAT_NO_SHADOWING_EQUAL_NOISE, [1 1]);
plot(pf_glrt, pd_glrt, 'o-', 'DisplayName', strcat('GLRT (AUC = ', num2str(AUC_glrt), ')'));

legend('-DynamicLegend', 'location', 'southeast'); % set legend position
saveas(gcf, strcat('images\', tit, '.png')); % save image
close;

dlmwrite(['dats\ROCsData_SSSCenario_' num2str(FLAT_NO_SHADOWING_EQUAL_NOISE) '_AUC_WCFCPSC_' num2str(AUC_wcfcpsc) '_AUC_CFCPSC_' num2str(AUC_cfcpsc) '_AUC_GLRT_' num2str(AUC_glrt) '.dat'], [pf_wcfcpsc(:),pd_wcfcpsc(:),pf_cfcpsc(:),pd_cfcpsc(:),pf_glrt(:),pd_glrt(:)], ' ')
%==========================================================================================================
%==========================================================================================================
figure;
grid on;
hold on;

tit = strcat('Selective, no shadowing, equal noise');
title(tit);
xlabel('Pf');
ylabel('Pd');

%[pf, pd, AUC, Thres] = func_SpectrumSensing(U, M, snr, L, sps, mod_ord, nEvent, nPtROC, algorith, channel)
[pf_cfcpsc, pd_cfcpsc, AUC_cfcpsc, Thres] = func_SpectrumSensing(6, 160, -10, 5, 4, 2, 50e3, 60, CFCPSC, SELECTIVE_NO_SHADOWING_EQUAL_NOISE, [1 1]);
plot(pf_cfcpsc, pd_cfcpsc, 'DisplayName', strcat('CFCPSC (AUC = ', num2str(AUC_cfcpsc), ')'));

[pf_wcfcpsc, pd_wcfcpsc, AUC_wcfcpsc, Thres] = func_SpectrumSensing(6, 160, -10, 5, 4, 2, 50e3, 60, WCFCPSC, SELECTIVE_NO_SHADOWING_EQUAL_NOISE, [1 1]);
plot(pf_wcfcpsc, pd_wcfcpsc, 'o-', 'DisplayName', strcat('WCFCPSC (AUC = ', num2str(AUC_wcfcpsc), ')'));

[pf_glrt, pd_glrt, AUC_glrt, Thres] = func_SpectrumSensing(6, 160, -10, 5, 4, 2, 50e3, 60, GLRT, SELECTIVE_NO_SHADOWING_EQUAL_NOISE, [1 1]);
plot(pf_glrt, pd_glrt, 'o-', 'DisplayName', strcat('GLRT (AUC = ', num2str(AUC_glrt), ')'));

legend('-DynamicLegend', 'location', 'southeast'); % set legend position
saveas(gcf, strcat('images\', tit, '.png')); % save image
close;

dlmwrite(['dats\ROCsData_SSSCenario_' num2str(SELECTIVE_NO_SHADOWING_EQUAL_NOISE) '_AUC_WCFCPSC_' num2str(AUC_wcfcpsc) '_AUC_CFCPSC_' num2str(AUC_cfcpsc) '_AUC_GLRT_' num2str(AUC_glrt) '.dat'], [pf_wcfcpsc(:),pd_wcfcpsc(:),pf_cfcpsc(:),pd_cfcpsc(:),pf_glrt(:),pd_glrt(:)], ' ')
%==========================================================================================================
%==========================================================================================================
figure;
grid on;
hold on;

tit = strcat('Selective, shadowing, equal noise');
title(tit);
xlabel('Pf');
ylabel('Pd');

%[pf, pd, AUC, Thres] = func_SpectrumSensing(U, M, snr, L, sps, mod_ord, nEvent, nPtROC, algorith, channel)
[pf_cfcpsc, pd_cfcpsc, AUC_cfcpsc, Thres] = func_SpectrumSensing(6, 160, -10, 5, 4, 2, 50e3, 60, CFCPSC, SELECTIVE_SHADOWING_EQUAL_NOISE, [1 1]);
plot(pf_cfcpsc, pd_cfcpsc, 'DisplayName', strcat('CFCPSC (AUC = ', num2str(AUC_cfcpsc), ')'));

[pf_wcfcpsc, pd_wcfcpsc, AUC_wcfcpsc, Thres] = func_SpectrumSensing(6, 160, -10, 5, 4, 2, 50e3, 60, WCFCPSC, SELECTIVE_SHADOWING_EQUAL_NOISE, [1 1]);
plot(pf_wcfcpsc, pd_wcfcpsc, 'o-', 'DisplayName', strcat('WCFCPSC (AUC = ', num2str(AUC_wcfcpsc), ')'));

[pf_glrt, pd_glrt, AUC_glrt, Thres] = func_SpectrumSensing(6, 160, -10, 5, 4, 2, 50e3, 60, GLRT, SELECTIVE_SHADOWING_EQUAL_NOISE, [1 1]);
plot(pf_glrt, pd_glrt, 'o-', 'DisplayName', strcat('GLRT (AUC = ', num2str(AUC_glrt), ')'));

legend('-DynamicLegend', 'location', 'southeast'); % set legend position
saveas(gcf, strcat('images\', tit, '.png')); % save image
close;

dlmwrite(['dats\ROCsData_SSSCenario_' num2str(SELECTIVE_SHADOWING_EQUAL_NOISE) '_AUC_WCFCPSC_' num2str(AUC_wcfcpsc) '_AUC_CFCPSC_' num2str(AUC_cfcpsc) '_AUC_GLRT_' num2str(AUC_glrt) '.dat'], [pf_wcfcpsc(:),pd_wcfcpsc(:),pf_cfcpsc(:),pd_cfcpsc(:),pf_glrt(:),pd_glrt(:)], ' ')
%==========================================================================================================
%==========================================================================================================
figure;
grid on;
hold on;

tit = strcat('Selective, shadowing, unequal noise');
title(tit);
xlabel('Pf');
ylabel('Pd');

%[pf, pd, AUC, Thres] = func_SpectrumSensing(U, M, snr, L, sps, mod_ord, nEvent, nPtROC, algorith, channel)
[pf_cfcpsc, pd_cfcpsc, AUC_cfcpsc, Thres] = func_SpectrumSensing(6, 160, -10, 5, 4, 2, 50e3, 60, CFCPSC, SELECTIVE_SHADOWING_UNEQUAL_NOISE, [1 1]);
plot(pf_cfcpsc, pd_cfcpsc, 'DisplayName', strcat('CFCPSC (AUC = ', num2str(AUC_cfcpsc), ')'));

[pf_wcfcpsc, pd_wcfcpsc, AUC_wcfcpsc, Thres] = func_SpectrumSensing(6, 160, -10, 5, 4, 2, 50e3, 60, WCFCPSC, SELECTIVE_SHADOWING_UNEQUAL_NOISE, [1 1]);
plot(pf_wcfcpsc, pd_wcfcpsc, 'o-', 'DisplayName', strcat('WCFCPSC (AUC = ', num2str(AUC_wcfcpsc), ')'));

[pf_glrt, pd_glrt, AUC_glrt, Thres] = func_SpectrumSensing(6, 160, -10, 5, 4, 2, 50e3, 60, GLRT, SELECTIVE_SHADOWING_UNEQUAL_NOISE, [1 1]);
plot(pf_glrt, pd_glrt, 'o-', 'DisplayName', strcat('GLRT (AUC = ', num2str(AUC_glrt), ')'));

legend('-DynamicLegend', 'location', 'southeast'); % set legend position
saveas(gcf, strcat('images\', tit, '.png')); % save image
close;

dlmwrite(['dats\ROCsData_SSSCenario_' num2str(SELECTIVE_SHADOWING_UNEQUAL_NOISE) '_AUC_WCFCPSC_' num2str(AUC_wcfcpsc) '_AUC_CFCPSC_' num2str(AUC_cfcpsc) '_AUC_GLRT_' num2str(AUC_glrt) '.dat'], [pf_wcfcpsc(:),pd_wcfcpsc(:),pf_cfcpsc(:),pd_cfcpsc(:),pf_glrt(:),pd_glrt(:)], ' ')
%==========================================================================================================
%==========================================================================================================
