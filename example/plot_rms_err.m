%% This is the script to produce Fig. 2, 3, 4 in the paper.
close all;
clear all;
clc;

rmse = [0.42235126,0.41767446,0.40983385,0.39966678,0.39556111,0.38598635,0.37936417,0.37313032,0.36527429,0.36278003,0.35315544; ...
    0.32118076,0.31151669,0.2969377,0.28940455,0.27887382,0.26694434,0.26519972,0.2526769,0.24119928,0.23280941,0.23286781; ...
    0.23900282,0.22704481,0.21123443,0.20231016,0.19413591,0.18571777,0.18242978,0.17662149,0.18039344,0.17767866,0.18585065;  ...
    0.17518762,0.16327951,0.15225699,0.15147964,0.14414004,0.14188481,0.1430734,0.14670445,0.15513718,0.15813118,0.17832605;  ...
    0.13629329,0.12628484,0.11652005,0.11882943,0.11538459,0.12057396,0.12960163,0.13850856,0.15647623,0.16714719,0.18238656;  ...
    0.10076494,0.09402047,0.09511651,0.10082666,0.10525407,0.1182679,0.12264567,0.14284654,0.15321576,0.17348804,0.18770906;  ...
    0.07840269,0.07698277,0.08024929,0.08963198,0.09655355,0.1146941,0.12992416,0.15194244,0.16082,0.17595922,0.18921629;  ...
    0.06025655,0.06340545,0.07232947,0.08539853,0.09999081,0.11563396,0.13127047,0.14679153,0.16014966,0.17788652,0.19284637;  ...
    0.04685843,0.05611616,0.06876763,0.08533902,0.10302899,0.11531832,0.12894487,0.14825763,0.16116295,0.1781054,0.19763312;  ...
    0.03727294,0.05459861,0.06858265,0.08681023,0.10259839,0.11842943,0.13233816,0.14617259,0.16610292,0.17888917,0.19590398];

decay = [0.36547867 0.24896134 0.17447318 0.12951073 0.09855018 0.07940553 0.06871649 0.0544811 0.04458115 0.03504365];

[eps, sigs] = size(rmse);
decs = size(decay, 2);

rmseint = sum(rmse, 1) * 5;

n_eps = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50];
p_sig = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
decays = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1];

best_sig = zeros(eps, 1);
for i = 1:eps
    best_sig(i) = p_sig(rmse(i, :) == min(rmse(i, :)));
end

h = figure; hold on;
for i = 1:numel(n_eps)
    plot(p_sig, rmse(i, :), 'color', [1-i/eps,i/eps,0])
end
xlabel('P(\sigma=1)');
ylabel('RMS Error');
legend('5 episodes', '10 episodes', '15 episodes', '20 episodes', '25 episodes', '30 episodes', '35 episodes', '40 episodes', '45 episodes', '50 episodes', 'location', 'northwest')
title('RMS Error vs P(\sigma=1) at Interim Episodes');
fig = gcf;
fig.PaperPositionMode = 'auto';
fig_pos = fig.PaperPosition;
fig.PaperSize = [fig_pos(3) fig_pos(4)];
print(h, 'rmseP.pdf', '-dpdf', '-r600')

h = figure; hold on;
for i = 1:numel(p_sig)
    plot(n_eps, rmse(:, i), 'color', [1-i/sigs,i/sigs,0]);
end
xlabel('Episodes');
ylabel('RMS Error');
legend('P(\sigma=1)=0.0', 'P(\sigma=1)=0.1', 'P(\sigma=1)=0.2', 'P(\sigma=1)=0.3', 'P(\sigma=1)=0.4', 'P(\sigma=1)=0.5', 'P(\sigma=1)=0.6', 'P(\sigma=1)=0.7', 'P(\sigma=1)=0.8', 'P(\sigma=1)=0.9', 'P(\sigma=1)=1.0')
title('RMS Error vs Episodes for Various P(\sigma=1)');
fig = gcf;
fig.PaperPositionMode = 'auto';
fig_pos = fig.PaperPosition;
fig.PaperSize = [fig_pos(3) fig_pos(4)];
print(h, 'rmseep.pdf', '-dpdf', '-r600')

h = figure; hold on;
for i = 1:numel(p_sig)
    plot(n_eps, rmse(:, i), 'color', [1-i/sigs,i/sigs,0]);
end
plot(n_eps, decay, 'color', 'b');
xlabel('Episodes');
ylabel('RMS Error');
legend('P(\sigma=1)=0.0', 'P(\sigma=1)=0.1', 'P(\sigma=1)=0.2', 'P(\sigma=1)=0.3', 'P(\sigma=1)=0.4', 'P(\sigma=1)=0.5', 'P(\sigma=1)=0.6', 'P(\sigma=1)=0.7', 'P(\sigma=1)=0.8', 'P(\sigma=1)=0.9', 'P(\sigma=1)=1.0', '\beta=0.9')
title('Comparing Fixed vs Decaying P(\sigma=1)');
fig = gcf;
fig.PaperPositionMode = 'auto';
fig_pos = fig.PaperPosition;
fig.PaperSize = [fig_pos(3) fig_pos(4)];
print(h, 'rmsedecay.pdf', '-dpdf', '-r600')

h = figure; hold on;
axis([0 55 -0.05, 1.05]);
plot(n_eps, best_sig, 'b.-', 'markersize', 10);
xlabel('Episodes');
ylabel('P(\sigma=1)');
title('Best Setting of P(\sigma=1) vs. Elapsed Episodes');
fig = gcf;
fig.PaperPositionMode = 'auto';
fig_pos = fig.PaperPosition;
fig.PaperSize = [fig_pos(3) fig_pos(4)];
print(h, 'bestsig.pdf', '-dpdf', '-r600')

h = figure; hold on;
plot(p_sig, rmseint, 'ko-');
plot([p_sig(1) p_sig(end)], [sum(decay) sum(decay)] * 5, 'b');
for i = 1:numel(p_sig)
    plot(p_sig(i), rmseint(i), 'o', 'markeredgecolor', 'k', 'markerfacecolor', [1-i/sigs,i/sigs,0]);
end
xlabel('P(\sigma=1)');
ylabel('Cumulative RMS Error');
title('Integral of RMS Error for First 50 Episodes');
legend('Fixed P(\sigma=1)', '\beta=0.9', 'location', 'northwest');
fig = gcf;
fig.PaperPositionMode = 'auto';
fig_pos = fig.PaperPosition;
fig.PaperSize = [fig_pos(3) fig_pos(4)];
print(h, 'rmseint.pdf', '-dpdf', '-r600')
