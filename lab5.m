clc; close all;
rec = 1;          % 0 = однокроковий прогноз, 1 = багатокроковий (рекурсивний)
Out_proc = 0;

while Out_proc < 1
    clearvars -except Out_proc rec;
    
    % 1. Введення параметрів завдання
    F = input(' Enter the predicted function F(x) [sin(x)]: ', 's');
    if isempty(F), F = 'sin(x)'; end
    F = regexprep(F, '\btg\(', 'tan(');
    F = regexprep(F, '\bctg\(', 'cot(');
    F = regexprep(F, '\bln\(', 'log(');
    
    range = input(' Enter the range of the function [0:0.1:10]: ');
    if isempty(range), range = 0:0.1:10; end
    
    Z = input(' Number of obraz or Windows size [5]: ');
    if isempty(Z), Z = 5; end
    
    eg = input(' Enter sum-squared error goal [0.0001]: ');
    if isempty(eg), eg = 0.0001; end
    
    default_stru = [Z 10 1];
    Net_stru = input(sprintf(' Enter Structure of Neural Network [%s]: ', num2str(default_stru)));
    if isempty(Net_stru), Net_stru = default_stru; end
    if Net_stru(1) ~= Z
        fprintf(' Увага: розмір вікна Z синхронізовано зі входом мережі (%d -> %d)\n', Z, Net_stru(1));
        Z = Net_stru(1);
    end
    hidden = Net_stru(2:end-1);
    
    % 2. Обчислення значень функції та формування вибірки методом вікон
    P = zeros(size(range));
    for i = 1:length(range)
        x = range(i);
        P(i) = eval(F);
    end
    
    disp(' ...forming the learning matrix... please wait...');
    in_m = zeros(length(P) - Z, Z);
    for i = 1:(length(P) - Z)
        in_m(i, :) = P(i:i+Z-1);
    end
    target_m = P(Z+1:length(P))';
    
    % 3. Налаштування та навчання багатошарового персептрона
    rng(8);
    net1 = feedforwardnet(hidden, 'trainlm');
    for k = 1:numel(hidden)
        net1.layers{k}.transferFcn = 'tansig';
    end
    net1.layers{end}.transferFcn = 'purelin';
    net1.performFcn = 'mse';
    net1.trainParam.goal = eg;
    net1.trainParam.epochs = 1000;
    net1.divideFcn = 'dividetrain';
    
    [net1, tr] = train(net1, in_m', target_m');
    view(net1);
    figure; plotperform(tr);
    
    % 4. Прогнозування на розширеному інтервалі
    disp('-------Now you can simulate a prediction -------');
    interval = input(' Enter a prediction range [0:0.1:20]: ');
    if isempty(interval), interval = 0:0.1:20; end
    
    y_fun = zeros(size(interval));
    for i = 1:length(interval)
        x = interval(i);
        y_fun(i) = eval(F);
    end
    
    in_p = zeros(length(y_fun) - Z, Z);
    for i = 1:(length(y_fun) - Z)
        in_p(i, :) = y_fun(i:i+Z-1);
    end
    targ_p = y_fun(Z+1:end);
    OUT1 = sim(net1, in_p');
    
    % Рекурсивний розрахунок за межами навчальної вибірки
    nk = length(range);
    step_ok = abs((interval(2) - interval(1)) - (range(2) - range(1))) < 1e-9;
    start_ok = abs(interval(1) - range(1)) < 1e-9;
    
    if rec == 1
        if length(interval) > nk && start_ok && step_ok
            yy = y_fun;
            for i = nk+1:length(y_fun)
                yy(i) = sim(net1, yy(i-Z:i-1)');
            end
            OUT1(nk-Z+1:end) = yy(nk+1:end);
        else
            warning('Для рекурсивного режиму interval повинен починатися так само, як range, мати однаковий крок і бути довшим за range.');
        end
    end
    
    % 5. Розрахунок похибок і побудова графіків
    err = targ_p - OUT1;
    xp  = interval(Z+1:end);
    f0  = min(nk - Z + 1, length(err));
    rmse_tr = sqrt(mean(err(1:max(1, f0-1)).^2));
    rmse_pr = sqrt(mean(err(f0:end).^2));
    
    fprintf('\nРежим прогнозування: rec = %d\n', rec);
    fprintf('MSE навчання (норм.): %.6f\n', tr.best_perf);
    fprintf('RMSE на навчальній ділянці: %.4f\n', rmse_tr);
    fprintf('RMSE прогнозу: %.4f\n', rmse_pr);
    fprintf('MAE прогнозу: %.4f, max |err|: %.4f\n', mean(abs(err(f0:end))), max(abs(err(f0:end))));
    
    disp('        x        true        pred         err');
    disp([xp(f0:10:end)' targ_p(f0:10:end)' OUT1(f0:10:end)' err(f0:10:end)']);
    
    figure;
    subplot(2, 1, 1);
    plot(xp, targ_p, '-o', xp, OUT1); grid on;
    xlabel('x'); ylabel('F(x)');
    legend('Original value', 'Predicted value');
    title(sprintf('Forecasting (rec = %d, Structure: [%s])', rec, num2str(Net_stru)));
    
    subplot(2, 1, 2);
    plot(xp, err); grid on;
    xlabel('x'); ylabel('Error');
    title(sprintf('Standard error of prediction = %.4f', rmse_pr));
    
    k = input('Do you want quit a program? [No - Enter] [Yes - Any key]: ', 's');
    Out_proc = 1;
    if isempty(k), Out_proc = 0; end
end
