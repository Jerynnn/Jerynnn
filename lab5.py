clc; close all;  % Чистимо командне вікно і закриваємо старі графіки
rec = 1;          % Режим прогнозу: 0 = по реальних даних, 1 = багатокроковий (по своїх прогнозах)
Out_proc = 0;     % Прапорець циклу: 0 — крутимо далі, 1 — виходимо з програми

while Out_proc < 1
    % Видаляємо всі змінні з пам'яті, крім керуючих прапорців циклу
    clearvars -except Out_proc rec;
    
    % --- 1. Введення параметрів завдання ---
    F = input(' Enter the predicted function F(x) [sin(x)]: ', 's');
    if isempty(F), F = 'sin(x)'; end  % Якщо натиснули Enter, беремо sin(x)
    
    % Переводимо звичний математичний запис функцій у формат, який розуміє MATLAB
    F = regexprep(F, '\btg\(', 'tan(');
    F = regexprep(F, '\bctg\(', 'cot(');
    F = regexprep(F, '\bln\(', 'log(');
    
    % Зчитуємо відрізок, на якому мережа буде вчитися
    range = input(' Enter the range of the function [0:0.1:10]: ');
    if isempty(range), range = 0:0.1:10; end
    
    % Зчитуємо розмір ковзного вікна Z (скільки попередніх точок подаємо на вхід)
    Z = input(' Number of obraz or Windows size [5]: ');
    if isempty(Z), Z = 5; end
    
    % Зчитуємо потрібну точність навчання (похибку, до якої треба дійти)
    eg = input(' Enter sum-squared error goal [0.0001]: ');
    if isempty(eg), eg = 0.0001; end
    
    % Задаємо структуру мережі: [входи приховані_шари вихід]
    default_stru = [Z 10 1];
    Net_stru = input(sprintf(' Enter Structure of Neural Network [%s]: ', num2str(default_stru)));
    if isempty(Net_stru), Net_stru = default_stru; end
    
    % Якщо у структурі вказали іншу кількість входів, підганяємо розмір вікна Z під неї
    if Net_stru(1) ~= Z
        fprintf(' Увага: розмір вікна Z синхронізовано зі входом мережі (%d -> %d)\n', Z, Net_stru(1));
        Z = Net_stru(1);
    end
    hidden = Net_stru(2:end-1);  % Забираємо тільки розміри прихованих шарів
    
    % --- 2. Підготовка даних для навчання ---
    P = zeros(size(range));      % Створюємо порожній масив під значення функції
    for i = 1:length(range)
        x = range(i);            % Беремо поточний x
        P(i) = eval(F);          % Рахуємо значення формули F(x)
    end
    
    % Нарізаємо числовий ряд на приклади методом ковзного вікна
    disp(' ...forming the learning matrix... please wait...');
    in_m = zeros(length(P) - Z, Z);
    for i = 1:(length(P) - Z)
        in_m(i, :) = P(i:i+Z-1); % Вікно із Z точок поспіль — це вхід
    end
    target_m = P(Z+1:length(P))'; % Наступна (Z+1)-ша точка — це правильна відповідь для мережі
    
    % --- 3. Створення і навчання нейронної мережі ---
    rng(8);  % Фіксуємо початкові випадкові ваги, щоб результати не стрибали між запусками
    net1 = feedforwardnet(hidden, 'trainlm');  % Створюємо багатошаровий персептрон
    
    % На приховані шари ставимо нелінійну функцію активації tansig
    for k = 1:numel(hidden)
        net1.layers{k}.transferFcn = 'tansig';
    end
    net1.layers{end}.transferFcn = 'purelin';  % На вихід ставимо лінійну функцію без обмежень
    
    net1.performFcn = 'mse';           % Оцінюємо якість по середньоквадратичній похибці
    net1.trainParam.goal = eg;         % Зупиняємось, коли помилка впаде нижче eg
    net1.trainParam.epochs = 1000;     % Максимум 1000 проходів (епох) по вибірці
    net1.divideFcn = 'dividetrain';    % Віддаємо всі 100% даних на навчання, щоб не губити точки біля розривів
    
    % Запускаємо тренування мережі
    [net1, tr] = train(net1, in_m', target_m');
    view(net1);                        % Малюємо структурну схему мережі
    figure; plotperform(tr);           % Малюємо графік падіння похибки під час навчання
    
    % --- 4. Прогнозування на ширшому інтервалі ---
    disp('-------Now you can simulate a prediction -------');
    interval = input(' Enter a prediction range [0:0.1:20]: ');
    if isempty(interval), interval = 0:0.1:20; end
    
    % Рахуємо реальні значення функції на всьому відрізку, щоб потім було з чим порівняти
    y_fun = zeros(size(interval));
    for i = 1:length(interval)
        x = interval(i);
        y_fun(i) = eval(F);
    end
    
    % Формуємо вхідні вікна по всьому інтервалу
    in_p = zeros(length(y_fun) - Z, Z);
    for i = 1:(length(y_fun) - Z)
        in_p(i, :) = y_fun(i:i+Z-1);
    end
    targ_p = y_fun(Z+1:end);           % Реальні відповіді для перевірки
    OUT1 = sim(net1, in_p');           % Проганяємо мережу в однокроковому режимі
    
    % Перевіряємо, чи збігається початок і крок відрізків для рекурсивного прогнозу
    nk = length(range);
    step_ok = abs((interval(2) - interval(1)) - (range(2) - range(1))) < 1e-9;
    start_ok = abs(interval(1) - range(1)) < 1e-9;
    
    if rec == 1
        if length(interval) > nk && start_ok && step_ok
            yy = y_fun;  % Робимо копію ряду, куди будемо дописувати власні прогнози мережі
            for i = nk+1:length(y_fun)
                % Подаємо у вікно останні Z точок (включно з тими, що мережа вже сама передбачила)
                yy(i) = sim(net1, yy(i-Z:i-1)');
            end
            % Замінюємо прогноз за межами навчання на результати багатокрокового розрахунку
            OUT1(nk-Z+1:end) = yy(nk+1:end);
        else
            warning('Для рекурсивного режиму interval повинен починатися так само, як range, мати однаковий крок і бути довшим за range.');
        end
    end
    
    % --- 5. Підрахунок помилок і вивід графіків ---
    err = targ_p - OUT1;               % Різниця між реальним значенням і прогнозом у кожній точці
    xp  = interval(Z+1:end);           % Координати x для побудови графіків
    f0  = min(nk - Z + 1, length(err)); % Точка, з якої починається чистий прогноз
    
    % Рахуємо середньоквадратичну (RMSE) та середню абсолютну (MAE) похибки
    rmse_tr = sqrt(mean(err(1:max(1, f0-1)).^2));
    rmse_pr = sqrt(mean(err(f0:end).^2));
    
    fprintf('\nРежим прогнозування: rec = %d\n', rec);
    fprintf('MSE навчання (норм.): %.6f\n', tr.best_perf);
    fprintf('RMSE на навчальній ділянці: %.4f\n', rmse_tr);
    fprintf('RMSE прогнозу: %.4f\n', rmse_pr);
    fprintf('MAE прогнозу: %.4f, max |err|: %.4f\n', mean(abs(err(f0:end))), max(abs(err(f0:end))));
    
    % Друкуємо таблицю порівняння в командне вікно з кроком через 10 точок
    disp('        x        true        pred         err');
    disp([xp(f0:10:end)' targ_p(f0:10:end)' OUT1(f0:10:end)' err(f0:10:end)']);
    
    % Будуємо підсумкове вікно з двома графіками
    figure;
    subplot(2, 1, 1);  % Верхній графік: накладаємо прогноз мережі на реальну функцію
    plot(xp, targ_p, '-o', xp, OUT1); grid on;
    xlabel('x'); ylabel('F(x)');
    legend('Original value', 'Predicted value');
    title(sprintf('Forecasting (rec = %d, Structure: [%s])', rec, num2str(Net_stru)));
    
    subplot(2, 1, 2);  % Нижній графік: показуємо, де саме і наскільки сильно мережа помиляється
    plot(xp, err); grid on;
    xlabel('x'); ylabel('Error');
    title(sprintf('Standard error of prediction = %.4f', rmse_pr));
    
    % Питаємо, чи повторити запуск з новими параметрами, чи завершити роботу
    k = input('Do you want quit a program? [No - Enter] [Yes - Any key]: ', 's');
    Out_proc = 1;
    if isempty(k), Out_proc = 0; end
end