% Load the data
data = readtable('Model_data.csv');

% Convert Screw_Configuration to categorical
data.Screw_Configuration = categorical(data.Screw_Configuration);

% Normalize continuous variables
data.Screw_speed_norm = (data.Screw_speed - mean(data.Screw_speed)) / std(data.Screw_speed);
data.Liquid_content_norm = (data.Liquid_content - mean(data.Liquid_content)) / std(data.Liquid_content);
data.Liquid_binder_norm = (data.Liquid_binder - mean(data.Liquid_binder)) / std(data.Liquid_binder);

% Prepare the data
X = [data.Screw_speed_norm, data.Liquid_content_norm, data.Liquid_binder_norm];
y_coverage = data.Seed_coverage;
y_number = data.number_seeded;

% Create dummy variables for Screw_Configuration
config_dummy = dummyvar(data.Screw_Configuration);
X = [X, config_dummy(:, 1:end-1)]; % Exclude one category to avoid multicollinearity

% Add interaction terms and quadratic terms
X = [X, X(:,1).*X(:,2), X(:,1).*X(:,3), X(:,2).*X(:,3), X(:,1).^2, X(:,2).^2, X(:,3).^2];

% Fit models using stepwise regression
mdl_coverage = stepwiselm(X, y_coverage, 'Upper', 'quadratic', 'Criterion', 'aic');
mdl_number = stepwiselm(X, y_number, 'Upper', 'quadratic', 'Criterion', 'aic');

% Display model information
disp('Coverage Model:');
disp(mdl_coverage);
disp('Number of Seeded Granules Model:');
disp(mdl_number);

% Function to predict using the model
function predictions = predict_model(model, new_data, data_means, data_stds, config_categories)
    X_new = [(new_data.Screw_speed - data_means(1)) / data_stds(1), ...
             (new_data.Liquid_content - data_means(2)) / data_stds(2), ...
             (new_data.Liquid_binder - data_means(3)) / data_stds(3)];
    
    % Add dummy variables for Screw_Configuration
    config_dummy = zeros(1, length(config_categories) - 1);
    config_index = find(config_categories == new_data.Screw_Configuration);
    if config_index < length(config_categories)
        config_dummy(config_index) = 1;
    end
    X_new = [X_new, config_dummy];
    
    % Add interaction terms and quadratic terms
    X_new = [X_new, X_new(1)*X_new(2), X_new(1)*X_new(3), X_new(2)*X_new(3), X_new(1)^2, X_new(2)^2, X_new(3)^2];
    
    predictions = predict(model, X_new);
    predictions = max(0, predictions); % Ensure non-negative predictions
    if strcmp(model.ResponseName, 'y_coverage')
        predictions = min(100, predictions); % Ensure coverage doesn't exceed 100%
    end
end

% Calculate means and standard deviations for normalization
data_means = [mean(data.Screw_speed), mean(data.Liquid_content), mean(data.Liquid_binder)];
data_stds = [std(data.Screw_speed), std(data.Liquid_content), std(data.Liquid_binder)];
config_categories = categories(data.Screw_Configuration);

% Generate 3D surface plots
figure;

% Coverage plot
subplot(1,2,1);
[X,Y] = meshgrid(linspace(min(data.Liquid_content), max(data.Liquid_content), 50), ...
                 linspace(min(data.Liquid_binder), max(data.Liquid_binder), 50));
Z = zeros(size(X));
for i = 1:numel(X)
    new_data = table(300, X(i), Y(i), categorical({'LS'}), ...
        'VariableNames', {'Screw_speed', 'Liquid_content', 'Liquid_binder', 'Screw_Configuration'});
    Z(i) = predict_model(mdl_coverage, new_data, data_means, data_stds, config_categories);
end
surf(X, Y, Z);
xlabel('Liquid Content');
ylabel('Liquid Binder (w/w%)');
zlabel('Seed Coverage (%)');
title('Predicted Seed Coverage');

% Number of seeded granules plot
subplot(1,2,2);
Z = zeros(size(X));
for i = 1:numel(X)
    new_data = table(300, X(i), Y(i), categorical({'LS'}), ...
        'VariableNames', {'Screw_speed', 'Liquid_content', 'Liquid_binder', 'Screw_Configuration'});
    Z(i) = predict_model(mdl_number, new_data, data_means, data_stds, config_categories);
end
surf(X, Y, Z);
xlabel('Liquid Content');
ylabel('Liquid Binder (w/w%)');
zlabel('Number of Seeded Granules');
title('Predicted Number of Seeded Granules');

% Interactive prediction script
while true
    disp('Enter parameters for prediction (or type "exit" to quit):');
    
    % Input for Liquid Content
    input_str = input('Liquid content (or type "exit" to quit): ', 's');
    if strcmpi(input_str, 'exit')
        break;
    end
    liquid_content = str2double(input_str);
    if isnan(liquid_content)
        disp('Invalid input. Please enter a number for liquid content.');
        continue;
    end
    
    % Input for Liquid Binder
    input_str = input('Liquid binder (w/w%): ', 's');
    liquid_binder = str2double(input_str);
    if isnan(liquid_binder)
        disp('Invalid input. Please enter a number for liquid binder.');
        continue;
    end
    
    % Input for Screw Speed
    input_str = input('Screw speed (rpm): ', 's');
    screw_speed = str2double(input_str);
    if isnan(screw_speed)
        disp('Invalid input. Please enter a number for screw speed.');
        continue;
    end
    
    % Input for Screw Configuration
    screw_config = input('Screw configuration (LS, MS, HSS, or HSL): ', 's');
    if isempty(screw_config) || ~ismember(screw_config, {'LS', 'MS', 'HSS', 'HSL'})
        disp('Invalid input. Please enter a valid screw configuration (LS, MS, HSS, or HSL).');
        continue;
    end

    % Predict seed coverage and number of seeded granules
    new_data = table(screw_speed, liquid_content, liquid_binder, categorical({screw_config}), ...
        'VariableNames', {'Screw_speed', 'Liquid_content', 'Liquid_binder', 'Screw_Configuration'});

    predicted_coverage = predict_model(mdl_coverage, new_data, data_means, data_stds, config_categories);
    predicted_number = predict_model(mdl_number, new_data, data_means, data_stds, config_categories);

    % Calculate probability based on conditions
    expected_seeds = interp1([100, 300, 500], [18, 20, 13], screw_speed);
    prob_number = (predicted_number >= 0.85 * expected_seeds) && (predicted_number <= 1.15 * expected_seeds);
    prob_coverage = predicted_coverage >= 40;

    probability = mean([prob_number, prob_coverage]);

    % Display results
    fprintf('\nPredicted Seed Coverage: %.2f%%\n', predicted_coverage);
    fprintf('Predicted Number of Seeded Granules: %.2f\n', predicted_number);
    fprintf('Probability of Producing Seeded Granules: %.2f\n\n', probability);
end

disp('Exiting the prediction script. Thank you for using the model!');
