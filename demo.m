function demo()
% Run some CNN experiments

%experiment_setup;

images_target = ['data\images\target\fan_NoTV.jpg'];
images_source = ['data\images\source\paprika.jpg'];
% -------------------------------------------------------------------------
%                                                         Setup experiments
% -------------------------------------------------------------------------

exp = {} ;
ver = 'results' ;
% opts.learningRate = 0.004 * [...
%   ones(1,200), ...
%   0.1 * ones(1,200), ...
%   0.01 * ones(1,200),...
%   0.001 * ones(1,200)];
% opts.objective = 'l2' ;
% opts.beta = 6 ;
% opts.lambdaTV = 1e2 ;
% opts.lambdaL2 = 8e-10 ;
 opts.numRepeats = 1;

% opts1 = opts;
% opts1.lambdaTV = 1e0 ;
% opts1.TVbeta = 2;
% 
% opts2 = opts;
% opts2.lambdaTV = 1e1 ;
% opts2.TVbeta = 2;
% 
% opts3 = opts;
% opts3.lambdaTV = 1e2 ;
% opts3.TVbeta = 2;

% -------------------------------------------------------------------------
%                                                           Run experiments
% -------------------------------------------------------------------------

%for i = 1:numel(images) 
  exp{end+1} = experiment_init('caffe-vgg16', inf, images_target, images_source,opts);
%   exp{end+1} = experiment_init('hog', inf, images{i}, ver, 'hog2', opts2);
%   exp{end+1} = experiment_init('hog', inf, images{i}, ver, 'hog3', opts3);
% 
%   exp{end+1} = experiment_init('hog', inf, images{i}, ver, 'hoggle', opts1, 'useHoggle', true);
%   exp{end+1} = experiment_init('dsift', inf, images{i}, ver, 'dsift', opts1);
%   exp{end+1} = experiment_init('hogb', inf, images{i}, ver, 'hogb', opts1);
%end

experiment_run(exp) ;