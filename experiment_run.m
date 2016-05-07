function experiment_run(exp)
if matlabpool('size') > 0
  parfor i=1:numel(exp) % Easily run lots of experiments on a cluster
    run_one(exp{i}) ;
  end
else
  for i=1:numel(exp)
    ts = tic;
    fprintf(1, 'Starting an expeirment');
    run_one(exp{i}) ;
    fprintf(1, 'done an expeirment');
    toc(ts);
  end
end
end
function run_one(exp)
% -------------------------------------------------------------------------

 expPath = fullfile(exp.expDir, exp.name) ;
 expName = exp.name;%sprintf('l%02d', exp.layer) ;
 if ~exist(expPath, 'dir'), mkdir(expPath) ; end
 %if exist(fullfile(expPath, [expName '.mat'])), return ; end

%fprintf('running experiment %s\n', exp.name) ;

% read image
im.target = imread(exp.path_target);
im.source = imread(exp.path_source);
%if size(im,3) == 1, im = cat(3,im,im,im) ; end
switch exp.model
  case 'caffe-vgg16'
    net = load('network/imagenet-vgg-verydeep-16.mat') ;
    exp.opts.normalize = get_cnn_normalize(net.normalization) ;
    exp.opts.denormalize = get_cnn_denormalize(net.normalization) ;
    exp.opts.imgSize = net.normalization.imageSize;
    exp.opts.layer.content = 'relu2_2';
    exp.opts.layer.style = {'relu1_1','relu2_1','relu3_1','relu4_1','relu5_1'};
  case 'caffe-vgg19'
    net = load('networks/places-caffe-ref-upgraded.mat');
    exp.opts.normalize = get_cnn_normalize(net.meta.normalization) ;
    exp.opts.denormalize = get_cnn_denormalize(net.meta.normalization) ;
    exp.opts.imgSize = net.meta.normalization.imageSize;
    exp.opts.layer.content = 'relu5_2';
    exp,opts.layer.style = 'relu1_1,relu2_1,relu3_1,relu4_1,relu5_1';
  case 'caffe-alex'
    net = load('networks/imagenet-caffe-alex.mat') ;
    exp.opts.normalize = get_cnn_normalize(net.meta.normalization) ;
    exp.opts.denormalize = get_cnn_denormalize(net.meta.normalization) ;
    exp.opts.imgSize = net.meta.normalization.imageSize;
end
net = vl_simplenn_tidy(net);
if isinf(exp.layer), exp.layer = numel(net.layers) ; end
net.layers = net.layers(1:37) ;
pool = {['pool1'],['pool2'],['pool3'],['pool4'],['pool5']};
head =1;
for i = 1:37
    for j =head:5
    if strcmp(net.layers{i}.name,pool{j})
        head = head +1;
      net.layers{i}.method = 'avg';
    end
    end
end
[feats_target, feats_source,content,style,scale] = compute_features(net, im, exp.opts);
exp.opts.scale = scale;
exp.opts.resour = feats_source;
exp.opts.target = feats_target;
exp.opts.content = content;
exp.opts.style = style;


% run experiment
%args = expandOpts(exp.opts) ;
if(strcmp(exp.opts.optim_method , 'gradient-descent'))
  res = style_content(net,exp.opts) ;
else
  fprintf(1, 'Unknown optimization method %s\n', exp.opts.optim_method);
  return;
end

if isfield(net.layers{end}, 'name')
  res.name = net.layers{end}.name ;
else;
  res.name = sprintf('%s%d', net.layers{end}.type, exp.layer) ;
end

% save images
if size(res.output{end},4) > 1
  im = vl_imarraysc(res.output{end}) ;
else
  im = vl_imsc(res.output{end}) ;
end
vl_printsize(1) ;
print('-dpdf',  fullfile(expPath, [expName '-opt.pdf'])) ;
%imwrite(input / 255, fullfile(expPath, [expName '-orig.png'])) ;
imwrite(im, fullfile(expPath, [expName '-recon.png'])) ;

% save videos
makeMovieFromCell(fullfile(expPath, [expName '-evolution']), res.output) ;
makeMovieFromArray(fullfile(expPath, [expName '-recon']), res.output{end}) ;

% save results
res.output = res.output{end} ; % too much disk space
save(fullfile(expPath, [expName '.mat']), '-struct', 'res');
end


% -------------------------------------------------------------------------
function args = expandOpts(opts)
% -------------------------------------------------------------------------
args = horzcat(fieldnames(opts), struct2cell(opts))' ;
end

% -------------------------------------------------------------------------
function makeMovieFromCell(moviePath, x)
% -------------------------------------------------------------------------
writerObj = VideoWriter(moviePath,'Motion JPEG AVI');
open(writerObj) ;
for k = 1:numel(x)
  if size(x{k},4) > 1
    im = vl_imarraysc(x{k}) ;
  else
    im = vl_imsc(x{k}) ;
  end
  writeVideo(writerObj,im2frame(double(im)));
end
close(writerObj);
end

% -------------------------------------------------------------------------
function makeMovieFromArray(moviePath, x)
% -------------------------------------------------------------------------
writerObj = VideoWriter(moviePath,'Motion JPEG AVI');
open(writerObj) ;
for k = 1:size(x,4)
  im = vl_imsc(x(:,:,:,k)) ;
  writeVideo(writerObj,im2frame(double(im)));
end
close(writerObj);
end