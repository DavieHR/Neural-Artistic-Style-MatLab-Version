function [feat_cont, feat_style, content, style,scale] = compute_features(net, img, opts)
% feats = compute_features(net, img) - compute features from a network
%
%  Inputs: net - the neural network (same as for vl_simplenn)
%          img - the input image (H x W x 3 - RGB image)
%
%  Output: feats - the reference given as argument to invert_nn.m
%
% Author: Aravindh Mahendran
%      New College, University of Oxford

% normalize the input image
content = [];
style = [];
cnt_cont = 1;
cnt_sty = 1;
head = 1;
for i = 1:numel(net.layers)
    if strcmp(net.layers{i}.name,opts.layer.content)
        content(cnt_cont) = i;
    end
    for j = head:numel(opts.layer.style)
        if strcmp(net.layers{i}.name,opts.layer.style{j})
        style(cnt_sty) = i;
        cnt_sty = cnt_sty + 1;
        head = head + 1;
        end
    end
end
x0_target = opts.normalize(img.target);
x0_source = opts.normalize(img.source);
scale = size(img.target);
% Convert the image into a 4D matrix as required by vl_simplenn
%x0 = repmat(x0, [1, 1, 1, 1]);
net_gpu = vl_simplenn_move(net, 'gpu');
% Run feedforward for network
res = vl_simplenn(net_gpu, gpuArray(x0_target));
res_ = vl_simplenn(net_gpu, gpuArray(x0_source));
feat_cont = res(content).x;
feat_style = {};
for i =1:numel(style)
feat_style{i} = res_(style(i)).x;
end
end
