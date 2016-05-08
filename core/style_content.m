function res = style_content(net,opts)
%opts.ll = 0;
%[opts, varargin] = vl_argparse(opts, varargin) ;
scale = opts.scale;
opts.momentum = 0.9;
resource = opts.resour;
target = opts.target;
imSize = [224,224,3]; 
layer_of_content = opts.content;
layer_of_styles = opts.style;
weight_of_content = opts.content_weight;
weight_of_styles = opts.style_weight;
MaxIter = opts.MaxIter;
x_momentum = 0;
lr =  opts.learningRate;
x0_size = imSize;
x = randn(x0_size,'single');
x = x / norm(x(:));
x_momentum = zeros(x0_size,'single');
layer_num = numel(net.layers) ;
net_tmp.layers = {};
net_tmp2.layers = {};
net_tmp2.meta = net.meta;
net_group = {};
switch opts.objective
  case 'l2'
    % Add the l2 loss over the network
    for i =1:layer_of_content
    net_tmp.layers{end+1}=net.layers{i};
    end
    ly.type = 'custom' ;
    ly.wt = target ;
    ly.weight_of_content = weight_of_content;
   % ly.mask = mask ;
    ly.precious = 0;
    ly.forward = @nndistance_forward ;
    ly.backward = @nndistance_backward ;
    net_tmp.layers{end+1} = ly ;
    net_tmp.meta = net.meta;
    net_gpu = vl_simplenn_move(net_tmp, 'gpu');
    net_group{1}=net_gpu;
    %clear 'net_cpu';
    clear net_tmp;
    clear ly;
    net_tmp.layers = {};
    count = 2;
    for j = 1:numel(layer_of_styles)
       for i =1:layer_of_styles(j)
        net_tmp.layers{end+1}=net.layers{i};
       end
        ly.type = 'custom';
        ly.wt = resource{j};
        ly.weight_of_styles = weight_of_styles;
    %    ly.mask = mask;
    %ly.layers = layer_of_styles;
        ly.forward = @ssdistance_forward;
        ly.backward = @ssdistance_backward;
        %ly.precious = 0;
        net_tmp.layers{end+1} = ly;
        net_tmp.meta = net.meta;
        net_gpu = vl_simplenn_move(net_tmp, 'gpu');
        net_group{count} = net_gpu;
        count = count+1;
        clear net_tmp;
        net_tmp.layers = {};
    end 
  case 'l1'
    % The L1 loss might want to use a dropout layer. 
    % This is just a guess and hasn't been tried.
    ly.type = 'dropout' ;
    ly.rate = opts.dropout ;
    net.layers{end+1} = ly ;
    ly.type = 'custom' ;
    ly.w = y0 ;
    ly.mask = mask ;
    ly.forward = @nndistance1_forward ;
    ly.backward = @nndistance1_backward ;
    net.layers{end+1} = ly ;
  case 'inner'
    % The inner product loss may be suitable for some networks
    ly.type = 'custom' ;
    ly.w = - y0 .* mask ;
    ly.forward = @nninner_forward ;
    ly.backward = @nninner_backward ;
    net.layers{end+1} = ly ;
  otherwise
    error('unknown opts.objective') ;
end
output = {} ;
%res = vl_simplenn(net, x);
res ={};
for t=1:MaxIter
 x=gpuArray(x);
 % opts.backPropDepth = layer_of_content;
  for i =1 :numel(layer_of_styles)+1,
 % res{i} = vl_simplenn(net_group{i}, x);
  res{i} = vl_simplenn(net_group{i},x, single(1));
  end
  dr = zeros(size(x),'single');
  Dr2 = zeros(size(x),'single');
  E(1,t) = gather(res{1}(end).x);
  Dr2 = gather(res{1}(1).dzdx);
  Sum  = 0;
  sum_ = 0;
  for j = 2:numel(layer_of_styles)+1,
  Sum = Sum + gather(res{j}(end).x);
  sum_ = sum_ + gather(res{j}(1).dzdx);
  end
  Dr2 = Dr2 + sum_;
  E(2,t) = Sum;
  x = gather(x);
  if opts.lambdaTV > 0 % Cost and derivative for TV\beta norm
    [r_,dr_] = tv(x,opts.TVbeta) ;
    E(3,t) = opts.lambdaTV/2 * r_ ;
    dr = dr + opts.lambdaTV/2 * dr_ ;
  else
    E(3,t) = 0 ;
  end
  E(4,t) = sum(E(1:3,t));   
  x_momentum = opts.momentum * x_momentum ...
      - lr * dr ...
      - lr * Dr2;
  x = x + x_momentum ;
   if mod(t-1,5)==0
    output{end+1} = imresize(opts.denormalize(x), scale(1:2),'bilinear');
    figure(1) ; clf ;
%    subplot(3,2,[1 3]) ;
    if opts.numRepeats > 1
      vl_imarraysc(output{end}) ;
    else
      imagesc(vl_imsc(output{end}),[2,3]) ;
    end
    axis image ; colormap hsv;
%    subplot(3,2,2) ;
%     len = min(1000, numel(y0));
%     a = squeeze(y0(1:len)) ;
%     b = squeeze(y(1:len)) ;
%     plot(1:len,a,'b'); hold on ;
%     plot(len+1:2*len,abs(b-a), 'r');
%     legend('\Phi_0', '|\Phi-\Phi_0|') ;
%     title(sprintf('reconstructed layer %d %s', ...
%       layer_num, ...
%       net.layers{layer_num}.type)) ;
%     legend('ref', 'delta') ;
%     subplot(3,2,4) ;
%     hist(x(:),100) ;
%     grid on ;
%     title('histogram of x') ;
%     subplot(3,2,5) ;
%     plot(E') ;
%     h = legend('recon', 'tv_reg', 'l2_reg', 'tot') ;
%     set(h,'color','none') ; grid on ;
%     title(sprintf('iter:%d \\lambda_{tv}:%g \\lambda_{l2}:%g rate:%g obj:%s', ...
%                   t, opts.lambdaTV, opts.lambdaL2, lr, opts.objective)) ;
% 
%     subplot(3,2,6) ;
%     semilogy(E') ;
%     title('log scale') ;
%     grid on ;
  drawnow ;
  end % end if(mod(t-1,25) == 0)
 disp(t);
end

%res_nn = vl_simplenn(net, x);

clear res;
res.input = NaN;
res.output = output ;
res.energy = E ;
res.y0.resourse = resource ;
res.y0.target = target;
%res.y = res_nn(end-1).x ;
res.opts = opts;
%res.err = res_nn(end).x;

function [e, dx] = tv(x,beta)
% --------------------------------------------------------------------
if(~exist('beta', 'var'))
  beta = 1; % the power to which the TV norm is raized
end
d1 = x(:,[2:end end],:,:) - x ;
d2 = x([2:end end],:,:,:) - x ;
v = sqrt(d1.*d1 + d2.*d2).^beta ;
e = sum(sum(sum(sum(v)))) ;
if nargout > 1
  d1_ = (max(v, 1e-5).^(2*(beta/2-1)/beta)) .* d1;
  d2_ = (max(v, 1e-5).^(2*(beta/2-1)/beta)) .* d2;
  d11 = d1_(:,[1 1:end-1],:,:) - d1_ ;
  d22 = d2_([1 1:end-1],:,:,:) - d2_ ;
  d11(:,1,:,:) = - d1_(:,1,:,:) ;
  d22(1,:,:,:) = - d2_(1,:,:,:) ;
  dx = beta*(d11 + d22);
  if(any(isnan(dx)))
  end
end

% --------------------------------------------------------------------
function test_tv()
% --------------------------------------------------------------------
x = randn(5,6,1,1) ;
[e,dr] = tv(x,6) ;
vl_testder(@(x) tv(x,6), x, 1, dr, 1e-3) ;

function res_ = nndistance_forward(ly,res,res_)
   res_.x = nndistance(res.x,ly.wt,ly.weight_of_content);
   
function  res = nndistance_backward (ly,res,res_)
   res.dzdx = nndistance(res.x,ly.wt,ly.weight_of_content,res_.dzdx);

function  y = nndistance(x,p,wt,dzdy)
       d = wt.*(x - p);
       if nargin==3
         y = sum(sum(sum(d.*d)))/2.0;
       elseif nargin==4
         dt = wt.*(x - p);
         y = dzdy*dt;
       end
       


function  res_ = ssdistance_forward(ly,res,res_)
res_.x = ssdistance(res.x,ly.wt,ly.weight_of_styles);

function  res = ssdistance_backward(ly,res,res_)
res.dzdx = ssdistance(res.x,ly.wt,ly.weight_of_styles,res_.dzdx);

function y = ssdistance(x,p,wt,dzdy)
 if nargin == 3
      tmp_t = reshape(x,size(x,1)*size(x,2),size(x,3));
      tmp_r = reshape(p,size(p,1)*size(p,2),size(p,3));
      g = tmp_t'*tmp_t;%grammatrix(tmp_t);
      a_ = tmp_r'*tmp_r;%grammatrix(tmp_r);
      Nl = size(x,3);
      Ml = size(x,1)*size(x,2);
      E = 1/(4*Nl^2*Ml^2)*sum(sum((g - a_).*(g - a_)));
      y = wt*E;
 else 
      tmp_t = reshape(x,size(x,1)*size(x,2),size(x,3));
      tmp_r = reshape(p,size(p,1)*size(p,2),size(p,3));
      g = tmp_t'*tmp_t;%grammatrix(tmp_t);
      a_ = tmp_r'*tmp_r;%grammatrix(tmp_r);
      Nl = size(x,3);
      Ml = size(x,1)*size(x,2);
      tmp = gpuArray(zeros(size(x,1),size(x,2),size(g-a_,1)));
      deta = g-a_;
   
%       for tt = 1:size(g-a_,1)
%         for tj = 1:size(g-a_,2)
%         tmp(:,:,tt) = x(:,:,tt)*deta(tt,tj);
%         end
%       end
      tmp = tmp_t*deta;
      tmp =reshape(tmp,size(x));
      dE = 1/(Nl^2*Ml^2)*tmp;
      y = wt*dE*dzdy;
 end
%  
% function  g = grammatrix(x)
%     g = gpuArray(zeros(size(x,2),size(x,2)));
%     for i=1:size(x,2)
%        for j = 1:size(x,2)
%          g(i,j)=x(:,i)'*x(:,j) ;  
%        end
%     end
         
         
         
         